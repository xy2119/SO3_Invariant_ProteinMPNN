from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                # x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]

                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                # mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            elif letter in masked_chains: 
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]               
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))
                    
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict!=None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx+v[0][v_count]-1)#make 0 to be the first
                                tied_beta[start_idx+v[0][v_count]-1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx+v_-1)#make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)


 
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list,0) #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list, 0)  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))

        pssm_coef_all[i,:] = pssm_coef_pad
        pssm_bias_all[i,:] = pssm_bias_pad
        pssm_log_odds_all[i,:] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[i,:] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)


    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:,:,0]
    else:
        X_out = X
    return X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta



def tied_featurize(batch, device, chain_dict, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None, bias_by_res_dict=None, ca_only=False):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0*np.ones([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []

    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order 

    
    
    # for i, b in enumerate(batch):
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []

        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                #print('letter in visible_chains:',letter)
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                #print('visible chain_seq',chain_seq)
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            elif letter in masked_chains:
                #print('letter in masked_chains:',letter)
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                #print('masked chain_seq',chain_seq)
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]               
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

       
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict!=None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx+v[0][v_count]-1)#make 0 to be the first
                                tied_beta[start_idx+v[0][v_count]-1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx+v_-1)#make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)


 
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list,0) #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list, 0)  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        #print('l = len(all_sequence):',l)

        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad
        #print('x_pad.shape',x_pad.shape)
        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        #print('m_pad.shape',m_pad.shape)
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        #print('m_pos_pad.shape:',m_pos_pad.shape)
        #print('omit_AA_mask_list len:',len(omit_AA_mask_list))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        #print('omit_AA_mask_pad.shape:',omit_AA_mask_pad.shape)
        #print('omit_AA_mask.shape:',omit_AA_mask.shape)
        
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,]=omit_AA_mask_pad[:,:21]

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))

        pssm_coef_all[i,:] = pssm_coef_pad
        pssm_bias_all[i,:] = pssm_bias_pad
        pssm_log_odds_all[i,:] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[i,:] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)


    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:,:,0]
    else:
        X_out = X
    return X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta



def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class CA_ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(CA_ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf*9 + 7
        self.node_embedding = nn.Linear(node_in,  node_features, bias=False) #NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)


    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        dX = X[:,1:,:] - X[:,:-1,:]
        dX_norm = torch.norm(dX,dim=-1)
        dX_mask = (3.6<dX_norm) & (dX_norm<4.0) #exclude CA-CA jumps
        dX = dX*dX_mask[:,:,None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)
        
        return AD_features, O_features



    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        print('X shape: ',X.shape)
        print('mask shape:', mask.shape)
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        print('mask2D shape:', mask_2D.shape)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        print('dX shape:', dX.shape)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:,1:,:] = Ca[:,:-1,:]
        Ca_1 = Ca
        Ca_2[:,:-1,:] = Ca[:,1:,:]

        V, O_features = self._orientations_coarse(Ca, E_idx)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx)) 
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))


        RBF_all = torch.cat(tuple(RBF_all), dim=-1)


        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        #E = torch.cat((E_positional, RBF_all,), -1)
        E = torch.cat((E_positional, RBF_all, O_features), -1)
        print('E shape :',E.shape)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        return E, E_idx 
    
class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs



    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def tied_sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(X.shape[0],1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:,t]==0).all():
                    S_t = S_true[:,t]
                    for t in t_list:
                        h_S[:,t,:] = self.W_s(S_t)
                        S[:,t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:,t:t+1,:]
                    h_E_t = h_E[:,t:t+1,:,:]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
                    mask_t = mask[:,t:t+1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:,t:t+1,:]
                        h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:,t,:]
                    logit_list.append((self.W_out(h_V_t) / temperature)/len(t_list))
                    logits += tied_beta[t]*(self.W_out(h_V_t) / temperature)/len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:,t,:] #[B, 21]
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:,t]
                    pssm_bias_gathered = pssm_bias[:,t]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:,t]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:,t]*S_t_repeat + (1-chain_mask[:,t])*S_true[:,t]).long() #hard pick fixed positions
                for t in t_list:
                    h_S[:,t,:] = self.W_s(S_t_repeat)
                    S[:,t] = S_t_repeat
                    all_probs[:,t,:] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
  
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0,:]==1)[:,0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:,idx,:] = log_probs[:,idx,:]
        return log_conditional_probs


    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    
class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx 

import math
import torch
from scipy.special import sph_harm

class CA_SH_ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(CA_SH_ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf*9 + 3
        self.node_embedding = nn.Linear(node_in,  node_features, bias=False) #NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)



    def _Y_lm(self,l, m, theta, phi):

        prefactor = torch.sqrt(torch.tensor((2*l+1)/(4*math.pi))) * torch.sqrt(torch.tensor(math.perm(l-m)/math.perm(l+m)))
        
        # calculate the value of the spherical harmonic
        harmonic = sph_harm(m, l, theta, phi)

        if m < 0:
            result = prefactor * torch.sin(-m*phi) * harmonic
        elif m == 0:
            result = prefactor * harmonic
        else:
            result = prefactor * torch.cos(m*phi) * harmonic

        result = prefactor * torch.cos(m*phi) * harmonic

        return result

    def _reduce_dimensions_with_spherical_harmonics(self, coordinates, l_max):
        num_coordinates = coordinates.shape[0] * coordinates.shape[1]

        """r = torch.norm(coordinates.float(), dim=2).reshape((num_coordinates,))
        theta = torch.acos(coordinates[:,:,2].reshape((num_coordinates,)) / r)
        phi = torch.atan2(coordinates[:,:,1].reshape((num_coordinates,)), coordinates[:,:,0].reshape((num_coordinates,)))"""
        
        r = np.linalg.norm(coordinates.cpu(), axis=2).reshape((num_coordinates,))
        theta = np.arccos(coordinates.cpu()[:,:,2].reshape((num_coordinates,)) / r)
        phi = np.arctan2(coordinates.cpu()[:,:,1].reshape((num_coordinates,)), coordinates.cpu()[:,:,0].reshape((num_coordinates,)))

        expand_coef = torch.empty((len(theta), l_max+1))
        expand_matrix = torch.tensor(np.zeros((len(theta), (l_max+1)**2),dtype=complex))
         

        for l in range(l_max+1):
            for m in range(-l, l+1):
                # expansion coefficient for the given l and m
                coef=self._Y_lm(l, m, theta, phi)
                expand_coef[:,l] +=  (coef.real**2+coef.imag**2).clone().detach()
                expand_matrix[:,l**2+l+m] =  coef.clone().detach()
        
        l2_norm = torch.sqrt(expand_coef).reshape((coordinates.shape[0],coordinates.shape[1],(l_max+1)))
        expand_matrix = expand_matrix.reshape((coordinates.shape[0],coordinates.shape[1],(l_max+1)**2))
        return expand_matrix.real,l2_norm




    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """

        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:,1:,:] = Ca[:,:-1,:]
        Ca_1 = Ca
        Ca_2[:,:-1,:] = Ca[:,1:,:]

        
        
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx)) 
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))


        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        m, SH_features = self._reduce_dimensions_with_spherical_harmonics(Ca, l_max=2)
        where_nan = torch.isnan(SH_features)
        SH_features[where_nan] = 0 
        SH_neighbors = gather_nodes(SH_features.to(Ca.device), E_idx)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)

        SH_neighbors = gather_nodes(SH_features.to(Ca.device), E_idx)

        print('E_positional, RBF_all, SH_neighbors shape:',E_positional.shape, RBF_all.shape, SH_neighbors.shape)
        print('SH_features:', SH_features)
        print('SH_neighbors:', SH_neighbors)
        
        E = torch.cat((E_positional, RBF_all, SH_neighbors), -1)     

        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        return E, E_idx 
    
class SH_ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super(SH_ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_SH_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs



    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def tied_sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(X.shape[0],1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:,t]==0).all():
                    S_t = S_true[:,t]
                    for t in t_list:
                        h_S[:,t,:] = self.W_s(S_t)
                        S[:,t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:,t:t+1,:]
                    h_E_t = h_E[:,t:t+1,:,:]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
                    mask_t = mask[:,t:t+1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:,t:t+1,:]
                        h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:,t,:]
                    logit_list.append((self.W_out(h_V_t) / temperature)/len(t_list))
                    logits += tied_beta[t]*(self.W_out(h_V_t) / temperature)/len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:,t,:] #[B, 21]
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:,t]
                    pssm_bias_gathered = pssm_bias[:,t]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:,t]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:,t]*S_t_repeat + (1-chain_mask[:,t])*S_true[:,t]).long() #hard pick fixed positions
                for t in t_list:
                    h_S[:,t,:] = self.W_s(S_t_repeat)
                    S[:,t] = S_t_repeat
                    all_probs[:,t,:] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
  
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0,:]==1)[:,0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:,idx,:] = log_probs[:,idx,:]
        return log_conditional_probs


    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

