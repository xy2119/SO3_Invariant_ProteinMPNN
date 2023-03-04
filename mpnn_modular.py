import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json, time, os, sys, glob, re
import numpy as np
if not os.path.isdir("SO3_Invariant_ProteinMPNN"):
    os.system("git clone -q https://github.com/xy2119/SO3_Invariant_ProteinMPNN.git")
    
if "SO3_Invariant_ProteinMPNN" not in sys.path:
    sys.path.append('./SO3_Invariant_ProteinMPNN')


#@title Setup ProteinMPNN
def run_mpnn(pdb, #@param {type:"string"}
             
             
            ):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.io as pio
    import shutil
    import warnings
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB,parse_fasta
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
    
    #helper funciton
    def make_tied_positions_for_homomers(pdb_dict_list):
        my_dict = {}
        for result in pdb_dict_list:
            all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
            tied_positions_list = []
            chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
            for i in range(1,chain_length+1):
                temp_dict = {}
                for j, chain in enumerate(all_chain_list):
                    temp_dict[chain] = [i] #needs to be a list
                tied_positions_list.append(temp_dict)
            my_dict[result['name']] = tied_positions_list
        return my_dict

    
    def get_pdb(pdb_code=""):
        if not os.path.exists("./pdb"):
            os.makedirs("./pdb")

        if pdb_code is None or pdb_code == "":
            upload_dict = files.upload()
            pdb_string = upload_dict[list(upload_dict.keys())[0]]
            filename = "./pdb/tmp.pdb"  # modify the file path to include './pdb'
            with open(filename, "wb") as out: 
                out.write(pdb_string)
                return filename
        else:
            filename = f"./pdb/{pdb_code}.pdb"  # modify the file path to include './pdb'
            os.system(f"wget -qnc -O {filename} https://files.rcsb.org/view/{pdb_code}.pdb")
            return filename
 

    def plot_conditional_probs_heatmap(all_log_probs_concat, alphabet, path):
        fig = px.imshow(np.exp(all_log_probs_concat).mean(0).T,
                        labels=dict(x="positions", y="amino acids", color="probability"),
                        y=list(alphabet),
                        template="simple_white")
        fig.update_xaxes(side="bottom")
        fig.update_layout(title=dict(text="Mean Probability of Amino Acids by Position (Conditional Probabilities)", y=0.95),
                          xaxis_title="positions")
        pio.write_html(fig, file=os.path.join(path, "log_probs.html"), auto_open=False)
        fig.show()

    def plot_sampled_probs_heatmap(all_probs_concat, alphabet, path):
        fig = px.imshow(all_probs_concat.mean(0).T,
                        labels=dict(x="positions", y="amino acids", color="probability"),
                        y=list(alphabet),
                        template="simple_white")
        fig.update_xaxes(side="bottom")
        fig.update_layout(title=dict(text="Mean Probability of Amino Acids by Position (Sampled)", y=0.95),
                          xaxis_title="positions")
        pio.write_html(fig, file=os.path.join(path, "probs.html"), auto_open=False)
        fig.show()
    print('ProteinMPNN setup all done!')
    
 
    # ProteinMPNN predict

    ## Input Options
    pdb = pdb.replace(" ","").split(",")
    pdb_list = [pro.strip() for pro in pdb]

    homomer = True #@param {type:"boolean"}
    designed_chain = "A" #@param {type:"string"}
    fixed_chain = "" #@param {type:"string"}
    #@markdown - specified which chain(s) to design and which chain(s) to keep fixed. 
    #@markdown   Use comma:`A,B` to specifiy more than one chain

    omit_AAs='X'#@param {type:"string"}
    #@markdown - Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

    if designed_chain == "":
      designed_chain_list = []
    else:
      designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

    if fixed_chain == "":
      fixed_chain_list = []
    else:
      fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))

 
    #@markdown ## Design Options
    device = torch.device("cuda")
    #v_48_010=version with 48 edges 0.10A noise
    model_type = "benchmark_Ca_48_020" #@param ["benchmark_Ca_48_002", "benchmark_Ca_48_010", "benchmark_Ca_48_020","Ca_48_020_80e","Ca_48_020_150e","Ca_48_020_200e","Ca_SH_48_020_80e","Ca_SH_48_020_150e","Ca_SH_48_020_200e"]

    num_seqs = 8 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
    num_seq_per_target = num_seqs

    save_score=1                      # 0 for False, 1 for True; save score=-log_prob to npy files
    save_probs=1                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
    score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
    conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
    conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)
    unconditional_probs_only=0

    batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
    max_length=20000                  #@param {type:"integer"}
                                      #@markdown - Max sequence length
    backbone_noise=0.00               #@param ["0.00","0.10","0.20","0.25","0.30"]  
                                      #@markdown - Standard deviation of Gaussian noise to add to backbone atoms

    pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
    pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
    pssm_log_odds_flag=0               # 0 for False, 1 for True
    pssm_bias_flag=0                   # 0 for False, 1 for True



    if model_type[:5] == 'Ca_SH':
      path_to_model_weights='./SO3_Invariant_ProteinMPNN/ca_model_weights/Ca_SH_48_020'  
      model_name = model_type
      ca_only=True
      rsh_expand=True
    elif model_type[:2] == 'Ca':
      path_to_model_weights='./SO3_Invariant_ProteinMPNN/ca_model_weights/Ca_48_020'  
      model_name = model_type
      ca_only=True
      rsh_expand=False
    elif model_type[:12] == 'benchmark_Ca':
      model_name='v_'+model_type[-6:]
      path_to_model_weights='./SO3_Invariant_ProteinMPNN/ca_model_weights'  
      ca_only=True
      rsh_expand=False
    elif model_type[:17] == 'benchmark_vanilla':
      """tbc"""
      path_to_model_weights='./SO3_Invariant_ProteinMPNN/vanilla_model_weights'  
      ca_only=False
      rsh_expand=False
    else:
      """tbc"""
      path_to_model_weights='./SO3_Invariant_ProteinMPNN/vanilla_model_weights'  
      ca_only=False
      #rsh_expand=True
        
    print('model type:',model_type,'model name:',model_name)
    print('Ca level:',ca_only,'SH Expansion:',rsh_expand)

    
    out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
    jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
    path_to_fasta=''
    
    # Build paths for experiment
    base_folder = out_folder
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    if not os.path.exists(base_folder + model_name):
        os.makedirs(base_folder + model_name)

    if not os.path.exists(base_folder + model_name + '/seqs'):
        os.makedirs(base_folder + model_name + '/seqs')

    if save_score:
        if not os.path.exists(base_folder + model_name + '/scores'):
            os.makedirs(base_folder + model_name + '/scores')

    if score_only:
        if not os.path.exists(base_folder + model_name + '/score_only'):
            os.makedirs(base_folder + model_name + '/score_only')

    if conditional_probs_only:
        if not os.path.exists(base_folder + model_name + '/conditional_probs_only'):
            os.makedirs(base_folder + model_name + '/conditional_probs_only')

    if unconditional_probs_only:
        if not os.path.exists(base_folder + model_name + '/unconditional_probs_only'):
            os.makedirs(base_folder + model_name + '/unconditional_probs_only')

    if save_probs:
        if not os.path.exists(base_folder + model_name + '/probs'):
            os.makedirs(base_folder + model_name + '/probs') 

    if not os.path.exists(base_folder + model_name + '/output'):
        os.makedirs(base_folder + model_name + '/output')

    if not os.path.exists(base_folder + 'results'):
        os.makedirs(base_folder + 'results')
        
        
    hidden_dim = 128
    num_layers = 3 
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(checkpoint.keys())

    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'], ca_only=ca_only,rsh_expand=rsh_expand)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")

    sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]
    #@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.


    ##############################################################

    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))   
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    suppress_print = 1 
    print_all = suppress_print 

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))


    ###############################################################
    sequences = []
    info_list=[]
    for pdb in pdb_list:
      print('\n','-'*25,'Generating sequences','-'*25)
    
      pdb_path = get_pdb(pdb)
      pdb_dict_list = parse_PDB(pdb_path , input_chain_list=chain_list)
      dataset_test = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
      
      chain_id_dict = {}
      chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

      print(chain_id_dict)
      for chain in chain_list:
        l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
        print(f"Length of chain {chain} is {l}")

      if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
      else:
        tied_positions_dict = None

      #################################################################
      # Timing
      start_time = time.time()
      with torch.no_grad():
        for ix, protein in enumerate(dataset_test):
          score_list = []
          global_score_list = []
          all_probs_list = []
          all_log_probs_list = []
          S_sample_list = []
          batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
          X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,ca_only=True)
          pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
          ssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
          name_ = batch_clones[0]['name']

          if score_only:
              loop_c = 0 
              if path_to_fasta:
                  fasta_names, fasta_seqs = parse_fasta(path_to_fasta, omit=["/"])
                  loop_c = len(fasta_seqs)
              for fc in range(1+loop_c):
                  if fc == 0:
                      structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + f'_pdb'
                  else:
                      structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + f'_fasta_{fc}'              
                  
                  native_score_list = []
                  global_native_score_list = []
                  if fc > 0:
                      input_seq_length = len(fasta_seqs[fc-1])
                      S_input = torch.tensor([alphabet_dict[AA] for AA in fasta_seqs[fc-1]], device=device)[None,:].repeat(X.shape[0], 1)
                      S[:,:input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                  for j in range(NUM_BATCHES):
                      randn_1 = torch.randn(chain_M.shape, device=X.device)
                      log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                      mask_for_loss = mask*chain_M*chain_M_pos
                      scores = _scores(S, log_probs, mask_for_loss)
                      native_score = scores.cpu().data.numpy()
                      native_score_list.append(native_score)
                      global_scores = _scores(S, log_probs, mask)
                      global_native_score = global_scores.cpu().data.numpy()
                      global_native_score_list.append(global_native_score)
                      
                  native_score = np.concatenate(native_score_list, 0)
                  global_native_score = np.concatenate(global_native_score_list, 0)
                  ns_mean = native_score.mean()
                  ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                  ns_std = native_score.std()
                  ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)

                  global_ns_mean = global_native_score.mean()
                  global_ns_mean_print = np.format_float_positional(np.float32(global_ns_mean), unique=False, precision=4)
                  global_ns_std = global_native_score.std()
                  global_ns_std_print = np.format_float_positional(np.float32(global_ns_std), unique=False, precision=4)

                  ns_sample_size = native_score.shape[0]
                  seq_str = _S_to_seq(S[0,], chain_M[0,])
                  #np.savez(structure_sequence_score_file, score=native_score, global_score=global_native_score, S=S[0,].cpu().numpy(), seq_str=seq_str)
                  if print_all:
                      if fc == 0:
                          print(f'Score for {name_} from PDB, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
                      else:
                          print(f'Score for {name_}_{fc} from FASTA, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
                
          elif conditional_probs_only:
              if print_all:
                  print(f'Calculating conditional probabilities for {name_}')
              conditional_probs_only_file = base_folder + '/conditional_probs_only/' + batch_clones[0]['name']
              log_conditional_probs_list = []
              for j in range(NUM_BATCHES):
                  randn_1 = torch.randn(chain_M.shape, device=X.device)
                  log_conditional_probs = model.conditional_probs(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, conditional_probs_only_backbone)
                  log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
              concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
              mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
              np.savez(conditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(), mask=mask[0,].cpu().numpy(), design_mask=mask_out)
          elif unconditional_probs_only:
              if print_all:
                  print(f'Calculating sequence unconditional probabilities for {name_}')
              unconditional_probs_only_file = base_folder + '/unconditional_probs_only/' + batch_clones[0]['name']
              log_unconditional_probs_list = []
              for j in range(NUM_BATCHES):
                  log_unconditional_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
                  log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
              concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
              mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
              np.savez(unconditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(), mask=mask[0,].cpu().numpy(), design_mask=mask_out)

          else:
              randn_1 = torch.randn(chain_M.shape, device=X.device)
              log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
              mask_for_loss = mask*chain_M*chain_M_pos
              scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
              native_score = scores.cpu().data.numpy()
              global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
              global_native_score = global_scores.cpu().data.numpy()
 
              base_path= base_folder + model_type

              if not os.path.exists(base_path + f"/output/{name_}"):
                  os.makedirs(base_path + f"/output/{name_}")
              if not os.path.exists(base_path + f"/seqs/{name_}"):
                  os.makedirs(base_path + f"/seqs/{name_}")
              if not os.path.exists(base_path + f"/scores/{name_}"):
                  os.makedirs(base_path + f"/scores/{name_}")
              if not os.path.exists(base_path + f"/probs/{name_}"):
                  os.makedirs(base_path + f"/probs/{name_}")
                  
              ali_file = base_path + f'/seqs/{name_}/' + name_ + '.fa'
              score_file = base_path + f'/scores/{name_}/' + name_ + '.npz'
              probs_file = base_path + f'/probs/{name_}/' + name_ + '.npz'
              if print_all:
                  print(f'Generating sequences for: {name_}')
              t0 = time.time()
              with open(ali_file, 'w') as f:
                  for temp in temperatures:
                      for j in range(NUM_BATCHES): 

  
                          randn_2 = torch.randn(chain_M.shape, device=X.device)
                          if tied_positions_dict == None:
                              sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                              S_sample = sample_dict["S"] 
                          else:
                              sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all)
                              S_sample = sample_dict["S"]
                          
                          log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                          mask_for_loss = mask*chain_M*chain_M_pos
                          scores = _scores(S_sample, log_probs, mask_for_loss)
                          scores = scores.cpu().data.numpy()
                           
                          global_scores = _scores(S_sample, log_probs, mask)
                          global_native_scores = global_scores.cpu().data.numpy()
                           

                          all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                          all_log_probs_list.append(log_probs.cpu().data.numpy())
                          S_sample_list.append(S_sample.cpu().data.numpy())
                                  
                          for b_ix in range(BATCH_COPIES):
                              masked_chain_length_list = masked_chain_length_list_list[b_ix]
                              masked_list = masked_list_list[b_ix]
                              seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                              seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                              
                              score = scores[b_ix]
                              score_list.append(score)
                              global_score = global_native_scores[b_ix]
                              global_score_list.append(global_score)
                              native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                              if b_ix == 0 and j==0 and temp==temperatures[0]:
                                  start = 0
                                  end = 0
                                  list_of_AAs = []
                                  for mask_l in masked_chain_length_list:
                                      end += mask_l
                                      list_of_AAs.append(native_seq[start:end])
                                      start = end
                                  native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                  l0 = 0
                                  for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                      l0 += mc_length
                                      native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                      l0 += 1
                                  sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                  print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                  sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                  print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                  native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                  global_native_score_print = np.format_float_positional(np.float32(global_native_score.mean()), unique=False, precision=4)
                                  line = '>{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n'.\
                                  format(name_, 
                                        native_score_print, 
                                        global_native_score_print,
                                        print_visible_chains, 
                                        print_masked_chains, 
                                        model_name, 
                                        native_seq)
                                  print(line.rstrip())
                                  
                                
                              start = 0
                              end = 0
                              list_of_AAs = []
                              for mask_l in masked_chain_length_list:
                                  end += mask_l
                                  list_of_AAs.append(seq[start:end])
                                  start = end

                              seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                              l0 = 0
                              for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                  l0 += mc_length
                                  seq = seq[:l0] + '/' + seq[l0:]
                                  l0 += 1
                              score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                          
                              global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)

                              seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                              sample_number=j*BATCH_COPIES+b_ix+1
                              line = '\n>T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.\
                              format(temp,
                                    j,
                                    score_print,
                                    global_score_print,
                                    seq_rec_print,
                                    seq)
                              sequences.append(seq)
                              print(line.rstrip())


                              info={'pdb_name':name_, 
                              'pdb_path':pdb_path,
                              'native_score':native_score_print, 
                              'fixed_chains':print_visible_chains, 
                              'designed_chains':print_masked_chains, 
                              'model_name':model_name, 
                              'native_sequence':native_seq,
                              'temperature':temp,
                              'log_probs_score':score_print,
                              'global_score':global_score_print,
                              'sequence_recovery':seq_rec_print,
                              'designed_sequence':seq,
                              'designed_sequence_no':j,}
                              info_list.append(info)
                  
              if save_score:
                np.savez(score_file, score=np.array(score_list, np.float32), global_score=np.array(global_score_list, np.float32))
                
              if save_probs:
                all_probs_concat = np.concatenate(all_probs_list)
                all_log_probs_concat = np.concatenate(all_log_probs_list)
                S_sample_concat = np.concatenate(S_sample_list)
                np.savez(probs_file, probs=np.array(all_probs_concat, np.float32), log_probs=np.array(all_log_probs_concat, np.float32), S=np.array(S_sample_concat, np.int32), mask=mask_for_loss.cpu().data.numpy(), chain_order=chain_list_list)

                ### Amino acid probabilties
                plot_conditional_probs_heatmap(all_log_probs_concat, alphabet, base_path + f"/probs/{name_}/")
                ### Sampling temperature adjusted amino acid probabilties
                plot_sampled_probs_heatmap(all_probs_concat, alphabet, base_path + f"/probs/{name_}/")
              
              t1 = time.time()
              dt = round(float(t1-t0), 4)
              num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
              total_length = X.shape[1]
              if print_all:
                  print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds \n')
    print('-'*15,'ProteinMPNN prediction all done!','-'*15,'\n')
    mpnn_result=pd.DataFrame(info_list)
    mpnn_result.to_csv(f'./results/{model_type}_result.csv') 
    print('ProteinMPNN result saved!')
 


if __name__ == "__main__":
    
    import pandas as pd
    
    proteinMPNN_path = './SO3_Invariant_ProteinMPNN'
    pdb=pd.read_csv(f"{proteinMPNN_path}/dataset/test_mono.csv",index_col=0)
    pdb=','.join(pdb["PROTEIN"][:1000])
    
    run_mpnn(pdb)
