import argparse
import os.path
from distutils.util import strtobool  
def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim,randn
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random

    import os.path
    import subprocess
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from generic_model_utils import tied_featurize,loss_nll,loss_smoothed, get_std_opt, ProteinMPNN, _scores
    
    
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.out_folder, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    
     

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}


    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
     

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        num_letters=21, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise,
                        ca_only=args.ca_only,
                        rsh_expand=args.rsh_expand,
                        rsh_lmax=args.rsh_lmax)
    model.to(device)

    print('ca_only:',args.ca_only,'rsh_expand:',args.rsh_expand,'rsh_lmax:',args.rsh_lmax)
    
        
    logfile = base_folder + 'log.txt'    
    
    if args.previous_checkpoint:
        PATH = base_folder +'/model_weights/'+args.previous_checkpoint
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        total_step = 0
        epoch = 0
        optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    with open(logfile, 'a') as f:
        if args.previous_checkpoint:
            f.write(f'Resuming from epoch {epoch+1}\n')
        else:
            f.write('Epoch\tTrain\tValidation\n')
 
    print("Model is set")
    
    # helper functions
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
  
    pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs 
    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    


    with ProcessPoolExecutor(max_workers=24) as executor:
        futures_train = [executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch) for i in range(3)]
        futures_valid = [executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch) for i in range(3)]
        

        pdb_dict_train = []
        pdb_dict_valid = []


        for f in concurrent.futures.as_completed(futures_train):
            pdb_dict_train.extend(f.result())
        for f in concurrent.futures.as_completed(futures_valid):
            pdb_dict_valid.extend(f.result())


        # chain id
        homomer=False
        chain_id_dict = {}
        if homomer:
            tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_train)
        else:
            tied_positions_dict = None
        ###

        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
          
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
         

        reload_c = 0 
        t_0 = time.time()
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e

            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
 

                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                #score_list = []
                #all_probs_list = []
                #all_log_probs_list = []
                #S_sample_list = []
                #batch_clones = [copy.deepcopy(batch) for i in range(args.batch_size)]
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,ca_only=args.ca_only)
                #print('X.shape:',X.shape)
                elapsed_featurize = time.time() - start_batch
                #print('elapsed_featurize:',elapsed_featurize)
                #pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
                #name_ = batch[0]['name']
                #print('X shape:',X.shape)
                #print('S shape:',S.shape)
                #print('mask shape:',mask.shape)
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                mask_for_loss = mask*chain_M*chain_M_pos
                optimizer.zero_grad()


                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                    scaler.scale(loss_av_smoothed).backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                total_step += 1

            model.eval()
            # chain id
            chain_id_dict = {}
            if homomer:
                tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_valid)
            else:
                tied_positions_dict = None

            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.

                for _, batch in enumerate(loader_valid):
                    #score_list = []
                    #all_probs_list = []
                    #all_log_probs_list = []
                    #S_sample_list = []
                    # batch_clones = [copy.deepcopy(batch) for i in range(args.num_epochs)]
                    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,ca_only=args.ca_only)

                    #pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
                    #name_ = batch[0]['name']

                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                    mask_for_loss = mask*chain_M*chain_M_pos

                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')

            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)

        t_1 = time.time()
        duration = np.format_float_positional(np.float32(t_1-t_0), unique=False, precision=1) 
        print(f"Total duration for training and validation: {duration}s")
                
 



if __name__ == "__main__":
    
    def parse_bool_or_string(value):
        if value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            return value
        
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--out_folder", type=str, default='./vanilla_model_weights', help="path for logs and model weights")
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. epoch_last.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--ca_only", default=False, type=lambda x: bool(strtobool(x)), help="construct geometry based on three carbon alpha")
    argparser.add_argument("--rsh_expand", default=False, type=parse_bool_or_string, choices=[True, False, "Only",'None','Cartesian'], help="expand the coordinates through real spherical harmonics")
    argparser.add_argument("--rsh_lmax", type=int, default=2, help="maximum degree that the real spherical harmonics expand")
    args = argparser.parse_args()     
    main(args)   

                       