# Re-train ProteinMPNN with SO3 Invariant Representation
To train/retrain ProteinMPNN clone this github repo and install Python>=3.0, PyTorch, Numpy. 

The multi-chain training data (16.5 GB, PDB biounits, 2021 August 2) can be downloaded from here: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz`; The small subsample (47 MB) of this data for testing purposes can be downloaded from here: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz`

```
Training set for ProteinMPNN curated by Ivan Anishchanko.

Each PDB entry is represented as a collection of .pt files:
    PDBID_CHAINID.pt - contains CHAINID chain from PDBID
    PDBID.pt         - metadata and information on biological assemblies

PDBID_CHAINID.pt has the following fields:
    seq  - amino acid sequence (string)
    xyz  - atomic coordinates [L,14,3]
    mask - boolean mask [L,14]
    bfac - temperature factors [L,14]
    occ  - occupancy [L,14] (is 1 for most atoms, <1 if alternative conformations are present)

PDBID.pt:
    method        - experimental method (str)
    date          - deposition date (str)
    resolution    - resolution (float)
    chains        - list of CHAINIDs (there is a corresponding PDBID_CHAINID.pt file for each of these)
    tm            - pairwise similarity between chains (TM-score,seq.id.,rmsd from TM-align) [num_chains,num_chains,3]
    asmb_ids      - biounit IDs as in the PDB (list of str)
    asmb_details  - how the assembly was identified: author, or software, or smth else (list of str)
    asmb_method   - PISA or smth else (list of str)

    asmb_chains    - list of chains which each biounit is composed of (list of str, each str contains comma separated CHAINIDs)
    asmb_xformIDX  - (one per biounit) xforms to be applied to chains from asmb_chains[IDX], [n,4,4]
                     [n,:3,:3] - rotation matrices
                     [n,3,:3] - translation vectors

list.csv:
   CHAINID    - chain label, PDBID_CHAINID
   DEPOSITION - deposition date
   RESOLUTION - structure resolution
   HASH       - unique 6-digit hash for the sequence
   CLUSTER    - sequence cluster the chain belongs to (clusters were generated at seqID=30%)
   SEQUENCE   - reference amino acid sequence

valid_clusters.txt - clusters used for validation

test_clusters.txt - clusters used for testing
```

Code organization:
* `generic_train.py` - the main script to train the model
* `generic_model_utils.py` - utility functions and classes for the model
* `utils.py` - utility functions and classes for data loading
* `generic_outputs` - sample output files
* `generic_train.pbs` - HPC job submit script
* `generic_train_resume.pbs` - HPC job submit script for resuming training from an interruption
-----------------------------------------------------------------------------------------------------
Input flags for `generic_training.py`:
```
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
```
-----------------------------------------------------------------------------------------------------
For example to make a conda environment to run ProteinMPNN:
* `conda create --name mlfold` - this creates conda environment called `mlfold`
* `source activate mlfold` - this activate environment
* `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` - install pytorch following steps from https://pytorch.org/
-----------------------------------------------------------------------------------------------------
## Model Options
Models were trained for the SO3 Invariant ProteinMPNN using the following input flags. The original training repository can be accessed at [link](https://github.com/dauparas/ProteinMPNN/tree/main/training).
 

#### Carbon alpha based geometry trained with Radial Basis Function Lifting (RBF)
* `Ca_benchmark.pt`  `--ca_only True --rsh_expand None `

#### Carbon alpha based geometry trained with RBF + Orientation and Rotation representation
* `Ca_OR.pt`  `--ca_only True --rsh_expand False `

#### Carbon alpha based geometry trained with RBF + Real Spherical Harmonics representation(expanded through 3 degrees)
* `Ca_RSH3`  `--ca_only True --rsh_expand True --rsh_lmax 3 `

#### Carbon alpha based geometry trained with RBF + Real Spherical Harmonics representation (expanded through 4 degrees)
* `Ca_RSH4`  `--ca_only True --rsh_expand True --rsh_lmax 4 `

#### Carbon alpha based geometry trained with RBF + Real Spherical Harmonics representation (expanded through 5 degrees)
* `Ca_RSH5`  `--ca_only True --rsh_expand True --rsh_lmax 5 `

---

#### Full backbone geometry trained with Radial Basis Function Lifting (RBF)
* `vanilla_benchmark.pt`  `--ca_only False --rsh_expand None `

#### Full backbone geometry trained with RBF + Orientation and Rotation representation
* `vanilla_OR.pt`  `--ca_only False --rsh_expand False `

#### Full backbone geometry trained with RBF + Real Spherical Harmonics representation (expanded through 3 degrees)
* `vanilla_RSH3`  `--ca_only False --rsh_expand True --rsh_lmax 3 `

#### Full backbone geometry trained with RBF + Real Spherical Harmonics representation (expanded through 4 degrees)
* `vanilla_RSH4`  `--ca_only False --rsh_expand True --rsh_lmax 4 `

#### Full backbone geometry trained with RBF + Real Spherical Harmonics representation (expanded through 5 degrees)
* `vanilla_RSH5`  `--ca_only False --rsh_expand True --rsh_lmax 5 `


 -----------------------------------------------------------------------------------------------------
