## This is the code and script for retraining the model using different representation methods

`./script` contains training scripts and log files.

`./training.py` and `./model_utils.py` are the modified files where the major changes are made as follows:

1. Update the complete structural encoding, that include orientation and rotation for building the edge features, functiona are adopted from 
protein_mpnn_run.py and protein_mpnn_utils.py 
2. Update the feature pre-processing functions from `featurize` to `tied_featurize`, to enable 
