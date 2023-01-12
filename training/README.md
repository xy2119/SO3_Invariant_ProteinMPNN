## This is the code and script for retraining the model using different representation methods. `./script` contains training scripts and log files. The core changes to the training code are.


1. Update the complete structural encoding when building the edge features, which include orientation and rotation, from 
protein_mpnn_run.py and protein_mpnn_utils.py 
2. Update the feature processing functions from `featurize` to `tied_featurize`
