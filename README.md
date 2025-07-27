# Overview
This repo houses our NPN construction probing experiments - [paper](https://aclanthology.org/2025.conll-1.24/) presented at ConLL 2025! 

# How to run
The main script for these experiments is the get_embeddings.py script. The other scripts are just for preprocessing, and aren't necessary to run anymore. The cleaned data is already in the repo.

# Example arguments
python3 get_embeddings.py -d ./data/raw_NPN_data_cleaned_subtype.tsv -i ./data/train_test_split_train_balanced_Y.json --semantic

# Data folder
raw_NPN_data_cleaned_subtype.tsv is the main data.

# Outputs folder
The main output files are NPN results_clustering.csv, NPNresults_semantic.csv, and the clustering_sims.json files. These are used for outputting the visualizations. Full classification reports (for all random seeds), as well as full test outputs alongside the datasets, are all logged in this folder for future error analysis. The "perturbed" subdirectory contains all results for Experiment 2: Perturbations
