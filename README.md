# NPN_probing
NPN probing experiments

# How to run
The main script for these experiments is the get_embeddings.py script. The other scripts are just for preprocessing, and aren't necessary to run anymore. The cleaned data is already in the repo.

# Example arguments
python3 get_embeddings.py -d ./data/raw_NPN_data_cleaned_subtype.tsv -i ./data/train_test_split_train_balanced_Y.json --semantic

# Data folder
subtype_annotation.tsv is the main data for the ML final project. Other data files were used for QP1 but aren't needed for this project.

# Outputs folder

for the ML final, the main output files are NPN results_clustering.csv, NPNresults_semantic.csv, and the clustering_sims.json files. These are used for outputting the visualizations. Full classification reports, as well as full test outputs alongside the datasets, are all logged in this folder for future error analysis. The "perturbed" subdirectory isn't relevant to this project.
