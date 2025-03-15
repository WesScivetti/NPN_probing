from transformers import AutoModel, AutoTokenizer
import sys
import pandas as pd
import json
import re
import torch
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import argparse
from collections import defaultdict, Counter
import random
from sklearn.neural_network import MLPClassifier

def get_tokenized_input(row, tokenizer, pert=0):
    """
    takes a row of a df and constructs a tokenized (by BERT's tokenizer)
    """
    #"to" ID in bert-large-cased = 1106
    #[UNK] ID = 100
    text = row["Sentence Raw Text"]
    pert = row["Perturbation"]
    noun = row["N1"]


    if pert == 0:
        pattern = noun.lower() + " to " + noun.lower()
        repl_pattern = noun.lower() + " [UNK] " + noun.lower()

    elif pert == 1:
        pattern = noun.lower() + " " + noun.lower() + " " + "to"
        repl_pattern = noun.lower() + " " + noun.lower() + " " + "[UNK]"

    elif pert == 2:
        pattern = "to " + noun.lower() + " " + noun.lower()
        repl_pattern = "[UNK] " + noun.lower() + " " + noun.lower()

    elif pert == 3:
        pattern = noun.lower() + " " + "to"
        repl_pattern = noun.lower() + " " + "[UNK]"

    elif pert == 4:
        pattern = "to" + " " + noun.lower()
        repl_pattern = "[UNK]" + " " + noun.lower()
    #print(text)
    replaced_text = re.sub(pattern, repl_pattern, text, flags=re.IGNORECASE)
    # print(pert, file=sys.stderr)
    # print(replaced_text, file=sys.stderr)
    tokenized_text = tokenizer(replaced_text, truncation=True)

    #print(tokenized_text)
    #print(replaced_text)
    target_id = tokenized_text["input_ids"].index(100)
    tokenized_text["input_ids"][target_id] = 1106

    #convert to pt tensors
    new_tokenized_text = {
        "input_ids": torch.tensor([tokenized_text["input_ids"]]),
        "token_type_ids": torch.tensor([tokenized_text["token_type_ids"]]),
        "attention_mask": torch.tensor([tokenized_text["attention_mask"]])
    }
    return new_tokenized_text, target_id

def get_embeddings(model, tokenized_text, target_id):
    """given a BERT tokenized text and a target id, returns the bert embeddings from each layer"""
    embeddings_list = []
    with torch.no_grad():
        outputs = model(**tokenized_text)
        hidden_states = outputs.hidden_states
        for layer in range(0, 13):
            embeddings = outputs.hidden_states[layer]
            target_embedding = embeddings[0, target_id, :].numpy()
            embeddings_list.append(target_embedding)

    return embeddings_list

def get_control_label_set(df, idx_set, semantic=False):
    """
    given the dataframe, returns a dictionary with the majority label for each N1 lemma.
    used for shuffling labels for the control classifier.
    """
    noun_labels = defaultdict(list)
    for r in df.index:
        if r in idx_set:
            row = df.loc[r]
            noun = row["N1"].lower()
            text = row["Sentence Raw Text"]

            if not semantic:
                label = 1 if row["Subtype"] in ["I", "J"] else 0

            if semantic:
                label2id = {"F": 0, "B": 1, "I": 1, "J": 2}
                id2label = {0: "F", 1: "I", 2: "J"}

                raw_sem = row["Subtype"]

                sem = raw_sem if pd.notna(raw_sem) else 'N/A'

                label = label2id[sem]

            noun_labels[noun].append(label)

    majority_labels = {}
    
    for n in sorted(noun_labels.keys()):
        n_counter = Counter(noun_labels[n])
        majority_val = max(n_counter, key=n_counter.get)
        majority_labels[n] = majority_val

    return majority_labels


def shuffle_labels(majority_labels, semantic=False):
    """
    takes the dictionary of majority labels and shuffles them to return the new control labels
    """
    items = sorted(majority_labels.items())
    nouns = [i[0] for i in items]
    labels = [i[1] for i in items]

    random.seed(99)
    random.shuffle(labels)
    #print("Counter", Counter(labels))
    #Bias a bit toward the positive for lemmas to get closer to a balanced dataset
    random.seed(42)
    choices = [0, 1]
    if semantic:
        choices = [0, 1, 2]

    labels = [random.choice(choices) for i in range(len(labels))]
    #print("Counter (new)", Counter(labels))
    shuffled_labels = {}

    for i in range(len(nouns)):
        n = nouns[i]
        shuff_lab = labels[i]
        shuffled_labels[n] = shuff_lab

    return shuffled_labels

def load_glove_embeddings(filepath):
    """loads glove embeddings from file"""
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # First value is the word
            vector = np.array(values[1:], dtype='float32')  # The rest are the vector values
            embeddings_index[word] = vector
    return embeddings_index


def make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=6, control=False, semantic=False, glove=False):
    """
    makes train test split with embeddings, also can create control datasets for control classifiers
    """
    print(f"making the train/test sets for layer {layer_num}", file=sys.stderr)
    if semantic:
        print("SEMANTIC VERSION", file=sys.stderr)

    if control==True:
        print("This is the control dataset", file=sys.stderr)
    print("---------------------------", file=sys.stderr)
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    if control == True:
        majority_labels_train = get_control_label_set(df, train_idx, semantic=semantic)
        majority_labels_test = get_control_label_set(df, test_idx, semantic=semantic)
        shuffed_labels_train = shuffle_labels(majority_labels_train, semantic=semantic)
        shuffed_labels_test = shuffle_labels(majority_labels_test, semantic=semantic)

    train_labs_orig = []
    train_labs_shuff = []
    if glove:
        print("loading the glove embeddings", file=sys.stderr)
        glove_path = "glove/glove.6B.300d.txt"  # Update path if necessary
        glove_embeddings = load_glove_embeddings(glove_path)
        print("glove loaded!", file=sys.stderr)

    for r in df.index:
        # print(r)
        row = df.loc[r]
        # print(row)
        noun = row["N1"]
        text = row["Sentence Raw Text"]
        if not semantic:
            label = 1 if row["Subtype"] in ["I", "J"] else 0
        if semantic:
            label2id = {"F": 0, "B": 1, "I": 1, "J": 2, "N/A": 3}
            id2label = {0: "F", 1: "I",  2: "J", 3: "N/A"}

            raw_sem = row["Subtype"]

            sem = raw_sem if pd.notna(raw_sem) else 'N/A'

            label = label2id[sem]

        tokenized_text, target_id = get_tokenized_input(row, tokenizer)

        if not glove:
            embeddings_list = get_embeddings(model, tokenized_text, target_id)
            # # print(len(embeddings_list), file=sys.stderr)
            current_embedding = embeddings_list
        # print(len(current_embedding))

        if glove:
            if noun in glove_embeddings:
                current_embedding = glove_embeddings[noun]
            else: #if not in glove, just make a random 300 dimensional vector
                current_embedding = np.random.uniform(-0.1, 0.1, 300)


        if r in train_idx:
            train_X.append(current_embedding)
            if control == False:
                train_y.append(label)
            if control == True:
                train_labs_orig.append(label)
                new_label = shuffed_labels_train[noun.lower()]
                train_labs_shuff.append(new_label)
                #print("training old+new:", label, new_label)
                #assert new_label == label
                train_y.append(new_label)

        if r in test_idx:
            test_X.append(current_embedding)
            if control == False:
                test_y.append(label)
            if control == True:

                new_label = shuffed_labels_test[noun.lower()]
                test_y.append(new_label)

    if control == True:
        print("orig proportions", Counter(train_labs_orig))
        print("shuffled proportions", Counter(train_labs_shuff))

    return train_X, train_y, test_X, test_y

def open_pert_df(pert, semantic=False):
    if not semantic:
        if pert == 1:
            pert_df = pd.read_csv("./data/Kat_clean6_NNP_perturbed_NNP.tsv", delimiter="\t")
            pert = 1
        if pert == 3:
            pert_df = pd.read_csv("./data/Kat_clean6_NP_perturbed_NP.tsv", delimiter="\t")
            pert = 3
        if pert == 4:
            pert_df = pd.read_csv("./data/Kat_clean6_PN_perturbed_PN.tsv", delimiter="\t")
            pert = 4
        if pert == 2:
            pert_df = pd.read_csv("./data/Kat_clean6_PNN_perturbed_PNN.tsv", delimiter="\t")
            pert = 2
    else:
        if pert == 1:
            pert_df = pert_df = pd.read_csv("./data/Kat_clean6_NNP_perturbed_NNP.tsv", delimiter="\t")
            pert = 1
        if pert == 3:
            pert_df = pert_df = pd.read_csv("./data/Kat_clean6_NP_perturbed_NP.tsv", delimiter="\t")
            pert = 3
        if pert == 4:
            pert_df = pd.read_csv("./data/Kat_clean6_PN_perturbed_PN.tsv", delimiter="\t")
            pert = 4
        if pert == 2:
            pert_df = pd.read_csv("./data/Kat_clean6_PNN_perturbed_PNN.tsv", delimiter="\t")
            pert = 2

        ##add in different file names here once they are made
    return pert_df


def make_pert_test_data(test_idx, model, tokenizer, pert=1, layer_num=6, control=False, semantic=False):
    pert_df = open_pert_df(pert, semantic=semantic)
    test_X = []
    test_y = []
    test_y_orig = []
    if control == True:
        majority_labels = get_control_label_set(pert_df, test_idx, semantic=semantic)
        shuffed_labels_test = shuffle_labels(majority_labels, semantic=semantic)

    for r in pert_df.index:
        if r in test_idx:
            row = pert_df.loc[r].copy()
            noun = row["N1"]
            if not semantic:
                orig_label = 1 if row["Subtype"] == ["I", "J"] else 0
                label = 1 if row["Perturbation"] == 0 else 0 #should always be 0 for perturbed cases
            else:
                label2id = {"F": 0, "B": 1, "I": 1, "J": 2} #Iteration and Boundary are considered the same semantic subtype
                id2label = {0: "F", 1: "I", 2: "J"}

                raw_sem = row["Subtype"]

                sem = raw_sem if pd.notna(raw_sem) else 'N/A'

                label = label2id[sem]
                orig_label = label

            tokenized_text, target_id = get_tokenized_input(row, tokenizer, pert=pert)
            embeddings_list = get_embeddings(model, tokenized_text, target_id)
            # print(len(embeddings_list))
            current_embedding = embeddings_list #all embeddings
            test_X.append(current_embedding)
            if control == False:
                test_y_orig.append(orig_label)
            if control == True:
                new_orig_label = shuffed_labels_test[noun.lower()]
                test_y_orig.append(new_orig_label)

            test_y.append(label)
    return test_X, test_y, test_y_orig


def run_model(trainX, train_y, test_X, test_y, outfile = None, clf_type="LR"):
    """
    Given lists of train and test embeddings and labels, run probing classifier of specified type
    """
    print("running the model", file=sys.stderr)
    print("Classifier type:", clf_type)
    print("----------------", file=sys.stderr)

    if clf_type == "LR":
        clf = LogisticRegression(random_state=0, max_iter=10000)
    elif clf_type == "MLP1":
        clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, random_state=42)
    elif clf_type == "MLP2":
        clf = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=5000, random_state=42)
    clf.fit(trainX, train_y)
    preds = clf.predict(test_X)
    print(classification_report(test_y, preds), file=sys.stderr)
    if outfile:
        with open(outfile, "w") as fout:
            print(classification_report(test_y, preds), file=fout)
    return preds, clf

def load_split_indices(index_file, num_train=None, split_seed=None):
    """load a file containing the train and test indices and read it back into lists of indices"""
    with open(index_file) as fin:
        data = json.load(fin)

    if split_seed:
        # print(data["F_train"], file=sys.stderr)
        random.shuffle(data["F_train"])
        # print(data["F_train"], file=sys.stderr)
        random.shuffle(data["I_train"])
        random.shuffle(data["J_train"])


    if not num_train:
        train_idx = data["F_train"] + data["I_train"] + data["J_train"]
        test_idx = data["F_test"] + data["I_test"] + data["J_test"]
    else:
        train_idx = data["F_train"][:num_train] + data["I_train"][:num_train] + data["J_train"][:num_train]
        test_idx = data["F_test"] + data["I_test"] + data["J_test"]
    return train_idx, test_idx

def clustering_experiment(df, train_idx, test_idx, trainX, train_y, test_X, test_y, preds):
    """clustering experiment to assess prototypicality"""
    #actually need train_idx, test_idx, trainX, train_y, testX, test_y, preds
    #
    train_count = 0
    embed_dict = defaultdict(list)

    for r in df.index:
        if r in train_idx:
            embedding = trainX[train_count]
            label = train_y[train_count]
            embed_dict[label].append(embedding) #add embedding to list for its subtype, used for centroid calculation
            train_count += 1

    #calculate centroids for each semantic subtype
    #label 0 = F, label 1 = I, label 2 = J

    # print("embed dict", embed_dict)
    array_0 = np.stack(embed_dict[0])
    centroid_0 = np.mean(array_0, axis=0) #1 vector with mean value of each dimension (columns in original array)

    array_1 = np.stack(embed_dict[1])
    centroid_1 = np.mean(array_1, axis=0)

    array_2 = np.stack(embed_dict[2])
    centroid_2 = np.mean(array_2, axis=0)
    # print(centroid_0)
    # print(centroid_1)
    # print(centroid_2)
    #loop through test
    #calculate which is the closest centroid in terms of euclidean distance

    #gather sim_lists
    #nested dictionary with list of similarities for each label

    sim_dict = defaultdict(lambda: defaultdict(list))

    cluster_preds = []

    test_count = 0
    for r in df.index:
        if r in test_idx:
            embedding = test_X[test_count]
            gold_label = test_y[test_count]
            pred_label = preds[test_count]
            #calculate cosine similarity of embedding to existing clusters
            sims = cosine_similarity([embedding], [centroid_0, centroid_1, centroid_2])
            # print("GOLD LABEL", gold_label)
            # print("SIM to False:", sims[0])
            # print("SIM to Succ", sims[1])
            # print("SIM to Juxta", sims[2])
            # print("--------")
            # print(top_cluster)
            top_sim = np.max(sims)
            top_cluster = np.argmax(sims)
            # print(gold_label, top_cluster)

            # print(top_sim, top_cluster)
            cluster_preds.append(top_cluster)

            top_sim = float(top_sim)

            gold_label_sim = float(sims[0][gold_label])  #record sim to gold label

            if gold_label == pred_label:
                sim_dict[gold_label]["C"].append(gold_label_sim)

            if gold_label != pred_label:
                sim_dict[gold_label]["I"].append(gold_label_sim)

            test_count += 1


    #print(sim_dict)
    return cluster_preds, sim_dict







def main(data_file, index_file, pert=None, semantic=False, num_train = 270, split_seed=None):
    """
    full pipeline
    """
    if semantic:
        print("SEMANTIC EXPERIMENTS", file=sys.stderr)
    train_idx, test_idx = load_split_indices(index_file, num_train=num_train, split_seed=split_seed)
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    df = pd.read_csv(data_file, delimiter="\t")

    trainX_glove, train_y_glove, test_X_glove, test_y_glove = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=0, semantic=semantic, glove=True)
    trainX_control_glove, train_y_control_glove, test_X_control_glove, test_y_control_glove = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=0, control=True, semantic=semantic, glove=True)


    for clf_type in ["LR"]:  # add in MLPs later

        if not semantic:
            out_file = f"./outputs/predictions_glove_base_lt_{clf_type}2_2025_{num_train}_seed_{split_seed}.tsv"
            out_file1 = f"./outputs/classification_report_glove" + f"_base_lt_{clf_type}2_2025_{num_train}_seed_{split_seed}.txt"
            out_file_control = "./outputs/classification_report_glove" + f"_base_lt_control_{clf_type}2_2025_{num_train}_seed_{split_seed}.txt"

        else:
            out_file = "./outputs/predictions_glove_" + f"_base_lt_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.tsv"
            out_file1 = "./outputs/classification_report_glove" + f"_base_lt_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.txt"
            out_file_control = "./outputs/classification_report_glove"  + f"_base_lt_control_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.txt"

        columns = list(df.columns) + ["Pred", "Control Pred", "Control Gold"]
        new_df = pd.DataFrame(columns=columns)

        preds, clf = run_model(trainX_glove, train_y_glove, test_X_glove, test_y_glove, outfile=out_file1, clf_type=clf_type)
        preds_control, control_clf = run_model(trainX_control_glove, train_y_control_glove, test_X_control_glove, test_y_control_glove,
                                               outfile=out_file_control, clf_type=clf_type)

        # if clustering, do clustering experiment
        # cluster_preds, sim_dict = clustering_experiment(df, train_idx, test_idx, trainX_glove, train_y_glove, test_X_glove, test_y_glove, preds)
        # fout_clust_fname = "./outputs/clustering_sims_glove" + f"_lt_{clf_type}_2025_{num_train}_seed_{split_seed}.json"
        # fout_clust_report_fname = "./outputs/clustering_report_glove" + f"_lt_{clf_type}_2025_{num_train}_seed_{split_seed}.txt"
        # print(test_y_glove)
        # print(cluster_preds)
        # print results of clustering as classifications to a file
        # with open(fout_clust_report_fname, "w") as fout_clust_report:
        #     print(f"CLASSIFICATION REPORT CLUSTERING GloVe: {clf_type}",
        #
        #           classification_report(test_y_glove, cluster_preds))
        #     print(f"CLASSIFICATION REPORT CLUSTERING GloVe: {clf_type}",
        #           classification_report(test_y_glove, cluster_preds), file=fout_clust_report)
        #
        # # print clustering results to json
        # with open(fout_clust_fname, 'w') as cluster_sim_file:
        #     json.dump(sim_dict, cluster_sim_file, indent=4)

        #     # store models for perturbed usage
        # clf_dict[clf_type] = clf
        # clf_control_dict[clf_type] = control_clf

        count = 0
        for r in df.index:
            if r in test_idx:
                row = df.loc[r].copy()
                pred = preds[count]
                row["Pred"] = pred
                c_pred = preds_control[count]
                row["Control Pred"] = c_pred
                row["Control Gold"] = test_y_control_glove[count]
                # print(row)
                count += 1
                new_df.loc[len(new_df.index)] = row
        print("writing results to tsv", file=sys.stderr)
        new_df.to_csv(out_file, sep="\t", index=False)

    trainXall, train_y, test_Xall, test_y = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=6, semantic=semantic)
    trainX_controlall, train_y_control, test_X_controlall, test_y_control = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=6, control=True, semantic=semantic)

    perts = defaultdict(dict)
    for j in range(1, 5):
        print("working on pert datasets", file=sys.stderr)
        pert_test_X, pert_test_y, pert_test_y_orig = make_pert_test_data(test_idx, model, tokenizer, pert=j,
                                                                         semantic=semantic)
        pert_test_X_control, pert_test_y_control, pert_test_y_orig_control = make_pert_test_data(test_idx, model,
                                                                                                 tokenizer, pert=j,
                                                                                                 control=True,
                                                                                                 semantic=semantic)
        perts[j]["test"] = (pert_test_X, pert_test_y, pert_test_y_orig)
        perts[j]["control"] = (pert_test_X_control, pert_test_y_control, pert_test_y_orig_control)

    print("starting the loop", file=sys.stderr)
    #this is actually the same indent level as above, wtf
    for i in range(0, 13):
        # out_file = "./outputs/predictions_layer_" + str(i) + f"_base_lt_{clf_type}.tsv"
        # out_file1 = "./outputs/classification_report_layer" + str(i) + f"_base_lt_{clf_type}.txt"
        # out_file_control = "./outputs/classification_report_layer" + str(i) + f"_base_lt_control_{clf_type}.txt"
        # columns = list(df.columns) + ["Pred", "Control Pred", "Control Gold"]
        # new_df = pd.DataFrame(columns=columns)
        print("narrowing the training sets")
        print(trainXall)
        trainX = [layer[i] for layer in trainXall]
        test_X = [layer[i] for layer in test_Xall]
        print(trainX)

        trainX_control = [layer[i] for layer in trainX_controlall]
        test_X_control = [layer[i] for layer in test_X_controlall]




        clf_dict = {}
        clf_control_dict = {}
        for clf_type in ["LR"]: #add in MLPs later

            if not semantic:
                out_file = "./outputs/predictions_layer_" + str(i) + f"_base_lt_{clf_type}2_2025_{num_train}_seed_{split_seed}.tsv"
                out_file1 = "./outputs/classification_report_layer" + str(i) + f"_base_lt_{clf_type}2_2025_{num_train}_seed_{split_seed}.txt"
                out_file_control = "./outputs/classification_report_layer" + str(i) + f"_base_lt_control_{clf_type}2_2025_{num_train}_seed_{split_seed}.txt"

            else:
                out_file = "./outputs/predictions_layer_" + str(i) + f"_base_lt_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.tsv"
                out_file1 = "./outputs/classification_report_layer" + str(i) + f"_base_lt_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.txt"
                out_file_control = "./outputs/classification_report_layer" + str(i) + f"_base_lt_control_{clf_type}_subtype2_2025_{num_train}_seed_{split_seed}.txt"

            columns = list(df.columns) + ["Pred", "Control Pred", "Control Gold"]
            new_df = pd.DataFrame(columns=columns)


            print("training", file=sys.stderr)
            preds, clf = run_model(trainX, train_y, test_X, test_y, outfile=out_file1, clf_type=clf_type)
            preds_control, control_clf = run_model(trainX_control, train_y_control, test_X_control, test_y_control, outfile=out_file_control, clf_type=clf_type)

            #if clustering, do clustering experiment
            # cluster_preds, sim_dict = clustering_experiment(df, train_idx, test_idx, trainX, train_y, test_X, test_y, preds)
            # fout_clust_fname = "./outputs/clustering_sims_layer" + str(i) + f"_lt_{clf_type}_2025_{num_train}_seed_{split_seed}.json"
            # fout_clust_report_fname = "./outputs/clustering_report_layer" + str(i) + f"_lt_{clf_type}_2025_{num_train}_seed_{split_seed}.txt"
            #
            # #print results of clustering as classifications to a file
            # with open(fout_clust_report_fname, "w") as fout_clust_report:
            #     print(f"CLASSIFICATION REPORT CLUSTERING {i}: {clf_type}",
            #           classification_report(test_y, cluster_preds))
            #     print(f"CLASSIFICATION REPORT CLUSTERING {i}: {clf_type}",
            #           classification_report(test_y, cluster_preds), file=fout_clust_report)

            #print clustering results to json
            # with open(fout_clust_fname, 'w') as cluster_sim_file:
            #     json.dump(sim_dict, cluster_sim_file, indent=4)

                #store models for perturbed usage
            clf_dict[clf_type] = clf
            clf_control_dict[clf_type] = control_clf

            count = 0
            for r in df.index:
                if r in test_idx:
                    row = df.loc[r].copy()
                    pred = preds[count]
                    row["Pred"] = pred
                    c_pred = preds_control[count]
                    row["Control Pred"] = c_pred
                    row["Control Gold"] = test_y_control[count]
                    # print(row)
                    count += 1
                    new_df.loc[len(new_df.index)] = row
            print("writing results to tsv", file=sys.stderr)
            new_df.to_csv(out_file, sep="\t", index=False)

            if not semantic:
                #do perts
                for j in range(1, 5):
                    print("working on pert datasets", file=sys.stderr)
                    pert_test_Xall, pert_test_y, pert_test_y_orig = perts[j]["test"]
                    pert_test_X_control_all, pert_test_y_control, pert_test_y_orig_control = perts[j]["control"]

                    pert_test_X = [layer[i] for layer in pert_test_Xall]

                    pert_test_X_control = [layer[i] for layer in pert_test_X_control_all]

                    for clf_type in ["LR"]: #add in MLPs later
                        clf = clf_dict[clf_type]
                        control_clf = clf_control_dict[clf_type]
                        pert_preds = list(clf.predict(pert_test_X))
                        pert_preds_control = list(control_clf.predict(pert_test_X_control))
                        # print(pert_preds,file=sys.stderr)
                        print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL {clf_type}", classification_report(pert_test_y_orig, pert_preds))
                        print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL {clf_type}", classification_report(pert_test_y, pert_preds))

                        #add classification report renamed
                        fout_pert_fname = "./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) + f"_lt_{clf_type}2_{num_train}_{split_seed}.txt"

                        if semantic:
                            fout_pert_fname = "./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) + f"_lt_{clf_type}_subtype2_{num_train}_{split_seed}.txt"


                        with open(fout_pert_fname, "w") as fout_pert:
                            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL",
                                  classification_report(pert_test_y_orig, pert_preds), file=fout_pert)
                            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL",
                                  classification_report(pert_test_y, pert_preds), file=fout_pert)

                        f_out_pert_control_f = "./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) + f"_lt_control_{clf_type}2_{num_train}_{split_seed}.txt"

                        if semantic:
                            f_out_pert_control_f = "./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) + f"_lt_control_{clf_type}_subtype2_{num_train}_{split_seed}.txt"

                        with open(f_out_pert_control_f, "w") as fout_pert_control:
                            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL, CONTROL CLASSIFIER {clf_type}", classification_report(pert_test_y_orig_control, pert_preds_control), file=fout_pert_control)
                            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL, CONTROL CLASSIFIER {clf_type}", classification_report(pert_test_y_control, pert_preds_control), file=fout_pert_control)

                        pert_df = open_pert_df(j)
                        new_pert_df = pd.DataFrame(columns=columns)
                        p_count = 0
                        # print(p_count, file=sys.stderr)
                        # print("PREDS:  ", pert_preds, len(pert_preds), pert_preds[0], file=sys.stderr)
                        for r  in pert_df.index:
                            if r  in test_idx:
                                row = pert_df.loc[r].copy()
                                # print("WTF", p_count, pert_preds[0])
                                predd = pert_preds[p_count]
                                row["Pred"] = predd
                                row["Control Pred"] = pert_preds_control[p_count]
                                row["Control Gold"] = pert_test_y_orig_control[p_count]
                                p_count += 1
                                new_pert_df.loc[len(new_pert_df.index)] = row

                        out_data_name = "./outputs/perturbed/" + f"pert_predictions_layer_{i}_pert_{j}_lt_{clf_type}2_{num_train}_{split_seed}.tsv"
                        if semantic:
                            out_data_name = "./outputs/perturbed/" + f"pert_predictions_layer_{i}_pert_{j}_lt_{clf_type}_subtype2_{num_train}_{split_seed}.tsv"

                        new_pert_df.to_csv(out_data_name, index=False, sep="\t")

                    columns = list(df.columns) + ["Pred"]
                    new_pert_df = pd.DataFrame(columns=columns)

            count = 0
            for r in df.index:
                if r in test_idx:
                    row = df.loc[r].copy()
                    pred = preds[count]
                    row["Pred"] = pred
                    c_pred = preds_control[count]
                    row["Control Pred"] = c_pred
                    row["Control Gold"] = test_y_control[count]
                    # print(row)
                    count += 1
                    new_df.loc[len(new_df.index)] = row
            new_df.to_csv(out_file, sep="\t", index=False)




    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file")
    parser.add_argument("-i", "--index_file")
    parser.add_argument("-sm", "--semantic", action="store_true")
    parser.add_argument("--num_train", type=int)
    parser.add_argument("--split_seed", type=int)
    args = parser.parse_args()
    main(args.data_file, args.index_file, semantic=args.semantic, num_train=args.num_train, split_seed=args.split_seed)


