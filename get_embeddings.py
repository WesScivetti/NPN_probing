from transformers import AutoModel, AutoTokenizer
import sys
import pandas as pd
import json
import re
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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

    replaced_text = re.sub(pattern, repl_pattern, text, flags=re.IGNORECASE)
    # print(pert, file=sys.stderr)
    # print(replaced_text, file=sys.stderr)
    tokenized_text = tokenizer(replaced_text)
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

def get_control_label_set(df, idx_set):
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
            label = 1 if row["Orig Label"] == "Y" else 0
            noun_labels[noun].append(label)

    majority_labels = {}
    
    for n in sorted(noun_labels.keys()):
        n_counter = Counter(noun_labels[n])
        majority_val = max(n_counter, key=n_counter.get)
        majority_labels[n] = majority_val

    return majority_labels


def shuffle_labels(majority_labels):
    """
    takes the dictionary of majority labels and shuffles them to return the new control labels
    """
    items = sorted(majority_labels.items())
    nouns = [i[0] for i in items]
    labels = [i[1] for i in items]

    random.seed(99)
    random.shuffle(labels)
    print("Counter", Counter(labels))
    #Bias a bit toward the positive for lemmas to get closer to a balanced dataset
    random.seed(42)
    choices = [0, 1]
    labels = [random.choice(choices) for i in range(len(labels))]
    print("Counter (new)", Counter(labels))
    shuffled_labels = {}

    for i in range(len(nouns)):
        n = nouns[i]
        shuff_lab = labels[i]
        shuffled_labels[n] = shuff_lab

    return shuffled_labels

def make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=6, control=False):
    """
    makes train test split with embeddings, also can create control datasets for control classifiers
    """
    print(f"making the train/test sets for layer {layer_num}", file=sys.stderr)
    if control==True:
        print("This is the control dataset", file=sys.stderr)
    print("---------------------------", file=sys.stderr)
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    if control == True:
        majority_labels_train = get_control_label_set(df, train_idx)
        majority_labels_test = get_control_label_set(df, test_idx)
        shuffed_labels_train = shuffle_labels(majority_labels_train)
        shuffed_labels_test = shuffle_labels(majority_labels_test)

    train_labs_orig = []
    train_labs_shuff = []
    for r in df.index:
        row = df.loc[r]
        noun = row["N1"]
        text = row["Sentence Raw Text"]
        label = 1 if row["True NPN"] == "Y" else 0
        tokenized_text, target_id = get_tokenized_input(row, tokenizer)
        embeddings_list = get_embeddings(model, tokenized_text, target_id)
        # print(len(embeddings_list), file=sys.stderr)
        current_embedding = embeddings_list[layer_num]
        # print(len(current_embedding))

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

def open_pert_df(pert):
    if pert == 1:
        pert_df = pd.read_csv("./data/raw_NPN_data_cleaned.tsv_perturbed_NNP.tsv", delimiter="\t")
        pert = 1
    if pert == 3:
        pert_df = pd.read_csv("./data/raw_NPN_data_cleaned.tsv_perturbed_NP.tsv", delimiter="\t")
        pert = 3
    if pert == 4:
        pert_df = pd.read_csv("./data/raw_NPN_data_cleaned.tsv_perturbed_PN.tsv", delimiter="\t")
        pert = 4
    if pert == 2:
        pert_df = pd.read_csv("./data/raw_NPN_data_cleaned.tsv_perturbed_PNN.tsv", delimiter="\t")
        pert = 2
    return pert_df


def make_pert_test_data(test_idx, model, tokenizer, pert=1, layer_num=6, control=False):
    pert_df = open_pert_df(pert)
    test_X = []
    test_y = []
    test_y_orig = []
    if control == True:
        majority_labels = get_control_label_set(pert_df, test_idx)
        shuffed_labels_test = shuffle_labels(majority_labels)

    for r in pert_df.index:
        if r in test_idx:
            row = pert_df.loc[r].copy()
            noun = row["N1"]
            orig_label = 1 if row["Orig Label"] == "Y" else 0
            label = 1 if row["True NPN"] == "Y" else 0 #should always be 0 for perturbed cases
            tokenized_text, target_id = get_tokenized_input(row, tokenizer, pert=pert)
            embeddings_list = get_embeddings(model, tokenized_text, target_id)
            # print(len(embeddings_list))
            current_embedding = embeddings_list[layer_num]
            test_X.append(current_embedding)
            if control == False:
                test_y_orig.append(orig_label)
            if control == True:
                new_orig_label = shuffed_labels_test[noun.lower()]
                test_y_orig.append(new_orig_label)

            test_y.append(label)
    return test_X, test_y, test_y_orig


def run_model(trainX, train_y, test_X, test_y, outfile = None, clf_type="LR"):
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

def load_split_indices(index_file):
    with open(index_file) as fin:
        data = json.load(fin)
    train_idx = data["Y_train"] + data["N_train"]
    test_idx = data["N_test"] + data["Y_test"]
    return train_idx, test_idx

def main(data_file, index_file, pert=None):
    """
    full pipeline
    """
    train_idx, test_idx = load_split_indices(index_file)
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    df = pd.read_csv(data_file, delimiter="\t")
    for i in range(0, 13):
        # out_file = "./outputs/predictions_layer_" + str(i) + f"_base_lt_{clf_type}.tsv"
        # out_file1 = "./outputs/classification_report_layer" + str(i) + f"_base_lt_{clf_type}.txt"
        # out_file_control = "./outputs/classification_report_layer" + str(i) + f"_base_lt_control_{clf_type}.txt"
        # columns = list(df.columns) + ["Pred", "Control Pred", "Control Gold"]
        # new_df = pd.DataFrame(columns=columns)
        trainX, train_y, test_X, test_y = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=i)
        trainX_control, train_y_control, test_X_control, test_y_control = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=i, control=True)

        clf_dict = {}
        clf_control_dict = {}
        for clf_type in ["LR"]: #add in MLPs later
            out_file = "./outputs/predictions_layer_" + str(i) + f"_base_lt_{clf_type}.tsv"
            out_file1 = "./outputs/classification_report_layer" + str(i) + f"_base_lt_{clf_type}.txt"
            out_file_control = "./outputs/classification_report_layer" + str(i) + f"_base_lt_control_{clf_type}.txt"
            columns = list(df.columns) + ["Pred", "Control Pred", "Control Gold"]
            new_df = pd.DataFrame(columns=columns)

            preds, clf = run_model(trainX, train_y, test_X, test_y, outfile=out_file1, clf_type=clf_type)
            preds_control, control_clf = run_model(trainX_control, train_y_control, test_X_control, test_y_control, outfile=out_file_control, clf_type=clf_type)

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


        for j in range(1, 5):
            print("working on pert datasets", file=sys.stderr)
            pert_test_X, pert_test_y, pert_test_y_orig = make_pert_test_data(test_idx, model, tokenizer, pert=j)
            pert_test_X_control, pert_test_y_control, pert_test_y_orig_control = make_pert_test_data(test_idx, model, tokenizer, pert=j, control=True)


            for clf_type in ["LR"]: #add in MLPs later
                clf = clf_dict[clf_type]
                control_clf = clf_control_dict[clf_type]
                pert_preds = list(clf.predict(pert_test_X))
                pert_preds_control = list(control_clf.predict(pert_test_X_control))
                # print(pert_preds,file=sys.stderr)
                print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL {clf_type}", classification_report(pert_test_y_orig, pert_preds))
                print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL {clf_type}", classification_report(pert_test_y, pert_preds))


                with open("./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) +  f"_lt_{clf_type}.txt", "w") as fout_pert:
                    print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL",
                          classification_report(pert_test_y_orig, pert_preds), file=fout_pert)
                    print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL",
                          classification_report(pert_test_y, pert_preds), file=fout_pert)

                f_out_pert_control_f = "./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) + f"_lt_control_{clf_type}.txt"

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
                out_data_name = "./outputs/perturbed/" + f"pert_predictions_layer_{i}_pert_{j}_lt_{clf_type}.tsv"
                new_pert_df.to_csv(out_data_name, index=False, sep="\t")

            # columns = list(df.columns) + ["Pred"]
            # new_pert_df = pd.DataFrame(columns=columns)

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
    args = parser.parse_args()
    main(args.data_file, args.index_file)


