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


def make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=6):
    print(f"making the train/test sets for layer {layer_num}", file=sys.stderr)
    print("---------------------------", file=sys.stderr)
    train_X = []
    train_y = []
    test_X = []
    test_y = []

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
            train_y.append(label)
        if r in test_idx:
            test_X.append(current_embedding)
            test_y.append(label)

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


def make_pert_test_data(test_idx, model, tokenizer, pert=1, layer_num=6):
    pert_df = open_pert_df(pert)
    test_X = []
    test_y = []
    test_y_orig = []
    for r in pert_df.index:
        if r in test_idx:
            row = pert_df.loc[r].copy()
            noun = row["N1"]
            orig_label = 1 if row["Orig Label"] == "Y" else 0
            label = 1 if row["True NPN"] == "Y" else 0
            tokenized_text, target_id = get_tokenized_input(row, tokenizer, pert=pert)
            embeddings_list = get_embeddings(model, tokenized_text, target_id)
            # print(len(embeddings_list))
            current_embedding = embeddings_list[layer_num]
            test_X.append(current_embedding)
            test_y_orig.append(orig_label)
            test_y.append(label)
    return test_X, test_y, test_y_orig


def run_model(trainX, train_y, test_X, test_y, outfile = None):
    print("running the model", file=sys.stderr)
    print("----------------", file=sys.stderr)
    clf = LogisticRegression(random_state=0, max_iter=10000)
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
        out_file = "./outputs/predictions_layer_" + str(i) + "_base_lt.tsv"
        out_file1 = "./outputs/classification_report_layer" + str(i) + "_base_lt.txt"
        columns = list(df.columns) + ["Pred"]
        new_df = pd.DataFrame(columns=columns)
        trainX, train_y, test_X, test_y = make_train_test_set(df, model, tokenizer, train_idx, test_idx, layer_num=i)
        preds, clf = run_model(trainX, train_y, test_X, test_y, outfile=out_file1)
        for j in range(1, 5):
            pert_test_X, pert_test_y, pert_test_y_orig = make_pert_test_data(test_idx, model, tokenizer, pert=j)
            pert_preds = list(clf.predict(pert_test_X))
            # print(pert_preds,file=sys.stderr)
            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL", classification_report(pert_test_y_orig, pert_preds))
            print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL", classification_report(pert_test_y, pert_preds))
            with open("./outputs/perturbed/classification_report_layer" + str(i) + "_perturb_" + str(j) +  "_lt.txt", "w") as fout_pert:
                print(f"CLASSIFICATION REPORT PERTURBATION {j}: ORIGINAL LABEL",
                      classification_report(pert_test_y_orig, pert_preds), file=fout_pert)
                print(f"CLASSIFICATION REPORT PERTURBATION {j}: ACTUAL (NEGATIVE) LABEL",
                      classification_report(pert_test_y, pert_preds), file=fout_pert)
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
                    p_count += 1
                    new_pert_df.loc[len(new_pert_df.index)] = row
            out_data_name = "./outputs/perturbed/" + f"pert_predictions_layer_{i}_pert_{j}_lt.tsv"
            new_pert_df.to_csv(out_data_name, index=False, sep="\t")

            # columns = list(df.columns) + ["Pred"]
            # new_pert_df = pd.DataFrame(columns=columns)

        count = 0
        for r in df.index:
            if r in test_idx:
                row = df.loc[r].copy()
                pred = preds[count]
                row["Pred"] = pred
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


