import sys
import argparse
from collections import defaultdict, Counter
import pandas as pd
import random
import json

def print_word_stats(in_file):
    """
    prints counters for the words in "Y" and "N" categories
    """
    df = pd.read_csv(in_file, delimiter="\t")
    counter = {}
    counter["Y"] = defaultdict(int)
    counter["N"] = defaultdict(int)
    for r in df.index:
        row = df.loc[r]
        noun = row["N1"].lower()
        label = row["True NPN"]
        counter[label][noun] += 1
    # print(sorted(counter["N"].items(), key= lambda x: x[1], reverse=True), file=sys.stderr)
    # print(sorted(counter["Y"].items(), key=lambda x: x[1], reverse=True), file=sys.stderr)

    trimmed_total = 0
    for noun, count in counter["Y"].items():
        if count >= 20:
            count = 20
        trimmed_total += count
    print("TRIMMED TOTAL Y", trimmed_total, file=sys.stderr)

    trimmed_total = 0
    for noun, count in counter["N"].items():
        if count >= 20:
            count = 20
        trimmed_total += count
    print("TRIMMED TOTAL N", trimmed_total, file=sys.stderr)

    positive_indices = []
    negative_indices = []

def vanilla_split(in_file, train_size=304):
    """
    splits into train/test where no word is seen in both splits and no lemma represented >20 times
    true negative examples total = 380 train_size = 304, test_size = 76
    positive_examples total = 1977, train_size = 304, test_size = 304?
    positive_Examples total = 1977, train_size = 1582 test_size = 395?
    """
    print("________________________")
    print("Splitting")
    df = pd.read_csv(in_file, delimiter="\t")
    counter = {}
    counter["Y"] = []
    counter["N"] = []
    for r in df.index:
        row = df.loc[r]
        noun = row["N1"].lower()
        df.loc[r, "N1"] = row["N1"].lower()
        label = row["True NPN"]
        counter[label].append(noun)

    y_train_count = 0
    y_train_idx = []
    random.seed(1)
    Y_nouns = sorted(list(set(counter["Y"])))
    random.shuffle(Y_nouns)
    #(Y_nouns)

    #get a split of y indices for training, no greater than 20 examples per N1 lemma
    while y_train_count < train_size:
        Y_noun = Y_nouns.pop()
        #print(Y_noun, file=sys.stderr)
        noun_idxs = list(df.loc[df["N1"] == Y_noun].index)
        #print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)
        y_train_idx += sampled_idx
        y_train_count += len(sampled_idx)
        #print(y_train_count)
        # print(Y_nouns)


    y_test_size = 1977 - y_train_count
    if train_size == 304:
        y_test_size = 76

    y_test_count = 0
    y_test_idx = []

    #get the correct test indices based on which were selected for training
    while y_test_count < y_test_size:
        Y_noun = Y_nouns.pop()
        #print(Y_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == Y_noun) & (df["True NPN"] == "Y") ].index)
        #print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)
        y_test_idx += sampled_idx
        y_test_count += len(sampled_idx)
        #print(y_test_count)


    #now do the same thing for "N" examples (not NPNs)
    n_nouns = sorted(list(set(counter["N"])))
    random.seed(3)
    random.shuffle(n_nouns)
    n_test_count = 0
    n_train_count = 0
    n_test_size = 76

    n_train_idx = []
    n_test_idx = []

    while n_test_count < n_test_size:
        n_noun = n_nouns.pop()
        #print(n_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == n_noun) & (df["True NPN"] == "N") ].index)
        #print(n_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)

        n_test_idx += sampled_idx
        n_test_count += len(sampled_idx)
        #print(n_test_count)

    n_train_size = 380 - n_test_count

    while n_train_count < n_train_size:
        n_noun = n_nouns.pop()
        #print(n_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == n_noun) & (df["True NPN"] == "N") ].index)
        #print(n_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)

        n_train_idx += sampled_idx
        n_train_count += len(sampled_idx)
        #print(n_train_count)
        #print(n_nouns)

    print("Y train", y_train_count)
    print("Y test", y_test_count)
    print("N train", n_train_count)
    print("N test", n_test_count)
    id_dict = {}
    id_dict["Y_train"] = y_train_idx
    id_dict["Y_test"] = y_test_idx
    id_dict["N_train"] = n_train_idx
    id_dict["N_test"] = n_test_idx

    with open("./data/train_test_split_train_full_Y_vsmall_Test.json", "w") as fout:
        json.dump(id_dict, fout)

    return y_train_idx, y_test_idx, n_train_idx, n_test_idx

def perturbed_split(in_file1, in_file2):
    """
    splits
    """



if __name__ == "__main__":
    print("Hello", file=sys.stderr)
    parser = argparse.ArgumentParser(description="argument parser for train test split")
    parser.add_argument("-f", "--file", required=True)
    args = parser.parse_args()
    print_word_stats(args.file)
    vanilla_split(args.file)