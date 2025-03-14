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
    counter["F"] = defaultdict(int)
    counter["I"] = defaultdict(int)
    counter["J"] = defaultdict(int)
    for r in df.index:
        row = df.loc[r]
        noun = row["N1"].lower()
        label = row["Subtype"]
        counter[label][noun] += 1
    print(sorted(counter["F"].items(), key= lambda x: x[1], reverse=True), file=sys.stderr)
    print(sorted(counter["I"].items(), key=lambda x: x[1], reverse=True), file=sys.stderr)
    print(sorted(counter["J"].items(), key=lambda x: x[1], reverse=True), file=sys.stderr)

    trimmed_total = 0
    for noun, count in counter["F"].items():
        if count >= 20:
            count = 20
        trimmed_total += count
    print("TRIMMED TOTAL F", trimmed_total, file=sys.stderr)

    trimmed_total = 0
    for noun, count in counter["I"].items():
        if count >= 20:
            count = 20
        trimmed_total += count
    print("TRIMMED TOTAL I", trimmed_total, file=sys.stderr)


    trimmed_total = 0
    for noun, count in counter["J"].items():
        if count >= 20:
            count = 20
        trimmed_total += count
    print("TRIMMED TOTAL J", trimmed_total, file=sys.stderr)

    f_indices = []
    i_indices = []
    j_indices = []

def vanilla_split(in_file, train_size=287):
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
    counter["F"] = []
    counter["I"] = []
    counter["J"] = []
    for r in df.index:
        row = df.loc[r]
        noun = row["N1"].lower()
        df.loc[r, "N1"] = row["N1"].lower()
        label = row["Subtype"]
        counter[label].append(noun)

    i_train_count = 0
    i_train_idx = []
    random.seed(1)
    i_nouns = sorted(list(set(counter["I"])))
    random.shuffle(i_nouns)
    #(Y_nouns)

    #get a split of y indices for training, no greater than 20 examples per N1 lemma
    while i_train_count < train_size:
        i_noun = i_nouns.pop()
        #print(Y_noun, file=sys.stderr)
        noun_idxs = list(df.loc[df["N1"] == i_noun].index)
        #print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)
        i_train_idx += sampled_idx
        i_train_count += len(sampled_idx)
        #print(y_train_count)
        # print(Y_nouns)


    i_test_size = 1019 - i_train_count
    # if train_size == 304:
    #     y_test_size = 1977 - y_train_count

    i_test_count = 0
    i_test_idx = []

    #get the correct test indices based on which were selected for training
    while i_test_count < i_test_size:
        i_noun = i_nouns.pop()
        #print(Y_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == i_noun) & (df["Subtype"] == "I") ].index)
        #print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)
        i_test_idx += sampled_idx
        i_test_count += len(sampled_idx)
        #print(y_test_count)


###################################

    j_train_count = 0
    j_train_idx = []
    random.seed(1)
    j_nouns = sorted(list(set(counter["J"])))
    random.shuffle(j_nouns)
    # (Y_nouns)

    # get a split of y indices for training, no greater than 20 examples per N1 lemma
    while j_train_count < train_size:
        j_noun = j_nouns.pop()
        # print(Y_noun, file=sys.stderr)
        noun_idxs = list(df.loc[df["N1"] == j_noun].index)
        # print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        # print(sampled_idx)
        j_train_idx += sampled_idx
        j_train_count += len(sampled_idx)
        # print(y_train_count)
        # print(Y_nouns)

    j_test_size = 964 - j_train_count
    # if train_size == 304:
    #     y_test_size = 1977 - y_train_count

    j_test_count = 0
    j_test_idx = []

    # get the correct test indices based on which were selected for training
    while j_test_count < j_test_size:
        j_noun = j_nouns.pop()
        # print(Y_noun, file=sys.stderr)
        noun_idxs = list(df.loc[(df["N1"] == j_noun) & (df["Subtype"] == "J")].index)
        # print(Y_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        # print(sampled_idx)
        j_test_idx += sampled_idx
        j_test_count += len(sampled_idx)
        # print(y_test_count)


################################



    #now do the same thing for "N" examples (not NPNs)
    f_nouns = sorted(list(set(counter["F"])))
    random.seed(3)
    random.shuffle(f_nouns)

    f_test_count = 0
    f_train_count = 0
    f_test_size = 72

    f_train_idx = []
    f_test_idx = []
    # print(f_nouns)
    while f_test_count < f_test_size:
        f_noun = f_nouns.pop()
        #print(n_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == f_noun) & (df["Subtype"] == "F") ].index)
        #print(n_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)

        f_test_idx += sampled_idx
        f_test_count += len(sampled_idx)
        # print(f_test_count)


    f_train_size = 359 - f_test_count

    while f_train_count < f_train_size:
        f_noun = f_nouns.pop()
        #print(n_noun, file=sys.stderr)
        noun_idxs = list( df.loc[ (df["N1"] == f_noun) & (df["Subtype"] == "F") ].index)
        #print(n_noun, noun_idxs)

        if len(noun_idxs) > 20:
            random.seed(1)
            sampled_idx = random.sample(noun_idxs, k=20)
        else:
            sampled_idx = noun_idxs
        #print(sampled_idx)

        f_train_idx += sampled_idx
        f_train_count += len(sampled_idx)
        #print(n_train_count)
        #print(n_nouns)

    print("I train", i_train_count)
    print("I test", i_test_count)
    print("F train", f_train_count)
    print("F test", f_test_count)
    print("J train", j_train_count)
    print("J test", j_test_count)
    id_dict = {}
    id_dict["I_train"] = i_train_idx
    id_dict["I_test"] = i_test_idx
    id_dict["F_train"] = f_train_idx
    id_dict["F_test"] = f_test_idx
    id_dict["J_train"] = j_train_idx
    id_dict["J_test"] = j_test_idx

    with open("./data/train_test_split_train_balanced_Y_big_yTest_Kat_Clean.json", "w") as fout:
        json.dump(id_dict, fout)

    return i_train_idx, i_test_idx, f_train_idx, f_test_idx, j_train_idx, j_test_idx

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