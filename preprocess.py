import pandas as pd
import argparse
import sys
import re


def clean_data(data_file, output_file):
    """
    cleans data?
    """
    print("Starting the cleaning", file=sys.stderr)
    raw_df = pd.read_csv(data_file, delimiter="\t")
    columns = ["Orig Label", "Perturbation"] + list(raw_df.columns)
    clean_df = pd.DataFrame(columns=columns)
    for r in raw_df.index:
        row = raw_df.loc[r].copy() #copy row
        label = row["Subtype"]
        noun = row["N1"]
        text = row["Sentence Raw Text"]
        row["Perturbation"] = 0
        row["Orig Label"] = label
        if label not in ["B", "T"]:
            if len(text.split()) > 4: #at least 5 words roughly
                pattern = noun.lower() + " to " + noun.lower()
                # print(pattern)
                matcher = re.compile(pattern, flags=re.IGNORECASE)
                match = matcher.search(text)
                assert match
                #print("FOUND 1:", match)
                clean_df.loc[len(clean_df.index)] = row
    clean_df.to_csv(output_file, index=False, sep="\t")
    return

def perturb_data(cleaned_data_file: str, out_file: str, repl_strat = "NNP"):
    """
    creates perturbed versions of data
    cleaned_data - pointing to cleaned data tsv
    out_file - pointing to output directory of perturbed tsv
    repl_strat - should be "NNP", "PNN", "NP", or "PN"
    """
    print("PERTURBING DATA FOLLOWING PATTERN:", repl_strat, file=sys.stderr)
    clean_df = pd.read_csv(cleaned_data_file, delimiter="\t")
    columns = clean_df.columns
    perturbed_df = pd.DataFrame(columns=columns)
    out_file = out_file + repl_strat + ".tsv"
    for r in clean_df.index:
        row = clean_df.loc[r].copy()
        # print(row)
        text = row["Sentence Raw Text"]
        noun = row["N1"]
        label = row["Subtype"]
        pattern = noun.lower() + " to " + noun.lower()
        # print(pattern)
        matcher = re.compile(pattern, flags=re.IGNORECASE)
        match = matcher.search(text)
        assert match

        if repl_strat == "NNP":
            repl_pattern = noun.lower() + " " + noun.lower() + " " + "to"
            row["Perturbation"] = 1

        if repl_strat == "PNN":
            repl_pattern = "to " + noun.lower() + " " + noun.lower()
            row["Perturbation"] = 2

        if repl_strat == "NP":
            repl_pattern = noun.lower() + " " + "to"
            row["Perturbation"] = 3

        if repl_strat == "PN":
            repl_pattern = "to" + " " + noun.lower()
            row["Perturbation"] = 4

        new_text = re.sub(pattern, repl_pattern, text, flags=re.IGNORECASE)
        row["True NPN"] = "N"
        row["Sentence Raw Text"] = new_text
        perturbed_df.loc[len(perturbed_df.index)] = row


    perturbed_df.to_csv(out_file, sep="\t", index=False)

    return



def main(raw_data_file, output_file, repl_strat):
    clean_data(raw_data_file, output_file)
    pert_file_name = output_file + "_perturbed_"
    perturb_data(output_file, pert_file_name, repl_strat=repl_strat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file")
    parser.add_argument("-o", "--output_file")
    parser.add_argument("-rs", "--repl_strat", default="NNP")
    args = parser.parse_args()
    main(args.data_file, args.output_file, args.repl_strat)
