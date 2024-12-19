# generate N samples from MSMARCO triples train
import argparse
from random import seed, sample
import os
import json

import numpy as np
import pandas as pd


from src.utils.paths import data_path, src_data_path


def initialize_directories(outdir, indices_file):
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(indices_file):
        with open(indices_file, "r") as f:
            return set(json.load(f))
    return set()


def update_sampled_indices(indices_file, sampled_indices):
    with open(indices_file, "w") as f:
        json.dump(list(sampled_indices), f)


def sample_triples_train_small(num_samples_ds, outdir):

    outdir = os.path.join(src_data_path, outdir)
    triples_train_small = os.path.join(data_path, "triples.train.small.tsv")

    outfile = os.path.join(str(outdir), "samples.tsv")
    indices_file = os.path.join(str(outdir), "indices.tsv")

    total_lines = 39780811

    sampled_indices = initialize_directories(outdir, indices_file)

    # Calculate the number of new samples needed
    new_samples_needed = num_samples_ds - len(sampled_indices)
    if new_samples_needed <= 0:
        print("No new samples needed.")
        return

    seed(2 + len(sampled_indices))

    # Generate a list of unique random indices for sampling
    if total_lines <= num_samples_ds:
        raise ValueError("Number of samples requested exceeds the the number of available lines.")
    random_indices = set(sample(range(total_lines), new_samples_needed))

    random_indices.difference_update(sampled_indices)

    # Load the datasets into a pandas DataFrame
    df = pd.read_csv(triples_train_small, sep='\t', header=None,
                     skiprows=lambda x: x not in random_indices)

    # Update sampled indices
    sampled_indices.update(random_indices)

    # Alternate class_rel and select the relevant passage
    df['class_rel'] = np.where(df.index % 2 == 0, 0, 1)
    df['relevant_passage'] = np.where(df['class_rel'] == 1, df.iloc[:, 1], df.iloc[:, 2])

    # Select the required columns and rename them
    df = df[['class_rel', 0, 'relevant_passage']]
    df.columns = ['label', 'query', 'passage']

    if not os.path.exists(outfile):
        # Insert a new id column
        df.insert(0, 'id', range(len(df)))
        df.to_csv(outfile, sep='\t', index=False)
    else:
        last_id = pd.read_csv(outfile, sep='\t', usecols=[0]).iloc[-1, 0]
        df.insert(0, 'id', range(last_id + 1, last_id + len(df)))  # Insert a new id column, continuing from last_id
        df.to_csv(outfile, mode='a', sep='\t', index=False, header=False)

    # Save the updated set of sampled indices
    update_sampled_indices(indices_file, sampled_indices)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate N samples from MSMARCO triples train dataset.")
    parser.add_argument("--out_dir", type=str, default="default", help='Output directory for generated files.')
    parser.add_argument("--num_samples", type=int, default=100, help='Number of samples to generate.')
    args = parser.parse_args()

    # Parameters from arguments
    num_samples_ds = args.num_samples

    sample_triples_train_small(num_samples_ds, args.out_dir)
