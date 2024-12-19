import io
import os

import numpy as np
import pandas as pd

from src.evaluation.ms_marco_eval import load_candidate_from_stream, load_reference, compute_metrics
from src.utils.paths import src_data_path

"""
script for calculating the MRR@10 for the results of the
eval_preranked.py-script
- reorders the results read from the tsv files from the results_dir path
- conditional sorts them
"""
def load_rank_results(folder, base_name, ranges, dtype):
    """Loads multiple CSV files into a single DataFrame.

    Args:
        folder (str): The folder containing the CSV files.
        base_name (str): The base file name pattern (without indices).
        ranges (list of tuples): Start and end indices for the file ranges.
        dtype (dict): Dictionary defining data types for reading the CSV files.

    Returns:
        pd.DataFrame: A concatenated DataFrame of all loaded files.
    """
    data_frames = []
    for start, end in ranges:
        file_path = os.path.join(folder, f"{base_name}_{start}-{end}")
        df = pd.read_csv(file_path, dtype={'pid': np.int64, 'passage': str, 'score': np.float64, 'text': str, 'qid': np.int64})
        # Convert the column to boolean
        df['label'] = df['label'].apply(
            lambda x: str(x).strip().lower() == 'true' if pd.notnull(x) else False
        )
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


results_dir = r'C:\P\masterThesis\data\t5large-exp'
base_name = 'results-t5largeexp-a10'

ranges = [(0, 2000000), (2000000, 4000000), (4000000, 6000000), (6000000, 6974598)]
dtype = {'pid': np.int64, 'passage': str, 'score': np.float64, 'text': str, 'label': bool, 'qid': np.int64}

# Load all data
rank_results = load_rank_results(results_dir, base_name, ranges, dtype)

# concat_res = pd.concat([rank_res1, rank_res2, rank_res3, rank_res4, rank_res5, rank_res6], ignore_index=True)
rank_results = rank_results.sort_values(by=['qid'])
df = rank_results[['qid', 'pid', 'score', 'label']]

# Sort by qid first to group them
df = df.sort_values(by='qid', ascending=True)


# Define a function to apply conditional sorting within each qid group
def conditional_sort(group):
    # Sort True labels by descending score
    true_labels = group[group['label']].sort_values(by='score', ascending=False)
    # Sort False labels by ascending score
    false_labels = group[~group['label']].sort_values(by='score', ascending=True)
    # Concatenate the sorted parts with True labels first
    sorted_group = pd.concat([true_labels, false_labels])
    # Add a rank column starting from 1 within this sorted group
    sorted_group['rank'] = range(1, len(sorted_group) + 1)
    return sorted_group


# Apply conditional sorting within each qid group
df = df.groupby('qid', group_keys=False).apply(conditional_sort)

candidate_df = df[['qid', 'pid', 'rank']]
candidate_data_str = candidate_df.to_csv(sep='\t', index=False, header=False)

candidate_data_stream = io.StringIO(candidate_data_str)

qid_to_ranked_candidate_passages = load_candidate_from_stream(candidate_data_stream)
references = load_reference(os.path.join(src_data_path, "top1000eval", "qrels.msmarco-passage.dev-subset.tsv"))
metrics = compute_metrics(references, qid_to_ranked_candidate_passages)
print(metrics)
