import glob
import os
import re
from typing import Tuple, List

import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

"""
constants and defaults
"""
DEFAULT_TRAINING_DATA_LOCATION = "../datasets/msmarco_50k"

class TrainingRationaleDataset(Dataset):
    """
    Dataset for training and testing.
    Creates the input prompt and tokenizes it.

    Args:
        data: pd.DataFrame with columns [id_samples, label, query, passage, explanation,
                                factuality, information_density, commonsense, id_rationales]
        tokenizer: PreTrainedTokenizer (google/flan-t5-... Tokenizer)
    """

    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self._data = data
        self.tokenizer = tokenizer
        self.categorical_label = data["label"].to_list()
        self.encodings, self.labels = self._tokenize_data()

    def _create_encodings(self, input_strs):
        # Tokenize a list of input strings in one go
        return self.tokenizer(input_strs, truncation=True, return_tensors='pt', max_length=512,
                              padding='max_length')

    def _create_labels(self, label_strs):
        # Tokenize a list of label strings in one go
        return self.tokenizer(label_strs, truncation=True, return_tensors="pt", max_length=512,
                              padding="max_length")

    def _tokenize_data(self):
        """ tokenize the input data and the labels (output) """
        encodings = []
        labels = []
        for idx in tqdm(range(self._data.shape[0])):
            item = self._data.iloc[idx]
            label = 'true' if item['label'] == 1 else 'false'
            query = item['query']
            passage = item['passage']
            attributes = {key: item[key] for key in item.index if key not in ['label', 'query', 'passage', 'id_rationales', 'id_samples']}
            rationales = ', '.join(attributes.keys())

            input_str_explanation = f"Is the question: \"{query}\" answered by the document: \"{passage}\". Give an explanation."
            input_str = f"Is the question: \"{query}\" answered by the document: \"{passage}\". Create the rationales: {rationales}."

            attribute_str = ' '.join([f"{key}: {value}" for key, value in attributes.items()])
            explanation_string = f"Explanation: {attributes['explanation']}"
            output_str = f"{label}. {attribute_str}"
            output_str_explanation = f"{label}. {explanation_string}"

            item1 = self.tokenizer(input_str, truncation=True, return_tensors='pt', max_length=512,
                                   padding='max_length')
            item2 = self.tokenizer(output_str, truncation=True, return_tensors='pt', max_length=512,
                                   padding='max_length')
            encodings.append(item1)
            labels.append(item2)

        return encodings, labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.squeeze(enc.input_ids)
        attention_mask = torch.squeeze(enc.attention_mask)
        labels = torch.squeeze(self.labels[idx].input_ids)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}


def get_train_dataset(dataset_name: str, train_samples_dir: str, n_samples: int, tokenizer: PreTrainedTokenizer,
                      train_size: int, path_to_training_data: str = DEFAULT_TRAINING_DATA_LOCATION,
                      use_workaround_neg_passage: bool = False) -> Tuple[Subset, Subset]:
    """ Prepare and load the training and validation datasets
        Args:
            dataset_name: name of the dataset
            train_samples_dir: location of the training data
            n_samples: number of samples
            tokenizer: tokenizer
            train_size: size of the training set
            path_to_training_data: location of the training data

        Returns:
            Tuple[Subset, Subset]: training and validation datasets
            :param use_workaround_neg_passage:
    """
    if dataset_name == 'msmarco':
        dataset = _create_dataset_msmarco(path_to_training_data, train_samples_dir,
                                          n_samples, tokenizer, workaround_neg_passage=use_workaround_neg_passage)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')
    # calculate the number of samples to include in the train and validation sets
    eval_size = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, val_dataset


def get_rationales(path_aug_results: str, n_samples: int) -> pd.DataFrame:
    """
       Load rationale data from JSON files in a specified directory.

       Args:
           path_aug_results (str): Path to the directory containing rationale JSON files.
           n_samples (int): Maximum number of samples to load.

       Returns:
           pd.DataFrame: DataFrame containing rationale data.
    """
    json_pattern = os.path.join(path_aug_results, 'aug_result_*.json')
    json_paths = sorted(glob.glob(json_pattern),
                        key=lambda x: int(re.search(r'aug_result_(\d+)-', os.path.basename(x)).group(1)))

    rationales_df = pd.DataFrame()

    for json_path in json_paths:
        match = re.search(r'aug_result_(\d+)-(\d+).json', os.path.basename(json_path))
        if not match:
            continue

        start_id, end_id = int(match.group(1)), int(match.group(2))
        # If the file's range is entirely beyond the max_samples, skip this file
        if start_id >= n_samples:
            break

        # If the file's range starts before max_samples, it may contain relevant datasets
        with open(json_path, 'r') as file:
            json_data = pd.read_json(file)
            json_data['id'] = range(start_id, start_id + len(json_data))

            if end_id >= n_samples:
                json_data = json_data[json_data['id'] < n_samples]

            rationales_df = pd.concat([rationales_df, json_data], ignore_index=True)

    return rationales_df


def _create_dataset_msmarco(path_to_training_data: str, samples_dir: str, n_samples: int,
                            tokenizer: PreTrainedTokenizer, workaround_neg_passage=False) -> Dataset:
    """ Create a TrainingRationaleDataset for MSMARCO data """

    rationales_data = pd.DataFrame()
    data = pd.DataFrame()

    training_datasets_dir = [samples_dir, samples_dir + "neg"] if workaround_neg_passage else [samples_dir]

    for dataset_dir in training_datasets_dir:
        samples_dir_path = os.path.join(path_to_training_data, dataset_dir)
        samples_tsv_file = os.path.join(samples_dir_path, 'samples.tsv')
        aug_results_dir = os.path.join(samples_dir_path, 'aug_results')

        temp_data = pd.read_csv(samples_tsv_file, sep='\t', nrows=n_samples)
        temp_rationales = get_rationales(path_aug_results=aug_results_dir, n_samples=n_samples)

        data = pd.concat([data, temp_data], ignore_index=True)
        rationales_data = pd.concat([rationales_data, temp_rationales], ignore_index=True)

    combined_data = data.join(rationales_data, lsuffix='_samples', rsuffix='_rationales')

    return TrainingRationaleDataset(data=combined_data, tokenizer=tokenizer)
