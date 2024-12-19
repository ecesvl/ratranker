import glob
import json
import os
from typing import List

import pandas as pd
from transformers import PreTrainedTokenizer
from src.training.dataset import TrainingRationaleDataset

def _load_and_transform_json(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Transform the nested dictionary into a list of dictionaries with 'id' as a key
    transformed_data = []
    for key in json_data.keys():
        for sub_key, value in json_data[key].items():
            # Find if the 'id' already exists in the transformed_data
            existing_entry = next((item for item in transformed_data if item['id'] == int(sub_key)), None)
            if existing_entry:
                # If 'id' exists, update the existing dictionary with the new key-value pair
                existing_entry[key] = value
            else:
                # If 'id' does not exist, create a new dictionary for this 'id'
                transformed_data.append({'id': int(sub_key), key: value})

    return pd.DataFrame(transformed_data)


def _get_json_files_for_ids(samples_dir_path: str, id_list: list):
    id_list.sort()
    json_data_list = []
    loaded_files = set()

    for sample_id in id_list:
        file_index = sample_id // 200
        start_id = file_index * 200
        end_id = start_id + 199
        json_filename = f'aug_result_{start_id}-{end_id}.json'
        json_path = os.path.join(samples_dir_path, 'aug_results', json_filename)

        if os.path.exists(json_path) and json_path not in loaded_files:
            json_data = _load_and_transform_json(json_path)
            json_data_list.append(json_data)
            loaded_files.add(json_path)

    combined_json_data = pd.concat(json_data_list, ignore_index=True)
    combined_json_data = combined_json_data[combined_json_data['id'].isin(id_list)]
    return combined_json_data


def create_dataset_msmarco_for_ids(samples_dir: str, id_list: list, tokenizer: PreTrainedTokenizer):
    dbfs_path = "/dbfs/mnt/mbti-genai/ECCANLI/traindata/"  #TODO modifizierbar
    samples_dir_path = os.path.join(dbfs_path, samples_dir)
    samples_tsv_file = os.path.join(samples_dir_path, 'samples.tsv')
    data = pd.read_csv(samples_tsv_file, sep='\t')

    # Filter the data to only include the rows with the specified IDs
    data = data[data['id'].isin(id_list)]

    # Get the corresponding JSON data for all IDs
    json_data = _get_json_files_for_ids(samples_dir_path, id_list)

    # Create the dataset with the filtered data
    ds = TrainingRationaleDataset(data=data, tokenizer=tokenizer)
    return ds

