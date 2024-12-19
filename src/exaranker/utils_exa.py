import glob
from pathlib import Path
from typing import Tuple

import pandas as pd
import re
import pytorch_lightning as pl
import torch
import csv
import json
import os
import torch.utils.data as data
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from pyserini.search.lucene import LuceneSearcher

MAP_DATASET_INDEX={
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat'
}

MAP_INSTRUCTION={
    'explanation': 'Give an explanation.',
    'factuality': 'Give the factuality.',
    'information_density': 'Give the information density.',
    'commonsense': 'Give the commonsense.',
    'textual_description': 'Give the textual description.',
    'rationales': 'Create the rationales: explanation, factuality, information_density, commonsense, textual_description.'
}


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


def _create_dataset_msmarco(path_to_training_data: Path, samples_dir: str, n_samples: int, n_train:int, rationale: str,
                            tokenizer: PreTrainedTokenizer, workaround_neg_passage=False) -> Tuple[data.Dataset, data.Dataset]:
    """ Create a TrainingRationaleDataset for MSMARCO data """

    rationales_data = pd.DataFrame()
    data = pd.DataFrame()
    eval_size = n_samples - n_train

    training_datasets_dir = [samples_dir, samples_dir + "neg"] if workaround_neg_passage else [samples_dir]
    n_samples = n_samples // 2 if workaround_neg_passage else n_samples

    for dataset_dir in training_datasets_dir:
        samples_dir_path = os.path.join(path_to_training_data, dataset_dir)
        samples_tsv_file = os.path.join(samples_dir_path, 'samples.tsv')
        aug_results_dir = os.path.join(samples_dir_path, 'aug_results')

        temp_data = pd.read_csv(samples_tsv_file, sep='\t', nrows=n_samples)
        temp_rationales = get_rationales(path_aug_results=aug_results_dir, n_samples=n_samples)

        data = pd.concat([data, temp_data], ignore_index=True)
        rationales_data = pd.concat([rationales_data, temp_rationales], ignore_index=True)

    combined_data = data.join(rationales_data, lsuffix='_samples', rsuffix='_rationales')
    # Shuffle and split data into train and eval

    train_data, eval_data = train_test_split(combined_data, test_size=eval_size, random_state=42)

    # Create datasets
    train_dataset = MyDataset(data=train_data, tokenizer=tokenizer, rationale=rationale)
    eval_dataset = MyDataset(data=eval_data, tokenizer=tokenizer, rationale=rationale)

    return train_dataset, eval_dataset

def get_rationale_prompt(rationale):
    map_instruction = {
    "explanation": "Give an explanation.",
    "factuality": "Give the factuality.",
    "information_density": "Give the information density.",
    "textual_description": "Give the textual description.",
    "commonsense": "Give the commonsense.",
    }
    map_output = {
    "explanation": "Explanation:",
    "factuality": "Factuality:",
    "information_density": "Information density:",
    "textual_description": "Textual description:",
    "commonsense": "Give the commonsense:"
    }
    return map_instruction[rationale], map_output[rationale]

class MyDataset(data.Dataset):
    """
        Dataset for training and testing.
        Creates the input prompt and tokenizes it.

        Args:
            data: pd.DataFrame with columns [id_samples, label, query, passage, explanation,
                                    factuality, information_density, commonsense, id_rationales]
            tokenizer: PreTrainedTokenizer (google/flan-t5-... Tokenizer)
        """

    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, rationale:str):
        self._data = data
        self.tokenizer = tokenizer
        self.categorical_label = data["label"].to_list()
        self.rationale = rationale
        self.instruction_prompt, self.output_rationale_prompt = get_rationale_prompt(rationale) if rationale != "all" else (None, None)
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
            attributes = {key: item[key] for key in item.index if
                          key not in ['label', 'query', 'passage', 'id_rationales', 'id_samples']}
            rationales = ', '.join(attributes.keys())
            attribute_str = ' '.join([f"{key}: {value}" for key, value in attributes.items()])

            if self.rationale  != "all":
                input_str = f"Is the question: \"{query}\" answered by the document: \"{passage}\". {self.instruction_prompt}"
                one_rat_string = f"{self.output_rationale_prompt} {attributes[self.rationale]}"
                output_str = f"{label}. {one_rat_string}"

            else:
                input_str = f"Is the question: \"{query}\" answered by the document: \"{passage}\". Create the rationales: {rationales}."
                output_str = f"{label}. {attribute_str}"

            item1 = self.tokenizer(input_str, truncation=True, return_tensors='pt', max_length=512,
                                   padding='max_length')
            item2 = self.tokenizer(output_str, truncation=True, return_tensors='pt', max_length=512,
                                   padding='max_length')
            encodings.append(item1)
            labels.append(item2)

        return encodings, labels

    def __len__(self):
        return len(self.categorical_label)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.squeeze(enc.input_ids)
        attention_mask = torch.squeeze(enc.attention_mask)
        labels = torch.squeeze(self.labels[idx].input_ids)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": labels}


class MyModel(pl.LightningModule):

    def __init__(self, model, device, tokenizer, model_name):
        super(MyModel, self).__init__()
        self.model = model
        self.mydevice = device
        self.tokenizer = tokenizer
        self.mystep = 0
        self.model_name = model_name

    def training_step(self, batch, batch_nb):
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # loss
        loss = outputs[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self, *args, **kwargs):
        self.save_model()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.mystep = self.mystep + 1

        if self.mystep % 5000 == 0 and self.mystep > 0:

            directory = 'monoT5-' +self.model_name +'/chk/model-' + str(self.mystep)
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(directory + '/out2')

            print('saving model --------' + str(self.mystep))
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(),
                       'monoT5-'+self.model_name+'/chk/model-' + str(self.mystep) + '/pytorch_model.bin')
            model_to_save.config.to_json_file('monoT5-'+self. model_name +'/chk/model-' + str(self.mystep) + '/config.json')
            self.tokenizer.save_pretrained('monoT5-' + self.model_name + '/chk/model-' + str(self.mystep) + '/')

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # loss
        loss = outputs[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)

        self.model.train()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=0.01)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.25),  # 12 ou 70
            'name': 'log_lr'
        }
        return [optimizer], [lr_scheduler]

    def save_model(self):
        directory = 'monoT5-'+self.model_name+'/chk/model'
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/out2')

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), 'monoT5-'+self.model_name+'/chk/model/pytorch_model.bin')
        model_to_save.config.to_json_file('monoT5-'+self.model_name+'/chk/model/config.json')
        self.tokenizer.save_pretrained('monoT5-'+self.model_name+'/chk/model/')


class MyUtils():
    def __init__(self, data_name, tokenizer):
        self.data_name = data_name
        index = MAP_DATASET_INDEX[data_name]
        self.tokenizer = tokenizer
        self.searcher = LuceneSearcher.from_prebuilt_index(index)

    def get_query(self, id):
        file_topic = f'../baseline/topics/topics.{self.data_name}.tsv'

        tsv_file = open(file_topic)
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            query_id = row[0]
            query_text = row[1]
            if query_id == id:
                tsv_file.close()
                return query_text
        tsv_file.close()
        return 'query not found'

    def get_doc(self, id):
        doc = self.searcher.doc(id)
        doc_raw = doc.raw()

        return json.loads(doc_raw)['text']

    def gen_prompt(self, query, doc, rationale_name, max_tokens=512):
        # Step 1: Get the instruction prompt
        instruction_prompt = MAP_INSTRUCTION[rationale_name]  # 15 tokens max
        instruction_length = len(self.tokenizer.tokenize(instruction_prompt))

        # Step 2: Create the fixed parts of the prompt
        fixed_text = f'Is the question: "{query}" answered by the document: "'
        fixed_text_length = len(self.tokenizer.tokenize(fixed_text))

        # Step 3: Calculate remaining tokens for `doc`
        remaining_tokens = max_tokens - (
                    fixed_text_length + instruction_length + 5)  # +2 for closing quotes and spacing
        if remaining_tokens < 0:
            raise ValueError("Query and instruction exceed max token limit!")

        # Step 4: Truncate the `doc` to fit within the remaining token budget
        doc_tokens = self.tokenizer.tokenize(doc)
        if len(doc_tokens) <= remaining_tokens:
            truncated_doc = doc
        else:
            truncated_doc_tokens = doc_tokens[:remaining_tokens]
            truncated_doc = self.tokenizer.convert_tokens_to_string(truncated_doc_tokens)

        # Step 5: Construct the final prompt
        prompt_txt_rat = f'{fixed_text}{truncated_doc}"? {instruction_prompt}'
        return prompt_txt_rat
