
import pathlib
import time

import fire
import ir_datasets
import numpy as np
import torch

import pandas as pd
from transformers import  T5Tokenizer, T5ForConditionalGeneration

from utils.globals import ALL_RATIONALES_LIST
from itertools import islice
import os

DBFS_PATH = "/dbfs/mnt/mbti-genai/ECCANLI"
LOCAL_DATA_PATH = "C:\P\masterThesis\data"
DATALIST = ['dl20', 'covid', 'nfc', 'dbpedia', 'news', 'robust04', 'fiqa']

MAPPING_IR_DATASET_NAME = {
    'dl20': "msmarco-passage/trec-dl-2020",
    'covid': "beir/trec-covid",
    'nfc': "beir/nfcorpus/test",
    'dbpedia': "beir/dbpedia-entity/test",
    'news': "beir/trec-news",
    'robust04': "trec-robust04",
    'fiqa': "beir/fiqa/test",
}


def load_data(data_name, local):
    beir_path = os.path.join(LOCAL_DATA_PATH, 'beir') if local else pathlib.Path(DBFS_PATH, 'beir')

    qrels = pd.read_csv(pathlib.Path(beir_path, f"qrels.{data_name}"), sep='\t', header=None)
    ranked_results = pd.read_csv(pathlib.Path(beir_path, f"rank_results_{data_name}"), sep='\s+', header=None,
                                 names=['qid', 'Q0', 'pid', 'rank', 'score', 'str_rank'])
    return ranked_results, qrels


def get_ir_dataset(data_name, local):

    ir_name = MAPPING_IR_DATASET_NAME.get(data_name)
    datasets = ir_datasets.load(ir_name)
    data_query = []
    data_doc = []
    for query in datasets.queries_iter():
        data_query.append([query.query_id, query.text])
    if data_name != 'dl20':
        for doc in datasets.docs_iter():
            data_doc.append([doc.doc_id, doc.text])
        doc_df = pd.DataFrame(data_doc, columns=['docid', 'text'])
    else:
        beir_path = pathlib.Path(LOCAL_DATA_PATH, 'beir') if local else pathlib.Path(DBFS_PATH, 'beir')
        docs_path = pathlib.Path(beir_path, 'docs_dl20')
        doc_df = pd.read_csv(docs_path, sep='\t', header=None, names=['docid', 'text'], dtype=str)

    doc_df.set_index('docid', inplace=True)
    query_df = pd.DataFrame(data_query, columns=['qid', 'text'])
    query_df.set_index('qid', inplace=True)
    return query_df, doc_df


def batched(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


def load_trained_model_and_tokenizer(model_path, device):
    # conf_path = os.path.join(model_path, "config.json")
    # conf = json.load(open(conf_path, 'r'))
    # model_id = conf["_name_or_path"]
    # model_id = conf["base_model_name_or_path"]
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    return model, tokenizer



def create_input_prompt_exp(query, passage):
    return f"Is the question: \"{query}\" answered by the document: \"{passage}\" Give an explanation."


def create_input_txt(query, passage):
    rationales_str = ', '.join(ALL_RATIONALES_LIST)
    return f"Is the question: \"{query}\" answered by the document: \"{passage}\" Create the rationales: {rationales_str}."



def evaluate(model_path, data_name, batch_size, local=True, just_explanation=False,**_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if str(data_name) not in DATALIST:
        raise ValueError(f"{data_name} is not a valid dataname. Must be one of {DATALIST}")

    ranked_results, qrels = load_data(data_name, local)
    query_df, doc_df = get_ir_dataset(data_name, local)

    model, tokenizer = load_trained_model_and_tokenizer(model_path, device)

    # #Group by 'qid'
    grouped = ranked_results.groupby('qid')

    all_rank_results = []
    saved_rank_results = []

    for qid, group in grouped:
        query = query_df.loc[str(qid), 'text']

        ranking_results = pd.DataFrame(columns=['pid', 'passage'])
        inputs = []
        for _, row in group.iterrows():
            pid = row['pid']
            # Retrieve the rows with the specified pid, removing duplicates if necessary
            unique_passages = doc_df.loc[doc_df.index == str(pid), 'text'].drop_duplicates()

            # If you only want one unique value, select the first occurrence
            if not unique_passages.empty:
                passage_text = unique_passages.iloc[0]
            else:
                passage_text = None  # Handle the case where no matching pid is found

            if just_explanation:
                input_text = create_input_prompt_exp(query, passage_text)
            else:
                input_text = create_input_txt(query, passage_text)
            new_row = {'pid': pid, 'passage': passage_text}
            ranking_results = ranking_results._append(new_row, ignore_index=True)
            inputs.append(input_text)
            #print(input_text)

        batches = list(batched(inputs, batch_size))

        output_texts = []
        transition_scores = []
        output_label = []
        start_rerank = time.time()

        for batch in batches:
            tokenized_input = tokenizer(batch, return_tensors='pt', max_length=512, padding=True, truncation=True)
            outputs = model.generate(input_ids=tokenized_input["input_ids"].to(device),
                                     attention_mask=tokenized_input["attention_mask"].to(device),
                                     output_scores=True,
                                     return_dict=True,
                                     return_dict_in_generate=True,
                                     max_new_tokens=2)
            len_output = len(outputs.sequences)
            for i in range(0, len_output):
                text_seq = tokenizer.decode(outputs.sequences[i][1:])
                tokens_seq = tokenizer.convert_ids_to_tokens(outputs.sequences[i][1:])
                tokens_seq = [s.replace('\u2581', '') for s in tokens_seq]

                mask = outputs.sequences != tokenizer.pad_token_id
                probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
                prob_values, prob_indices = probs.max(dim=2)
                score_seq = prob_values[i][:mask[0].sum()].tolist()
                label = {"true": True, "false": False}.get(tokens_seq[0].lower(), None)
                probability = np.exp(score_seq[0])
                if label is not None:
                    score = 1 + probability if label else 1 - probability
                else:
                    score = 0
                transition_scores.append(score)
                output_texts.append(text_seq)
                output_label.append(label)


        ranking_results['score'] = transition_scores
        ranking_results['text'] = output_texts
        ranking_results['label'] = output_label
        ranking_results['qid'] = qid
        end_rerank = time.time()
        print(f'qid: {qid}, query: {query}, Time for rerank: {end_rerank - start_rerank}')

        # Add a 'rank' column
        saved_rank_results.append(ranking_results)

    saved_rank_results_df = pd.concat(saved_rank_results, ignore_index=True)
    out_path = f'rank_results_{data_name}_exa'
    path_to_save = pathlib.Path(DBFS_PATH, out_path) if not local else pathlib.Path("C:\P\masterThesis\data\out", out_path)
    saved_rank_results_df.to_csv(path_to_save, index=False)


if __name__ == '__main__':
    fire.Fire(evaluate)
