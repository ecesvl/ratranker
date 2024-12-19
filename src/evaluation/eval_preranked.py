import json
import time

import fire

import torch

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.augmentation.globals import ALL_RATIONALES_LIST
from src.utils.paths import model_src_path, src_data_path

from itertools import islice
import os


def batched(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


def create_test_data():
    test_data_path = os.path.join(src_data_path, 'top1000eval', 'top1000dev_df_new.csv')
    top1000eval_df = pd.read_csv(test_data_path, header=None,
                                 names=['qid', 'query', 'list_of_pids', 'label'], skiprows=1)
    print(top1000eval_df.head())
    return top1000eval_df


def sample_from_test_data(test_df, num=1):
    data = test_df.sample(n=num)
    return data


def load_trained_model_and_tokenizer(model_path, device):
    conf_path = os.path.join(model_path, "config.json")
    conf = json.load(open(conf_path, 'r'))
    model_id = conf["_name_or_path"]
    # model_id = conf["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return model, tokenizer

def load_model_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def create_input_prompt_exp(query, passage):
    return f"Is the question: \"{query}\" answered by the document: \"{passage}\" Give an explanation."

def create_input_txt(query, passage):
    rationales_str = ', '.join(ALL_RATIONALES_LIST)
    return f"Is the question: \"{query}\" answered by the document: \"{passage}\" Create the rationales: {rationales_str}."

def few_shot_input_txt(query, passage):
   return  f"""I have some examples:
       (1) Is the query: \"What are the symptoms of the flu?\" answered by the document: \"The flu often causes fever, chills, muscle aches, cough, and fatigue. In some cases, nausea may also occ$
        Yes. The passage lists symptoms associated with the flu, directly answering the question.
       (2) Is the query: \"What is the process for filing taxes?\" answered by the document: \"Many people prefer e-filing taxes due to convenience and faster processing times.\"?
        No. The passage discusses e-filing but does not explain the actual process for filing taxes.
       (3) Is the query: \"{query}\" answered by the document: \"{passage}\"?"""

def evaluate(model_path, data_path, query_start, query_end, batch_size, outfile, just_explanation=False,**_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pre_retrieved_passages_path = os.path.join(data_path, 'preranked_results')
    retrieved_res = pd.read_csv(os.path.join(pre_retrieved_passages_path, 'bm25v1-results.tsv'),
                                sep='\t', header=None, names=['qid', 'pid', 'rank'])
    retrieved_passages = pd.read_csv(os.path.join(pre_retrieved_passages_path, 'bm25v1-passages.tsv'),
                                     sep='\t', header=None, names=['pid2', 'passage'], skiprows=1)
    retrieved_passages = retrieved_passages.reset_index()
    retrieved_passages = retrieved_passages.iloc[query_start:query_end]
    retrieved_res = retrieved_res.iloc[query_start:query_end]

    model, tokenizer = load_trained_model_and_tokenizer(model_path, device)

    retrieved_df = pd.concat([retrieved_res, retrieved_passages], axis=1)
    retrieved_df = retrieved_df.drop(columns=['pid2', 'index'])

    #Group by 'qid'
    grouped = retrieved_df.groupby('qid')

    all_rank_results = []
    saved_rank_results = []

    # Load the test data to create a mapping of qid to query
    top1000eval_df = create_test_data()
    qid_to_query = dict(zip(top1000eval_df['qid'], top1000eval_df['query']))

    for qid, group in grouped:
        query = qid_to_query.get(qid)

        ranking_results = pd.DataFrame(columns=['pid', 'passage'])
        inputs = []
        for _, row in group.iterrows():
            pid = row['pid']
            passage_text = row['passage']
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
                                     output_scores=True,
                                     return_dict=True,
                                     return_dict_in_generate=True,
                                     max_new_tokens=2)

            rank_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            out_txt = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            output_texts.extend(out_txt)
            scores = rank_scores[:, 0]
            scores = torch.nn.functional.log_softmax(scores, dim=0)
            transition_scores.extend(scores.cpu().numpy())

            generated_tokens = outputs.sequences[:, 1:]
            output_label.extend(tokenizer.batch_decode(generated_tokens[:, 0]))

        ranking_results['score'] = transition_scores
        ranking_results['text'] = output_texts
        ranking_results['label'] = output_label
        ranking_results['qid'] = qid
        end_rerank = time.time()
        print(f'qid: {qid}, query: {query}, Time for rerank: {end_rerank - start_rerank}')


        # Add a 'rank' column
        #all_rank_results.append(ranking_results[['qid', 'pid']])
        saved_rank_results.append(ranking_results)


    saved_rank_results_df = pd.concat(saved_rank_results, ignore_index=True)
    out_path = f'{outfile}_{query_start}-{query_end}'
    saved_rank_results_df.to_csv(out_path, index=False)

if __name__ == '__main__':
    fire.Fire(evaluate)
