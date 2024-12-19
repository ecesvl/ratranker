import json
import os
import random

import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

from src.scripts.text_metrics_evaluation.datasets import create_dataset_msmarco_for_ids
from src.training.training_utils import calculate_precision_recall_ngrams
from src.utils.paths import models_path, src_data_path

QUALITY_LEVEL = {"worst": "google-t5/t5-small",
                 "trained": "google-t5/t5-small",
                 "moderate": "t5-base",
                 "perfect": "t5-base"}
PREDICTION_QUALITY = "trained"
DATA_EXISTS = True

if __name__ == '__main__':
    random.seed(31)
    ids_eval_data = random.sample(range(25000, 26000), 20)
    if not DATA_EXISTS:
        match PREDICTION_QUALITY:
            case "worst":
                tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
                model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
            case "trained":
                model_id = "google/flan-t5-small"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                conf_path = os.path.join(models_path, "t5-small-trained", "adapter_config.json")
                model_path = os.path.join(models_path, "t5-small-trained")
                conf = json.load(open(conf_path, 'r'))

                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cpu')
                model.eval()
            case "moderate":
                model_id = "t5-base"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                conf_path = os.path.join(models_path, model_id, "adapter_config.json")
                model_path = os.path.join(models_path, model_id)
                conf = json.load(open(conf_path, 'r'))

                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cpu')
                model.eval()
            case "perfect":
                tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
                model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")



        val_dataset = create_dataset_msmarco_for_ids("msmarco_50k", id_list=ids_eval_data, tokenizer=tokenizer)

        if PREDICTION_QUALITY == "perfect":
            predictions = val_dataset.output_text
        else:
            predictions = []
            for sample in val_dataset:
                input_ids = sample['input_ids'].unsqueeze(0)
                out = model.generate(input_ids=input_ids.cpu(), max_new_tokens=200)
                out = tokenizer.batch_decode(out.detach().cpu().numpy(), skip_special_tokens=True)
                predictions.append(out)

        predictions_dict = {id_val: prediction[0] for id_val, prediction in zip(ids_eval_data, predictions)}
        file_name = f'eval_metrics_data/{PREDICTION_QUALITY}_predictions.json'
        full_path = os.path.join(src_data_path, file_name)

        # Assuming predictions_dict is already defined
        with open(full_path, 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=4)
        print(predictions)

    else:
        for quality in QUALITY_LEVEL.keys():
            file_name = f'{quality}_predictions.json'
            file_path = os.path.join(src_data_path, "eval_metrics_data", file_name)
            with open(file_path) as json_file:
                preds = json.load(json_file)
            tokenizer = AutoTokenizer.from_pretrained(QUALITY_LEVEL.get(quality))
            val_dataset = create_dataset_msmarco_for_ids("msmarco_50k", id_list=ids_eval_data, tokenizer=tokenizer)

            rouge = evaluate.load("rouge")
            bertscore = evaluate.load("bertscore")
            bleu = evaluate.load("bleu")
            meteor = evaluate.load("meteor")

            preds = list(preds.values())
            labels = np.where(val_dataset.labels != -100, val_dataset.labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            result_rouge = rouge.compute(predictions=preds, references=decoded_labels)
            result_bertscore = bertscore.compute(predictions=preds, references=decoded_labels, lang="en", model_type="distilbert-base-uncased")
            labels = [[s] for s in labels]
            result_bleu = bleu.compute(predictions=preds, references=decoded_labels, max_order=5)
            result_meteor = meteor.compute(predictions=preds, references=decoded_labels)

            # Calculate the average of each metric
            average_precision = sum(result_bertscore['precision']) / len(result_bertscore['precision'])
            average_recall = sum(result_bertscore['recall']) / len(result_bertscore['recall'])
            average_f1 = sum(result_bertscore['f1']) / len(result_bertscore['f1'])


            average_bertscore = {
                'bertscore_precision': average_precision,
                'bertscore_recall': average_recall,
                'bertscore_f1': average_f1
            }

            ngrams_results = calculate_precision_recall_ngrams(preds, decoded_labels, n=1)

            print(f"Evaluation Results for data quality level {quality}")
            print(f"ROUGE: {result_rouge}")
            print(f"BERT-score: {average_bertscore}")
            print(f"n-grams: {ngrams_results}")
            print(f"BLEU: {result_bleu}")
            print(f"Meteor: {result_meteor}")



