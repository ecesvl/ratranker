import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import json
from src.exaranker.utils_exa import MyUtils


class MyGenOut():
    def __init__(self):
        self.inicio = 1

    def run(self, model_n_in, dataset_name, rationale):
        os.system('clear')

        # control flags
        demo = 0
        # run verbose
        verbose = 0
        # run gpt api openai
        run_monoT5 = 1

        model_n = model_n_in

        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
        #tokenizer = T5Tokenizer.from_pretrained("chk/monoT5-" + str(model_n))
        model = T5ForConditionalGeneration.from_pretrained("chk/monoT5-" + str(model_n))

        if demo == 1:
            file_run = '../baseline/run.dl20demo.txt'
        else:
            file_run = f'../baseline/run/run.{dataset_name}.txt'

        id = 0

        func = MyUtils(dataset_name, tokenizer)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        print()
        print('CUDA: ', torch.cuda.is_available())
        print()

        model.to(device)

        with open(file_run, encoding='utf8') as f:
            for line in f:
                if id >= 0:

                    stringl = line.split()
                    query_id = stringl[0]
                    doc_id = stringl[2]
                    prompt_txt = func.gen_prompt(func.get_query(query_id), func.get_doc(doc_id), rationale_name=rationale)

                    if verbose == 1:
                        print('### ' + str(id))
                        print('Query ID: ' + query_id)
                        print(func.get_query(query_id))
                        print('Query DOC: ' + doc_id)
                        print(func.get_doc(doc_id))
                        print()
                        print(prompt_txt)

                    if run_monoT5 == 1:
                        item1 = tokenizer(prompt_txt, truncation=True, max_length=512, padding='max_length',
                                          return_tensors="pt")
                        input_ids, attention_mask = item1.input_ids.to(device), item1.attention_mask.to(device)

                        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                 max_new_tokens=2,
                                                 output_scores=True,
                                                 return_dict=True,
                                                 return_dict_in_generate=True)
                        # text_seq = ""
                        text_seq = tokenizer.decode(outputs.sequences[0][1:])
                        print(text_seq)
                        tokens_seq = tokenizer.convert_ids_to_tokens(outputs.sequences[0][1:])
                        tokens_seq = [s.replace('\u2581', '') for s in tokens_seq]

                        # Greedy decoding:
                        mask = outputs.sequences != tokenizer.pad_token_id
                        probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
                        prob_values, prob_indices = probs.max(dim=2)
                        score_seq = prob_values[0][:mask[0].sum()].tolist()

                        dict = {"text": text_seq, "tokens": tokens_seq, "scores": score_seq}

                        jsonf = json.dumps(dict)

                        fj = open("chk/monoT5-" + str(model_n) + f"/out{model_n}/output" + str(id) + ".json", "w")

                        fj.write(jsonf)
                        fj.close()

                        print(str(id) + ' - ' + str(model_n))

                id = id + 1

        f.close()

# class MyGenOut():
#     def __init__(self, batch_size=100):
#         self.inicio = 1
#         self.batch_size = batch_size  # Set batch size
#
#     def run(self, model_n_in, dataset_name, rationale):
#         os.system('clear')
#
#         # Control flags
#         verbose = 0
#         run_monoT5 = 1
#
#         model_n = model_n_in
#         if model_n == "t5large-128" or model_n == "t5largeexp-128":
#             tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
#         else:
#             tokenizer = T5Tokenizer.from_pretrained(f"./chk/monoT5-{model_n}")
#         model = T5ForConditionalGeneration.from_pretrained(f"./chk/monoT5-{model_n}")
#
#         file_run = f"../baseline/run/run.{dataset_name}.txt"
#         func = MyUtils(dataset_name, tokenizer)
#
#         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         print(f"\nCUDA: {torch.cuda.is_available()}\n")
#         model.to(device)
#
#         # Prepare output directory
#         output_dir = f"./chk/monoT5-{model_n}/out{dataset_name}"
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Read and process in batches
#         batch_queries = []
#         batch_doc_ids = []
#         batch_prompts = []
#         ids = []
#
#         with open(file_run, encoding='utf8') as f:
#             for id, line in enumerate(f):
#                 stringl = line.split()
#                 query_id = stringl[0]
#                 doc_id = stringl[2]
#                 prompt_txt = func.gen_prompt(func.get_query(query_id), func.get_doc(doc_id), rationale_name=rationale)
#
#                 # Accumulate batch
#                 batch_queries.append(query_id)
#                 batch_doc_ids.append(doc_id)
#                 batch_prompts.append(prompt_txt)
#                 ids.append(id)
#
#                 # If batch is full, process it
#                 if len(batch_prompts) == self.batch_size:
#                     self.process_batch(batch_prompts, batch_queries, batch_doc_ids, ids, model, tokenizer, device, output_dir, verbose)
#                     batch_queries, batch_doc_ids, batch_prompts, ids = [], [], [], []
#
#             # Process any remaining items in the last batch
#             if batch_prompts:
#                 self.process_batch(batch_prompts, batch_queries, batch_doc_ids, ids, model, tokenizer, device, output_dir, verbose)
#
#     def process_batch(self, batch_prompts, batch_queries, batch_doc_ids, ids, model, tokenizer, device, output_dir, verbose):
#         # Tokenize the batch
#         inputs = tokenizer(batch_prompts, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
#         input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
#
#         # Generate outputs
#         outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
#                                  max_new_tokens=2, output_scores=True, return_dict_in_generate=True)
#
#         for i, id in enumerate(ids):
#             # Decode and process output for each item
#             text_seq = tokenizer.decode(outputs.sequences[i][1:])
#             tokens_seq = tokenizer.convert_ids_to_tokens(outputs.sequences[i][1:])
#             tokens_seq = [s.replace('\u2581', '') for s in tokens_seq]
#
#             # Greedy decoding:
#             mask = outputs.sequences != tokenizer.pad_token_id
#             probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
#             prob_values, prob_indices = probs.max(dim=2)
#             score_seq = prob_values[i][:mask[i].sum()].tolist()
#
#             result_dict = {"text": text_seq, "tokens": tokens_seq, "scores": score_seq}
#             jsonf = json.dumps(result_dict)
#
#             output_file = os.path.join(output_dir, f"output{id}.json")
#             with open(output_file, "w") as fj:
#                 fj.write(jsonf)
#
#             if verbose:
#                 print(f"Processed ID: {id} - Query: {batch_queries[i]} - Doc: {batch_doc_ids[i]}")
#
