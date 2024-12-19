import pathlib
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils_exa import _create_dataset_msmarco, MyModel

import fire


# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_main(model_id, new_model_name, rationale, num_epoch=30, verbose=1, batch_n=16):
    # Save command-line parameters to a JSON file
    params_save_file = f'cmd_params_{new_model_name}.json'
    cmd_params = {
        "model_id": model_id,
        "new_model_name": new_model_name,
        "rationale": rationale,
        "num_epoch": num_epoch,
        "verbose": verbose,
        "batch_n": batch_n
    }
    with open(params_save_file, "w") as f:
        json.dump(cmd_params, f, indent=4)
    print("Command-line parameters saved to cmd_params.json")

    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    # shape  dataset
    print(tokenizer.pad_token_id)

    # create datasets

    max_length = 512

    train_dataset, eval_dataset = _create_dataset_msmarco(path_to_training_data=pathlib.Path("./src/datasets"),
                                                          samples_dir="msmarco_50k", n_samples=30400,
                                                          tokenizer=tokenizer, workaround_neg_passage=True,
                                                          n_train=30000, rationale=rationale)

    # view one sample
    if verbose == 1:
        print()
        idx = 2
        x1 = train_dataset.__getitem__(idx)
        print()
        print('     input_ids shape: ', x1['input_ids'].shape)
        print('Attention_mask shape: ', x1['attention_mask'].shape)
        print('               Label: ', x1['label'].shape)
        print('\n Decode one sample:\n', tokenizer.decode(x1['input_ids']))
        print('\n Decode one sample label:\n', tokenizer.decode(x1['label']))
        print()

    # create data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers=0, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_n, num_workers=0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'gpu' if torch.cuda.is_available() else 'cpu'

    if verbose == 1:
        print()
        print('CUDA: ', torch.cuda.is_available())
        print()

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = 1

    model.to(device)
    model.train()

    accum_batch = 128 // batch_n

    num_batch = int(np.ceil(train_dataset.__len__() / batch_n))
    model_pl = MyModel(model, device, tokenizer, new_model_name)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(enable_checkpointing=False, log_every_n_steps=1,
                         default_root_dir='monoT5-' + new_model_name + '/chk',
                         accumulate_grad_batches=accum_batch, devices=num_gpus, accelerator=device_str,
                         max_epochs=num_epoch,
                         callbacks=[lr_monitor])

    trainer.fit(model_pl, train_dataloader, eval_dataloader)


if __name__ == '__main__':
    fire.Fire(train_main)