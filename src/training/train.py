import os.path

import fire
import mlflow
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer)
from transformers.training_args import OptimizerNames

from dataset import get_train_dataset, DEFAULT_TRAINING_DATA_LOCATION
from training_utils import (calc_bertscore, calc_bleu, calc_bertscore_avg,
                            calc_rouge, CustomSeq2SeqTrainer)


def create_data_collator(model: AutoModelForSeq2SeqLM) -> DataCollatorForSeq2Seq:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8  # enable use of TensorCores on NVIDIA hardware with compute capability >=7.5 ???
    )
    return data_collator


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    logger.debug(f"Decoded Output: {decoded_preds}")

    result_rouge = calc_rouge(decoded_preds, decoded_labels)
    result_bertscore = calc_bertscore(decoded_preds, decoded_labels)
    result_bleu = calc_bleu(decoded_preds, decoded_labels)
    average_bertscore = calc_bertscore_avg(result_bertscore)

    # ngrams_results = calculate_precision_recall_ngrams(decoded_preds, decoded_labels, n=4)
    mlflow.log_metrics(result_rouge)
    mlflow.log_metrics(average_bertscore)
    mlflow.log_metrics(result_bleu)

    mlflow.log_dict(result_bertscore, "bertscore_results.json")
    return {"rouge": result_rouge, "bertscore": result_bertscore, "average_bertcore": average_bertscore,
            "bleu": result_bleu}


def set_tokenizer(model_name: str):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def set_model_name(model_name: str):
    global modelname
    modelname = model_name


def create_training_args(output_dir: str, lr: float = 3e-5, weight_decay: float = 0.01, epochs: int = 30,
                         batch_size: int = 8, optim: OptimizerNames = OptimizerNames.ADAMW_TORCH) \
        -> Seq2SeqTrainingArguments:
    gradient_accum_step = 128 // batch_size
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        logging_dir="logs",
        logging_strategy='epoch',
        predict_with_generate=True,
        save_strategy='epoch',
        save_total_limit=3,
        eval_strategy='no',
        optim=optim,
        gradient_accum_step=gradient_accum_step,
        report_to='mlflow',
        label_names=["labels"])


def train(model: AutoModelForSeq2SeqLM, train_dataset: Dataset, training_args: Seq2SeqTrainingArguments,
          data_collator: DataCollatorForSeq2Seq, eval_dataset: Dataset, use_custom_sampler: bool = False, ):
    if not use_custom_sampler:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            eval_dataset=eval_dataset
        )
    else:
        # !!! Custom Sampler does not work !!
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            eval_dataset=eval_dataset
        )
    try:
        trainer.train()
        model.generation_config.max_new_tokens = 120
        metrics = trainer.evaluate()
        print(metrics)
    finally:
        out_path = os.path.join(training_args.output_dir, 'results')
        trainer.model.save_pretrained(out_path)


def main(model_name_or_path: str, output_dir: str, lr_in: float = 3e-5, weight_decay_in: float = 0.01,
         epochs_in: int = 30, data_size: int = 31000, batch_size_in: int = 128,
         optim: OptimizerNames = OptimizerNames.ADAMW_TORCH,
         use_custom_sampler: bool = False, use_workaround_neg_passage: bool = True,
         logging_steps: int = 10000, path_to_training_data: str = DEFAULT_TRAINING_DATA_LOCATION,
         **_):
    try:
        # setup logging
        set_tokenizer(model_name_or_path)
        mlflow.transformers.autolog()
        lr = lr_in
        weight_decay = weight_decay_in
        epochs = epochs_in
        batch_size = batch_size_in

        model_id = model_name_or_path
        set_model_name(model_name_or_path)
        logger.info(f'Start loading model {model_id}...')
        n_samples = data_size // 2
        train_size = 30000

        train_dataset, val_dataset = get_train_dataset(dataset_name='msmarco', train_samples_dir="msmarco_50k",
                                                       n_samples=n_samples, tokenizer=tokenizer, train_size=train_size,
                                                       use_workaround_neg_passage=use_workaround_neg_passage,
                                                       path_to_training_data=path_to_training_data)

        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

        logger.info(f"Model loading finished")

        data_collator = create_data_collator(model)

        if optim not in [name.value for name in OptimizerNames]:
            raise ValueError(f'Invalid optimizer type: {optim}')

        training_args = create_training_args(output_dir=output_dir, lr=lr, weight_decay=weight_decay,
                                             epochs=epochs, batch_size=batch_size, optim=optim)
        logger.info("Training started...")
        train(model=model, data_collator=data_collator, training_args=training_args, train_dataset=train_dataset,
              eval_dataset=val_dataset, use_custom_sampler=use_custom_sampler)

    finally:
        mlflow.log_artifacts(output_dir)


if __name__ == '__main__':
    with mlflow.start_run(log_system_metrics=True):
        fire.Fire(main)
