from collections import Counter
from typing import List, Iterator
import evaluate
import nltk
import numpy as np
import torch

from torch.utils.data import DataLoader, Sampler, Subset, BatchSampler
from transformers import Seq2SeqTrainer

"""
Functions for metric computation
"""


def generate_ngrams(tokens, n):
    return list(nltk.ngrams(tokens, n))


def calculate_precision_recall_ngrams(predictions, references, n):
    total_precision = 0
    total_recall = 0

    precisions = []
    recalls = []
    for reference, prediction in zip(references, predictions):
        reference_ngrams = generate_ngrams(reference, n)
        prediction_ngrams = generate_ngrams(prediction, n)

        reference_ngrams_count = Counter(reference_ngrams)
        prediction_ngrams_count = Counter(prediction_ngrams)

        matching_ngrams_count = sum((reference_ngrams_count & prediction_ngrams_count).values())

        precision = matching_ngrams_count / len(prediction_ngrams) if prediction_ngrams else 0
        recall = matching_ngrams_count / len(reference_ngrams) if reference_ngrams else 0

        precisions.append(precision)
        recalls.append(recall)

        # total_precision += precision
        # total_recall += recall

    precision = geo_mean_overflow(precisions)
    recall = geo_mean_overflow(recalls)

    return {"ngram": n, "ngram_precision": precision, "ngram_recall": recall}


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


def calc_bertscore(predictions: List[str], references: List[str]):
    bertscore = evaluate.load("bertscore")
    return bertscore.compute(predictions=predictions, references=references, lang="en")


def calc_bertscore_avg(bertscore_results: dict):
    # Calculate the average of each metric
    average_precision = sum(bertscore_results['precision']) / len(bertscore_results['precision'])
    average_recall = sum(bertscore_results['recall']) / len(bertscore_results['recall'])
    average_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])

    return {'bertscore_precision': average_precision,
            'bertscore_recall': average_recall,
            'bertscore_f1': average_f1}


def calc_rouge(predictions: List[str], references: List[str]):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def calc_bleu(predictions: List[str], references: List[str]):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)


"""
Custom Trainer and Sampler:
!!! NOT WORKING !!!
"""

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        balanced_sampler = BalancedBatchSampler(self.train_dataset, self.args.train_batch_size)
        batch_sampler = BatchSampler(balanced_sampler, self.args.train_batch_size, drop_last=False)
        train_dataloader = DataLoader(self.train_dataset, sampler=balanced_sampler)
        return train_dataloader


class BalancedBatchSampler(Sampler[List[int]]):
    """
    A custom sampler to create balanced batches with equal positive and negative samples.

    Args:
        dataset: The dataset or subset to sample from.
        batch_size: The size of each batch. Must be even to divide equally between positive and negative samples.
    """

    def __init__(self, dataset: Subset, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even to create balanced batches.")

        self.dataset = dataset
        self.batch_size = batch_size

        # Access the original dataset and subset indices
        if isinstance(dataset, Subset):
            self.indices = dataset.indices  # Subset indices
            original_dataset = dataset.dataset
        else:
            self.indices = list(range(len(dataset)))
            original_dataset = dataset

        # Retrieve labels for the subset
        self.labels = [original_dataset.categorical_label[i] for i in self.indices]

        # Split indices into positive and negative
        self.positive_indices = [i for i, label in zip(self.indices, self.labels) if label == 1]
        self.negative_indices = [i for i, label in zip(self.indices, self.labels) if label == 0]

        # Ensure classes are balanced
        if len(self.positive_indices) != len(self.negative_indices):
            raise ValueError("Classes are not balanced in the dataset subset.")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle both positive and negative indices
        pos_perm = torch.randperm(len(self.positive_indices)).tolist()
        neg_perm = torch.randperm(len(self.negative_indices)).tolist()

        # Form batches by interleaving positive and negative indices
        for i in range(0, len(pos_perm), self.batch_size // 2):
            pos_batch = pos_perm[i:i + self.batch_size // 2]
            neg_batch = neg_perm[i:i + self.batch_size // 2]
            for pos_idx in pos_batch:
                yield self.positive_indices[pos_idx]
            for neg_idx in neg_batch:
                yield self.negative_indices[neg_idx]

    def __len__(self) -> int:
        # Total number of batches
        return len(self.positive_indices) // (self.batch_size // 2)
