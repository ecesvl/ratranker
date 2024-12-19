import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
from pandas import DataFrame

from augment import AugmentConfig, Augmenter
from globals import ALL_RATIONALES_LIST
from src.utils.paths import src_data_path, data_path
from loader import load_dataset


def read_out_checkpoint(checkpoint_file: str) -> int:
    # Attempt to resume from checkpoint if it exists
    last_id = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            last_id = checkpoint_data.get('last_id', 0)
    return last_id


def save_rationales(result_dataset: DataFrame, results_path: str, last_id: int, start_id: int) -> None:
    save_path = os.path.join(results_path, f"aug_result_{start_id}-{last_id - 1}.json")
    result_dataset.to_json(save_path, indent=4)
    logger.debug(f"Saved rationales {start_id} to {last_id - 1}")


def update_checkpoint(checkpoint_file: str, last_id: int):
    # Update the checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_id': last_id}, f)
        logger.debug(f"Saved checkpoint- last_id processed {last_id}")


def augment_ds(dataset: DataFrame, rationales: List[str], results_path: str,
               part_size: int, batch_size: int, save_interval: int):
    # Checkpoint file to save the last index processed
    checkpoint_file = os.path.join(results_path, 'checkpoint.json')

    # Attempt to resume from checkpoint if it exists
    last_id = read_out_checkpoint(checkpoint_file=checkpoint_file)
    final_id = last_id + part_size + 1
    augment_config = AugmentConfig(few_shot=True, num_examples=5)  # TODO: make num_examples configurable
    augmenter = Augmenter(augment_config)
    temp_ds = pd.DataFrame()
    dataset_part = dataset.loc[last_id:last_id + part_size]

    start_id = last_id
    for augmented_batch in augmenter.augment_batches(dataset_part, batch_size=batch_size):
        # Process each augmented batch
        for response in augmented_batch:
            for rational in rationales:
                if rational != "information_density":
                    temp_ds.loc[last_id, rational] = getattr(response, rational, None)
                else:
                    temp_ds.loc[last_id, rational] = getattr(response, rational).value
            last_id += 1

            if last_id % save_interval == 0  or last_id == final_id:
                save_rationales(result_dataset=temp_ds, results_path=results_path, last_id=last_id,
                                start_id=start_id)

                # Update the checkpoint
                update_checkpoint(checkpoint_file=checkpoint_file, last_id=last_id)

                if last_id != final_id:
                    # set a delay after saving to avoid hitting rate limits
                    logger.debug("Introducing 150s delay to prevent hitting rate limit ")
                    time.sleep(150)

                # Reset the temporary dataset
                temp_ds = pd.DataFrame()
                start_id = last_id


def create_rationales_tsv_file(rationales: List[str], part_size: int, results_path: str,
                               batch_size: int, save_interval: int, sample_dir: str):
    samples_data_path = os.path.join(src_data_path, sample_dir, "samples.tsv")
    dataset = load_dataset(dataset_name='msmarco_tsv', file_path=samples_data_path)

    augment_ds(dataset=dataset, rationales=rationales, results_path=results_path,
               part_size=part_size, batch_size=batch_size, save_interval=save_interval)


def create_rationales(sample_dir: str, rationales: List[str], size: int, batch_size: int, save_interval):
    logger.debug('Create_rationales start')
    results_path = os.path.join(src_data_path, sample_dir, "aug_results")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    create_rationales_tsv_file(rationales, size, results_path, batch_size, save_interval, sample_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create rationales for a dataset.')

    # Add the arguments
    parser.add_argument('--samples-dir', type=str,
                        help='Name of folder,where samples are stored masterThesis/src/datasets/{samples_dir}')
    parser.add_argument('--rationales', type=str, nargs='+', choices=ALL_RATIONALES_LIST,
                        help='List of rationales to create', default=ALL_RATIONALES_LIST)
    parser.add_argument('--size', type=int, help='Size of the subset to process', default=200)
    parser.add_argument('--batch_size', type=int, help='Batch size to augment concurrently', default=20)
    parser.add_argument('--save-interval', type=int, help='Interval to save results', default=200)

    # Execute the parse_args() method
    args = parser.parse_args()

    if args.samples_dir is None:
        parser.error("--samples-dir must be specified")

    logger.info(f"Input arguments for creating rationales:\n"
                f"Sample dir: {args.samples_dir}, Rationales: {args.rationales}, Size: {args.size}")

    create_rationales(sample_dir=args.samples_dir, rationales=args.rationales, size=args.size,
                      batch_size=args.batch_size, save_interval=args.save_interval)
