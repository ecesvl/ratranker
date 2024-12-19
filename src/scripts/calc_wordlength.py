import json
import glob
import os.path

import numpy as np

from utils.paths import src_data_path


def count_words(text):
    return len(text.split())


def count_words_json(json_data, extra_words_count = 6):
    rationale_types = ['explanation', 'factuality', 'information_density', 'commonsense', 'textual_description']
    word_counts = {key: np.array([count_words(rationale) for rationale in section.values()])
                   for key, section in json_data.items() if key in rationale_types}
    sum_array = np.add(np.sum([v for v in word_counts.values()], axis=0), extra_words_count)

    return word_counts, sum_array


def calc_avg_words_json(sample_dir, data_path):
    folder_path = os.path.join(data_path, sample_dir, 'aug_results', 'aug_*.json')
    json_files = glob.glob(folder_path)

    # Variables to store cumulative counts
    all_word_counts = []
    sum_word_counts = []

    # Iterate over all JSON files and calculate averages
    for file_path in json_files:
        with open(file_path, 'r') as file:
            json_content = json.load(file)
            word_counts, total_word = count_words_json(json_content)

            values = [v for v in word_counts.values()]
            all_word_counts.append(values)
            sum_word_counts.append(total_word)

    all_word_counts = np.array(all_word_counts)
    sum_word_counts = np.array(sum_word_counts)

    avg_rationales = np.mean(all_word_counts, axis=(0, 2))
    std_rationales = np.std(all_word_counts, axis=(0, 2))

    avg_output_length = np.mean(sum_word_counts)
    std_output_length = np.std(sum_word_counts)

    # Map the results to the corresponding rationale types
    rationale_types = ['explanation', 'factuality', 'information_density', 'commonsense', 'textual_description']
    results_avg_rationales = dict(zip(rationale_types, avg_rationales))
    results_std_rationales = dict(zip(rationale_types, std_rationales))

    return results_avg_rationales, results_std_rationales, avg_output_length, std_output_length


if __name__ == '__main__':
    sample_dir = 'msmarco_50k'

    avg_rationales, std_rationales, avg_output_words, std_output_words = calc_avg_words_json(sample_dir, src_data_path)

    print(f"Average number of words per rationales:")
    for k, v in avg_rationales.items():
        print(f"\t{k}: {v:.2f} word(s)")

    print(f"Standard deviation of words per rationales:")
    for k, v in std_rationales.items():
        print(f"\t{k}: {v:.2f} word(s)")

    print(f"Average number of words per output: {avg_output_words:.2f}")
    print(f"Standard deviation of words per output: {std_output_words:.2f}")
