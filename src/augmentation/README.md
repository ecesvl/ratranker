<br />

# Augmentation
___

### 1) Download training raw data
* download the train triples small(05/03/2019) from https://microsoft.github.io/msmarco/
* https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz
* extract the gzip file and get save triples.train.small.tsv into project-folder RatRanker/data/raw/triples.train.small.tsv
* The `sample_data.py` script generates a specified number of samples from the MSMARCO triples.train.small.tsv dataset. It allows users to efficiently extract smaller subsets of data for training, testing, or experimentation.
* The generated samples are saved in RatRanker/src/datasets/<out_dir>
#### Arguments

| Argument       | Type   | Default   | Description                                                           |
|----------------|--------|-----------|-----------------------------------------------------------------------|
| `--out_dir`    | `str`  | `default` | Specifies the name of the output directory for the generated samples. |
| `--num_samples`| `int`  | `100`     | Specifies the number of samples to generate.                          |

#### Example
```bash
python sample_data.py --out_dir msmarco_train --num_samples 500
```
### 2) Creating rationales with GPT4o
* This script provides functionality to augment datasets by creating rationales for each entry, processing the data in parts, and periodically saving results. It supports checkpointing to resume processing in case of interruptions.
  * Processes large datasets in manageable parts (part_size).
  * Periodically saves results after a specified save_interval
  * Supports checkpointing to resume from the last processed entry.
  * Handles batch processing for efficient augmentation.
* Rationales: 

#### Directory structure
* **Input:**
  * The dataset must be stored in:
  * src/datasets/{sample_dir}/samples.tsv

* **Output:**
  * Results are saved in:
  * src/datasets/{sample_dir}/aug_results/aug_result_{start_idx}-{end_idx}.json 
  * Checkpoint is stored as:
  * src/datasets/{sample_dir}/aug_results/checkpoint.json
  
#### Usage

Run the script using the following command:

#### Arguments
### Parameters

| Argument        | Type        | Default             | Description                                                             |
|-----------------|-------------|---------------------|-------------------------------------------------------------------------|
| `sample_dir`    | `str`       | Required            | Specifies the folder where samples are stored: masterThesis/src/datasets/{samples_dir}.|
| `rationales`    | `List[str]` | ALL_RATIONALES_LIST | A list of rationales to create. Options: _explanation, factuality, information_density, commonsense, textual_description._|
| `size`          | `int`       | 200                 | Size of the subset of the dataset to process.                                     |
| `batch_size`    | `int`       | 20                  | The batch size for concurrent augmentation.                             |
| `save_interval` | `int`       | 200                 | Number of records to process before saving results.      |

```bash
python create_rationales.py --samples-dir <SAMPLES_DIR> [OPTIONS]