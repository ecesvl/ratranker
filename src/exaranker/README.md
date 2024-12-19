# README

## monoT5 Training Pipeline

The code in this package is based on the https://github.com/unicamp-dl/ExaRanker.
It provides a training pipeline for fine-tuning the T5 model on the MSMARCO dataset.
The pipeline uses PyTorch Lightning for seq2seq model fine-tuning.
---

## Features
- **T5 Fine-Tuning**: Fine-tune T5 for the task of passage ranking.
- **Custom Dataset Handling**: Supports MSMARCO dataset preprocessing.
- **Logging**: Monitors learning rate and training progress.
---

## Usage

### Command Line
Run the training script with the following command:
```bash
python train.py \
    --model_id <model-name> \
    --new_model_name <output-model-name> \
    --rationale <rationale-string> \
    --num_epoch 30 \
    --verbose 1 \
    --batch_n 16
```

### Parameters
- `--model_id`: Name or path of the pre-trained T5 model to fine-tune (e.g., `t5-small`, `t5-base`).
- `--new_model_name`: Name for the fine-tuned model.
- `--rationale`: Rationales ('all', 'explanation', 'factuality', 'information_density', 'textual_description', 'commonsense').
- `--num_epoch`: Number of training epochs (default: 30).
- `--verbose`: Verbosity level (0 for silent, 1 for detailed output).
- `--batch_n`: Batch size (we use `accumulate_grad_batches`, therefore we divide 128 // batch_n)
---

## Workflow

### Data Preparation
The pipeline uses a custom `_create_dataset_msmarco` function to preprocess the MSMARCO dataset:
- Tokenizes input passages and labels using the T5 tokenizer.
- Splits data into training and evaluation datasets.

### Training
1. **Model Initialization**:
   - Loads the pre-trained T5 model and tokenizer.
   - Transfers the model to the appropriate device (CPU/GPU).

2. **Data Loading**:
   - Uses PyTorch DataLoader for efficient batch loading.

3. **Training Configuration**:
   - Utilizes PyTorch Lightning's `Trainer` for training and evaluation.
   - Supports gradient accumulation for large batch sizes.
---

## Example

Below is an example command to fine-tune a `t5-small` model:
```bash
python train.py \
    --model_id t5-small \
    --new_model_name my_t5_finetuned \
    --rationale "all" \
    --num_epoch 10 \
    --verbose 1 \
    --batch_n 32
```

---

## Project Structure
- `main_train.py`: Main training script.
- `utils_exa.py`: Utility functions for dataset preparation and model handling.
---

## Notes
- Ensure the MSMARCO dataset is downloaded and placed in the correct directory (`./src/datasets`).
- The `batch_n` parameter should be set based on available GPU memory.
- Checkpoints and logs are saved in the `monoT5-<new_model_name>/chk` directory.

---

## Acknowledgments
- Hugging Face Transformers Library
- PyTorch Lightning
- MSMARCO Dataset
---


