<br />

# README

## Sequence-to-Sequence Training Pipeline

This repository provides a training pipeline for sequence-to-sequence models using the Hugging Face Transformers library.

---

## Features
- **Seq2Seq Training**: Train sequence-to-sequence models with customizable configurations.
- **Evaluation Metrics**: Compute BLEU, and BERTScore metrics.
- **Logging**: Integrated with MLflow for tracking metrics and artifacts.
- **Tokenization**: Automatic handling of tokenization and padding.
- **Pre-trained Models**: Load and fine-tune pre-trained models from the Hugging Face model hub.

---
Ensure you have access to the necessary datasets and pre-trained models (e.g., `msmarco`).
---

## Usage

### Command Line
Run the training script with the following command:
```bash
python train.py \
    --model_name_or_path <model-name> \
    --output_dir <output-directory> \
    --lr_in 3e-5 \
    --weight_decay_in 0.01 \
    --epochs_in 30 \
    --batch_size_in 128 \
    --data_size 31000 \
    --optim adamw_torch \
    --path_to_training_data <path-to-data>
```

### Parameters
- `--model_name_or_path`: The name or path of the pre-trained model (e.g., `google/flan-t5-large`).
- `--output_dir`: Directory where model checkpoints and results will be saved.
- `--lr_in`: Learning rate for training.
- `--weight_decay_in`: Weight decay for the optimizer.
- `--epochs_in`: Number of training epochs.
- `--batch_size_in`: Batch size per device.
- `--data_size`: Total size of the dataset.
- `--optim`: Optimizer to use (`adamw_torch`, `adamw_hf`, etc.).
- `--path_to_training_data`: Path to the training dataset.

---

## Training Structure
- `train.py`: Main training script.
- `dataset.py`: Utilities for loading and preprocessing datasets.
- `training_utils.py`: Helper functions for evaluation metrics and custom trainers.

---

## Metrics
The following metrics are computed during evaluation:
- **BLEU**: Evaluates translation quality using n-grams.
- **BERTScore**: Leverages pre-trained language models to measure semantic similarity.
---

## Logging and Monitoring

This project uses MLflow to track:
- Training metrics (e.g., loss, accuracy, evaluation metrics).

To view the MLflow UI:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

---

### Adding New Metrics
Update the `compute_metrics` function in `train.py` to include additional evaluation metrics.

---

## Example

Below is an example command for training a BART model:
```bash
python train.py \
    --model_name_or_path facebook/bart-large \
    --output_dir ./output \
    --lr_in 5e-5 \
    --weight_decay_in 0.01 \
    --epochs_in 10 \
    --batch_size_in 16 \
    --data_size 50000 \
    --optim adamw_torch \
    --path_to_training_data ./data/train.json
```

---

## Notes
- Ensure the dataset is preprocessed and tokenized correctly.
- Adjust the `max_new_tokens` parameter for generation length.
- GPU usage is automatically enabled with `device_map="auto"`.

---

## Acknowledgments
- Hugging Face Transformers Library
- MLflow for experiment tracking
---


