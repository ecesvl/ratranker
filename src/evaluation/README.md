<br />

# README

---
## Evalution 

This package contains script required for evaluation.
---

### Scripts

#### MSMARCO DEV Small dataset
* we use `eval_preranked.py` to rerank the previousle retrieved 1000 passages per query

##### Command-Line Arguments

To use this script, pass the required arguments through the command line. Below is a breakdown of the key parameters:

| Argument         | Description                                                                                                                                                 | Default Value |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `model_path`      | Path to the directory containing the trained model.                                                                                                         | Required      |
| `data_path`       | Path to the directory containing the evaluation data. This must contain a directory `preranked_results` with contains a tsv-file named `bm25v1-results.tsv`. | Required      |
| `query_start`     | The starting index of queries to evaluate.                                                                                                                  | Required      |
| `query_end`       | The ending index of queries to evaluate.                                                                                                                    | Required      |
| `batch_size`      | Number of queries to process in a single batch.                                                                                                             | Required      |
| `outfile`         | File path where the evaluation results will be saved.                                                                                                       | Required      |
| `just_explanation`| If set to `True`, generates only explanations for the passage ranking.                                                                                      | `False`       |

##### Example

```bash
python evaluate.py --model_path="./trained_model" \
                   --data_path="./data" \
                   --query_start=0 \
                   --query_end=1000 \
                   --batch_size=16 \
                   --outfile="./results/evaluation.csv"
```

##### Output
The script outputs a CSV file containing the following columns:
- `pid`: Passage ID.
- `passage`: Passage text.
- `score`: The transition score for ranking.
- `text`: Generated output text.
- `label`: Generated labels for ranking.
- `qid`: Query ID.

* Then the `calculate_mrr_script.py` (which is not yet in its best version) is then used to calculate the mrr from the generated reranks.

#### BEIR benchmark

* implemented the `trec_eval.py`, to evaluate on some datasets from the BEIR benchmark -> **not working**
* using the `src/exaranker/eval` for evaluation

