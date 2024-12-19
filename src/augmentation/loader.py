import pandas as pd


def _load_tsv_file(data_file: str) -> pd.DataFrame:
    df = pd.read_csv(data_file, sep='\t')
    return df


def load_dataset(dataset_name: str, file_path: str = None) -> pd.DataFrame:
    """
    dataset_name: name of the dataset
    file_path: path to the dataset
    :rtype: object
    """
    match dataset_name:
        case "msmarco_tsv":
            try:
                res = _load_tsv_file(data_file=file_path)
                return res
            except FileNotFoundError:
                print("File not found or File is None")
        case _:
            raise ValueError(f"Unknown dataset {dataset_name}")
