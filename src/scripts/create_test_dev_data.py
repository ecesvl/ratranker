import os
import tarfile

import pandas as pd
import loguru as logger

from utils.paths import src_data_path, eval_data_path


def load_dev_data():
    test_data_path = os.path.join(eval_data_path, 'top1000dev.tsv')
    top1000eval_df = pd.read_csv(test_data_path, sep='\t', header=None,
                                 names=['qid', 'pid', 'query', 'passage'])
    return top1000eval_df


def load_dev_qrels():
    qrels_path = os.path.join(eval_data_path, 'qrelsdev.tsv')
    qrels = pd.read_csv(qrels_path, sep='\t', header=None,
                        names=['qid', '0', 'pid', '1'])
    return qrels


if __name__ == '__main__':
    #Check if data path exists, where the top1000dev.tar.gz
    # is stored and qrelsdev.tsv is stored
    source_dir_path = os.path.join(eval_data_path)
    dest_dir_path = os.path.join(src_data_path, 'top1000eval')
    if not os.path.exists(source_dir_path):
        os.mkdir(source_dir_path)
        logger.info(f'Created {source_dir_path}')
        raise FileNotFoundError(f'Load top1000.dev.gz.tar and qrels.dev.tsv from https://microsoft.github.io/msmarco/')
    else:
        dev_data_path = os.path.join(source_dir_path, 'top1000dev.tar.gz')
        qrels_path = os.path.join(source_dir_path, 'qrels.dev.tsv')
        if not os.path.isfile(dev_data_path) and not os.path.isfile(qrels_path):
            raise FileNotFoundError(f"top1000dev.tar.gz or qrelsdev.tsv not found. Download from https://microsoft.github.io/msmarco/")
        else:
            os.mkdir(dest_dir_path) if not os.path.exists(dest_dir_path) else None
            if not os.path.exists(os.path.join(source_dir_path, 'top1000dev.tsv')):
                tar_file = tarfile.open(dev_data_path)
                file_name = tar_file.getnames()[0]
                tar_file.extractall(source_dir_path)
                tar_file.close()

                os.rename(os.path.join(source_dir_path, file_name), os.path.join(source_dir_path, 'top1000dev.tsv'))
            if not os.path.exists(os.path.join(source_dir_path, 'qrelsdev.tsv')):
                os.rename(os.path.join(source_dir_path, 'qrels.dev.tsv'), os.path.join(source_dir_path, 'qrelsdev.tsv'))

    top1000eval_df = load_dev_data()
    qrels = load_dev_qrels()
    # Assuming top1000eval_df is your dataframe
    # Group by 'qid' and aggregate 'pid' into a list
    result_df = top1000eval_df.groupby(['qid', 'query'])['pid'].apply(list).reset_index()

    # Rename the columns for clarity
    result_df.columns = ['qid', 'query', 'list_of_pids']

    # Group by 'qid' and aggregate 'pid' into a list in qrels
    qrels_aggregated = qrels.groupby('qid')['pid'].apply(list).reset_index()
    qrels_aggregated.columns = ['qid', 'label']

    # Merge the result_df with the aggregated qrels on 'qid'
    merged_df = pd.merge(result_df, qrels_aggregated, on='qid', how='left')

    target_path = os.path.join(src_data_path, 'top1000eval', 'top1000dev_df_new.csv')
    merged_df.to_csv(target_path, index=False)

    print("Dataframe saved successfully to 'top1000dev_df.csv'")
