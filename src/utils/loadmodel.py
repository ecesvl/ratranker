import mlflow
from mlflow.tracking import MlflowClient

if __name__ == '__main__':
    mlflow.set_tracking_uri("databricks://GenAI")
    client = MlflowClient()


    local_path = client.download_artifacts(run_id="c0aadfe4923f40f9a2cba870bc0b1691", path="results",
                                           dst_path="C:\P\masterThesis\src\model\\t5-base-trained")
