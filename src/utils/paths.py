import os.path
from pathlib import Path

src_path = Path(__file__).parent.parent.absolute()

root_path = src_path.parent

src_data_path = os.path.join(src_path, 'datasets')
data_path = os.path.join(root_path, 'data')
models_path = os.path.join(root_path, 'model')
model_src_path = os.path.join(src_path, 'model')
eval_data_path = os.path.join(data_path, 'eval')
