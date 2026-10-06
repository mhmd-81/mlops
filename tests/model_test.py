import os
from pathlib import Path
import mlflow
import pandas as pd
from dotenv import load_dotenv

mlflow.set_tracking_uri('http://mlflow:5000')
base_dir = Path(__file__).parent.parent
env_dir = base_dir / 'src' / '.env'
load_dotenv(env_dir)
data_path = os.getenv("TEST_DATA_PATH")

data = pd.read_csv(data_path)
model_path = os.getenv('random_forest_model_reg')

# def test_mlflow_connection():
#     experiment = mlflow.get_experiment_by_name('iris_classification')

#     assert experiment is not None, (
#         "Could not find iris_classification experiment in MLflow"
#     )