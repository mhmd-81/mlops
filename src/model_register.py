import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()

mlflow.set_tracking_uri('http://localhost:5000')

client = MlflowClient()
model_name = os.getenv('MODEL_NAME')



version = client.search_model_versions(
    f"name='{model_name}'"
)

latest_version = max(
    version,
    key = lambda x: int(x.version)
)



client.set_registered_model_alias(
    model_name,
    'candidate',
    latest_version.version
)


print(
    f"Model version {latest_version.version} "
    f"promoted to candidate"
)

