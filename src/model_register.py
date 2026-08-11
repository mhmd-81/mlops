import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()

mlflow.set_tracking_uri('http://localhost:5000')

client = MlflowClient()
model_name = os.getenv('MODEL_NAME')


def candidate_model():
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

def eval_model():
    candidate = client.get_model_version_by_alias(
        model_name,
        'candidate'
    )
    model_uri = f"models/{model_name}@candidate"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_test)