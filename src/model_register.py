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
    run_id = candidate.run_id
    run = client.get_run(run_id)
    accuracy = run.data.metrics["accuracy"]
    return accuracy


def promote_model():
    accuracy = eval_model()
    if accuracy >= 0.85:
        candidate = client.get_model_version_by_alias(
            model_name,
            'candidate'
        )
        client.set_registered_model_alias(
            model_name,
            "production",
            candidate.version
        )
    else:
        print(
            f"Model rejected. Accuracy = {accuracy:.4f}"
        )
candidate_model()
promote_model()