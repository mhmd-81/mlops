from mlflow import MlflowClient

Model_name = 'iris_classifier'
Alias = 'champion'
Version = '1'


def promote_model(model_name: str, version: str, alias: str):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=Model_name,
        alias=Alias,
        version=Version
    )

    print(
        f"Model '{model_name}' version {version} "
        f"promoted to alias '{alias}'."
    )


if __name__ == "__main__":
    promote_model(
        model_name=Model_name,
        version=Version,
        alias=Alias,
    )