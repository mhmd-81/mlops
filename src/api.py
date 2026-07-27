from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import pandas as pd

from src.schemas import IrisRequest, PredictionResponse

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0"
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.pyfunc.load_model(
    model_uri="models:/iris_classifier@champion"
)


@app.get("/health")
def root():
    return {"message": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: IrisRequest):

    df = pd.DataFrame([{
        "sepal length (cm)": request.sepal_length,
        "sepal width (cm)": request.sepal_width,
        "petal length (cm)": request.petal_length,
        "petal width (cm)": request.petal_width
    }])

    prediction = model.predict(df)

    return PredictionResponse(
        prediction=int(prediction[0])
    )