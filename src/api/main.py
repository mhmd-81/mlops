import mlflow
import mlflow.pyfunc
from fastapi import FastAPI

app = FastAPI(title='iris_model')
mlflow.set_tracking_uri("http://localhost:5000")


model = mlflow.pyfunc.load_model(
    "models:/random_forest_model_reg/1"
)

@app.get('/health')
def health_check():
    return{"status":"healthy"}

@app.get('/')
def home():
    return{'message':'welcome to iris app'}

# @app.post('/prediction')
# def model_prediction():

