import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn

DATA_PATH = r'C:\Users\a.i04\Desktop\mlops\data\iris.csv'
MODEL_DIR = r'C:\Users\a.i04\Desktop\mlops\models'

mlflow.set_experiment('iris_classifier')


def train_model(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.log_params({
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42
    })

    rf_clf.fit(X_train, y_train)
    print('model trained successfully...')
    return rf_clf


def evaluate(X_test, y_test, rf_clf):
    predictions = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric('accuracy', accuracy)
    print(f'accuracy: {accuracy:.4f}')
    return accuracy


def save_model(rf_clf, model_dir, X_sample=None):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_model.joblib")
    
    joblib.dump(rf_clf, model_path)
    print(f'Model saved locally → {model_path}')

    mlflow.sklearn.log_model(
        sk_model=rf_clf,
        artifact_path="model",
        input_example=X_sample          
    )
    print('Model logged to MLflow!')


if __name__ == '__main__':

    data = pd.read_csv(DATA_PATH)
    data = data.drop('target_name', axis=1)   

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print('----- split details -----')
    print(f'X_train: {X_train.shape}')
    print(f'X_test:  {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_test:  {y_test.shape}')
    print('-------------------------')

    with mlflow.start_run():
        print('start training ...')
        rf_clf = train_model(X_train, y_train)
        evaluate(X_test, y_test, rf_clf)
        save_model(rf_clf, MODEL_DIR, X_sample=X_train.head())