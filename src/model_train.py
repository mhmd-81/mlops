import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import mlflow 
import mlflow.sklearn


# mlflow configureation
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment('iris_classification')



load_dotenv()

data_path = os.getenv("DATA_PATH")

data = pd.read_csv(data_path)

X = data.drop(columns=['target','target_name'],axis=0)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,test_size=0.2,shuffle=True)
# X_train.to_csv('./data/X_train.csv')
# X_test.to_csv('./data/X_test.csv')
# y_train.to_csv('./data/y_train.csv')
# y_test.to_csv('./data/y_test.csv')

model = RandomForestClassifier(max_depth=2,random_state=42,n_estimators=100,verbose=1)
with mlflow.start_run():
    # train
    model.fit(X_train,y_train)
    # model eval
    y_pred = model.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)


    # log params
    mlflow.log_param('max_depth', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_param('n_estimators',100)

    # log metrics
    mlflow.log_metric('accuracy',accuracy)

    # log model
    mlflow.sklearn.log_model(
        model,
        name= 'random_forest_model',
        registered_model_name='random_forest_model_reg'
    )







