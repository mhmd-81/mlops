import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import joblib


load_dotenv()

data_path = os.getenv("DATA_PATH")

data = pd.read_csv(data_path)

X = data.drop(columns=['target','target_name'],axis=0)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,test_size=0.2,shuffle=True)

model = RandomForestClassifier(max_depth=2,random_state=42,verbose=1)
model.fit(X_train,y_train)

joblib.dump(model,'./models/rf-clf.joblib')
