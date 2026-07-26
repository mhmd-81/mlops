import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

data_iris = pd.read_csv(r'C:\Users\a.i04\Desktop\mlops\data\iris.csv')
model_path = r'C:\Users\a.i04\Desktop\mlops\models'


data_iris.drop('target_name',axis=1,inplace=True)
X = data_iris.drop('target',axis=1)
y = data_iris['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True,test_size=0.2,random_state=42)

print('-----split details-----\n' 
f'X_train: {X_train.shape}\n'
f'X_test: {X_test.shape}\n'
f'y_train: {y_train.shape}\n'
f'y_test: {y_test.shape}')
print('---------------------------')
print('start training ...')

def train_model(X_train,y_train):
    rf_clf = RandomForestClassifier(n_estimators=100,
                                    random_state=42)
    rf_clf.fit(X_train,y_train)
    print('model trained successfully...')
    return rf_clf
    
def evaluate(X_test,y_test,rf_clf):
    predictions = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    print(f'accuracy: {accuracy:.4f}')

def save_model(rf_clf,model_path):
    model_path = os.path.join(model_path, "rf_model.joblib")
    joblib.dump(rf_clf,model_path)
    print('model saved!')

if __name__ == '__main__':
    rf_clf = train_model(X_train,y_train)
    evaluate(X_test,y_test,rf_clf)
    save_model(rf_clf,model_path)

