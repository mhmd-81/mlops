import pandas as pd
from sklearn.datasets import load_iris
import os

iris = load_iris(as_frame=True)
data = iris.frame

data['target_name'] = data['target'].map(
    {
        0:'setosa',
        1:'versicolor',
        2:'virginica'
    }
)

os.makedirs('data',exist_ok = True)
data.to_csv(r'data\iris.csv',index=False)

print(data.head())