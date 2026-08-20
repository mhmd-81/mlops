import pandas as pd
import pytest
from dotenv import load_dotenv
import os
from pathlib import Path

data_path = Path(__file__).parent.parent/"data"/"iris.csv"

expected_columns = {
    "sepal length (cm)": "float64",
    "sepal width (cm)": "float64",
    "petal length (cm)": "float64",
    "petal width (cm)": "float64",
    "target": "int64",
    "target_name": "object",
}

valid_outputs = {"setosa", "versicolor", "virginica"}


min_row_count = 100

numeric_ranges = {
    "sepal_length": (0, 10),
    "sepal_width": (0, 10),
    "petal_length": (0, 10),
    "petal_width": (0, 10),

}

@pytest.fixture(scope="module")
def data_existance():
    assert data_path.exists(), f"data file not found at {data_path}"
    return pd.read_csv(data_path)
def test_data_not_empty(data_existance):
    assert not data_existance.empty



def test_expected_columns_check(data_existance):
    missing = set(expected_columns) - set(data_existance.columns)
    assert not missing, f'missing expected columns {missing}'

def test_unexpected_columns(data_existance):
    unexpected = set(data_existance.columns) - set(expected_columns)
    assert not unexpected, 'unexpected columns found'