import pandas as pd
import pytest
from dotenv import load_dotenv
import os
from pathlib import Path

data_path = Path(__file__).parent.parent/"data"/"iris.csv"

expected_columns = {
    "sepal_length": "float64",
    "sepal_width": "float64",
    "petal_length": "float64",
    "petal_width": "float64",
    "species": "object",
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
def data():
    assert data_path.exists(), f"data file not found at {data_path}"
    return pd.read_csv(data_path)
def test_data_not_empty(data):
    assert not data.empty



