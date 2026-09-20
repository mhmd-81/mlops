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
    "sepal length (cm)": (0, 10),
    "sepal width (cm)": (0, 10),
    "petal length (cm)": (0, 10),
    "petal width (cm)": (0, 10),
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

def test_min_row_count(data_existance):
    assert len(data_existance) >= min_row_count, (
        f"Expected at least {min_row_count} rows, "
        f"but found {len(data_existance)}"
    )

def test_expected_columns(data_existance):
    for column, expected_dtype in expected_columns.items():
        actual_dtype = str(data_existance[column].dtype)

        if expected_dtype == "object":
            assert actual_dtype in ["object", "str"], (
                f"Column '{column}' expected dtype "
                f"{expected_dtype}, but got {actual_dtype}"
            )
        else:
            assert actual_dtype == expected_dtype, (
                f"Column '{column}' expected dtype "
                f"{expected_dtype}, but got {actual_dtype}"
            )
            
def test_valid_outputs(data_existance):
        acutal_targets = set(data_existance['target_name'].unique())
        assert valid_outputs == acutal_targets, (
            f"target values must be {valid_outputs}"
            f"but got {acutal_targets}"
        )
def test_feature_numeric_ranges(data_existance):
    for column, (minimum,maximum) in numeric_ranges.items():
        actual_min = data_existance[column].min()
        actual_max = data_existance[column].max()
        assert actual_min >= minimum, (
         f"{column} has minimum value {actual_min}"
    )
        assert actual_max <= maximum, (
            f"{column} has maximum value {actual_max}"

    )


def test_no_miss_values(data_existance):
     for column in expected_columns:
          assert not data_existance[column].isna().any() >= 1, (
               f'column {column} has null value'
          )

          