import pandas as pd
import pytest

from src.transform import(
    validate_data,
    standardize_columns,
    transform_data,
    EXPECTED_COLUMNS
)

## Create a dummy dataframe
def sample_wine_data():
    return pd.DataFrame(
    [[7.4]*len(EXPECTED_COLUMNS)],
    columns=EXPECTED_COLUMNS,
)

## Test validate_data() method
def test_validate_data_accepts_valid_data():
    ## Dummy data frame
    df = sample_wine_data()
    ## Use the method from transform module - validate_data()
    result = validate_data(df)
    ## Check
    assert not result.empty
    assert list(result.columns) == EXPECTED_COLUMNS

def test_validate_data_rejects_missing_columns():
    df = sample_wine_data().drop(columns=["quality"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_data(df)

## Test the standardize_column() method
def test_standardize_columns():
    ## Dummy data
    df = pd.DataFrame(
    [[7.4, 0.5, 3.4]],
    columns=["Fixed Acidity", "Volatile Acidity", "ph"]
    )
    result = standardize_columns(df)

    ## Check
    assert list(result.columns) == [
    "fixed_acidity",
    "volatile_acidity",
    "ph"
]

def test_standardize_column_not_modify_originals():
    df = pd.DataFrame(
    [[8.9]],
    columns=["ph"]
    )
    standardize_columns(df)

    assert list(df.columns) == ["ph"]

## Test the transform_data() method
def test_transform_data():
    df = sample_wine_data()

    result = transform_data(df)
    ## Check
    assert not result.empty
    assert "fixed_acidity" in result.columns
    assert "ph" in result.columns
    assert "quality" in result.columns
    assert len(result) == 1