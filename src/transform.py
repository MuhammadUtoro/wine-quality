## This module is to clean (if applicable) and to see overview of the data
import pandas as pd

# Expected columns in the original Wine Quality dataset
EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def validate_data(df):
    ## Check whether the input DataFrame has the expected structure
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    print("Data validation passed!")

    return df


def standardize_columns(df):
    ## Convert column names to lowercase snake_case
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

def transform_data(df):
    ## Validate and transform the dataset
    df = validate_data(df)
    df = standardize_columns(df)

    print("Transformation completed!")
    print(f"Transformed dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df