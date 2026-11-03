## Test file for extract.py
from src.extract import extract_csv
import os
import pandas as pd
import pytest

## Test should be a unit test - to test if the function actually works
## test_extract_csv() - To check if the function load csv file correctly
def test_extract_csv(tmp_path):
  filepath = tmp_path
  filename = "wine.csv"
  sep = ";"

  ## Arrange
  test_data = pd.DataFrame({
    "Alcohol": [3.3, 3.2, 3.1, 4.4],
    "Residual Sugar": [4.4, 4.3, 4.1, 3.3],
  })
  ## Import the dummy data to tmp_path
  test_data.to_csv(
    filepath / filename,
    sep=";",
    index=False
  )

  ## Act
  df = extract_csv(filepath, filename, separator=sep)

  ## Assert
  assert isinstance(df, pd.DataFrame)
  assert df.shape == (4,2)
  assert list(df.columns) == ["Alcohol", "Residual Sugar"]

## test_extract_csv_file_location() - to check if the path and csv file exist
def test_extract_csv_file_location(tmp_path):
  ## Arrange
  filepath = tmp_path
  filename = "not-wine.csv"
  file = os.path.join(filepath, filename)

  ## Act
  with pytest.raises(FileNotFoundError, match=f"cannot locate file at: {file}"):
    extract_csv(filepath, filename, separator=";")