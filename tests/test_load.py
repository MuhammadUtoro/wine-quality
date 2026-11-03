## Test file for load module
import os
import pytest
import pandas as pd
from src.load import save_csv

## Test the save_csv() if it is correctly load the data
def test_save_csv(tmp_path):
  filepath = tmp_path
  filename = "wine.csv"

  ## Arrange
  df = pd.DataFrame({
    "Acidity": [4.4, 4.1, 2.2, 3.4],
    "Volatile Acidity": [4.0, 3.2, 1.1, 2.2],
    "Quality": [3.0, 2.0, 1.0, 4.5]
  })

  ## Act
  saved_file = save_csv(df, filepath, filename)

  ## Assert
  assert os.path.isfile(saved_file)

## Test the makedirs
def test_save_csv_makedirs(tmp_path):
  ## We create the path which doesn't exist yet
  filepath = tmp_path / "output"
  filename = "wine.csv"

  ## Arrange
  df = pd.DataFrame({
    "Acidity": [4.4, 4.1, 2.2, 3.4],
    "Volatile Acidity": [4.0, 3.2, 1.1, 2.2],
    "Quality": [3.0, 2.0, 1.0, 4.5]
  })

  ## Act
  saved_file = save_csv(df, filepath, filename)

  ## Assert
  assert os.path.isdir(filepath)
  assert os.path.isfile(saved_file)

## Test the content of the data
def save_csv_match_content(tmp_path):
  filepath = tmp_path
  filename = "wine.csv"

  ## Arrange
  df = pd.DataFrame({
    "Acidity": [4.4, 4.1, 2.2, 3.4],
    "Volatile Acidity": [4.0, 3.2, 1.1, 2.2],
    "Quality": [3.0, 2.0, 1.0, 4.5]
  })

  ## Act
  saved_file = save_csv(df, filepath, filename)

  ## Read the data
  loaded_file = pd.read_csv(saved_file)

  ## Assert
  pd.testing.assert_frame_equal(df, loaded_file)