## We test the train module
import pandas as pd
import joblib
import pytest
from pathlib import Path
from train import load_data,prepare_features,build_pipeline,train_model,save_model
from sklearn.utils.validation import check_is_fitted

## Test the load_data()
def test_load_data():
  DATA_PATH = Path("data/winequality-white.csv")
  df = load_data(DATA_PATH)

  assert df.shape == (4898, 12)

## Test the prepare_features() method
def test_prepare_features():
  ## Create a dummy dataset
  df = pd.DataFrame({
    "Fixed Acidity": [5.6, 5.9],
    "Alcohol": [6.5, 8.9],
    "Quality": [4.0, 5.0]
  })

  # Act: prepare the features
  X = prepare_features(df)

  # Assert: quality is removed and columns are standardized
  assert "quality" not in X.columns
  assert "fixed_acidity" in X.columns
  assert "alcohol" in X.columns
  assert X.shape == (2,2)

## Test if quality column is missing
def test_prepare_features_missing_quality():
  df = pd.DataFrame({
    "Fixed Acidity": [6.7, 7.8],
    "Alcohol": [3.3, 4.9],
    "pH": [5.6, 6.7]
  })

  with pytest.raises(ValueError):
    prepare_features(df)

## Test the pipeline build
def test_build_pipline():
  pipeline = build_pipeline()

  assert list(pipeline.named_steps.keys()) == [
    "scaler", "pca", "kmeans"
  ]

  assert pipeline.named_steps["pca"].n_components == 5
  assert pipeline.named_steps["kmeans"].random_state == 42

## Test the train
def test_train_model():
  DATA_PATH = Path("data/winequality-white.csv")
  df = load_data(DATA_PATH)
  X = df.drop(columns=["quality"])
  model = train_model(X)

  check_is_fitted(model)
  assert len(model.predict(X)) == len(X)

## Test train with too few samples
def test_train_model_too_few_samples():
  df = pd.DataFrame({
    "Fixed Acidity": [4.4, 4.5, 4.6],
    "pH": [4.5, 6.7, 7.0],
    "Alcohol": [3.3, 5.6, 6.5],
    "Volatile Acidity": [3.3, 3.5, 4.3],
    "Residual Sugar": [4.4, 3.3, 5.5],
    "quality": [1.0, 2.0, 3.0]
  })

  X = df.drop(columns=["quality"])
  with pytest.raises(ValueError):
    train_model(X)


## Test the save model
def test_save_model(tmp_path):
  model_path = tmp_path/"wine_clustering.joblib"
  DATA_PATH = Path("data/winequality-white.csv")
  df = load_data(DATA_PATH)
  X = df.drop(columns=["quality"])
  model = train_model(X)
  save_model(model, model_path)

  assert model_path.exists()

## Test save model
def test_load_model(tmp_path):
  model_path = tmp_path/"wine_clustering.joblib"
  DATA_PATH = Path("data/winequality-white.csv")
  df = load_data(DATA_PATH)
  X = df.drop(columns=["quality"])

  model = train_model(X)
  save_model(model, model_path)
  assert model_path.exists()
  loaded_model = joblib.load(model_path)
  assert len(loaded_model.predict(X)) == len(X)
