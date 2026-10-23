## We test the train module
import pandas as pd
import joblib
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
