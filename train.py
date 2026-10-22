## Import libraries or package
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

## We will refactor the code so it will be reusable
DATA_PATH = Path("data/winequality-white.csv")
MODEL_PATH = Path("models/wine_clustering.joblib")

## Load the dataset
def load_data(filepath=DATA_PATH):
  return pd.read_csv(filepath, sep=";")

## Prepare the features
def prepare_features(df):
  df = df.copy()
  df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

  if "quality" not in df.columns:
    raise ValueError("Dataset must contain a 'quality' column")

  X = df.drop(columns=["quality"])
  if X.empty or X.shape[1] == 0:
    raise ValueError("No features available")
  return X

## Build pipeline
def build_pipeline():
  pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("kmeans", KMeans(n_clusters=2, random_state=42, n_init=10))
  ])
  return pipeline

## Train model
def train_model(X):
  if len(X) < 5:
    raise ValueError("At least 5 samples are required for PCA")
  if X.shape[1] < 5:
    raise ValueError("At least 5 features are required for PCA")

  pipeline = build_pipeline()
  pipeline.fit(X)

  return pipeline

## Save model
def save_model(pipeline, filepath=MODEL_PATH):
  filepath = Path(filepath)
  filepath.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(pipeline, filepath)
  print(f"Model saved to: {filepath}")


## Run the training process
def main():
  df = load_data()
  print(f"Dataset shape: {df.shape}")
  print(f"Stats overview: {df.describe()}")

  X = prepare_features(df)
  print(f"Fetures for training: {X.columns.tolist()}")

  pipeline = train_model(X)
  labels = pipeline.predict(X)
  print(f"Cluster size:\n{pd.Series(labels).value_counts().sort_index()}")

  save_model(pipeline)

## Run
if __name__ == "__main__":
  main()

