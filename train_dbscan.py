## We will use DBCAN algorithm to train the model
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/winequality-white.csv")
MODEL_PATH = Path("models/wine_clustering_dbscan.joblib")

## Load the data
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
    raise ValueError("No features available!")
  return X

## Build Pipeline
def build_pipeline(eps, min_samples):
  pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dbscan", DBSCAN(eps=eps, min_samples=min_samples)),
  ])

  return pipeline

## Train the model
def train_model(X, eps=1.5, min_samples=5):
  ## Some checks for testing later
  pipeline = build_pipeline(eps, min_samples)
  pipeline.fit(X)

  return pipeline

## Evaluate the model
def evaluate_model(pipeline, X):
  labels = pipeline.named_steps["dbscan"].labels_
  # print(pd.Series(labels).value_counts().sort_index())
  clusters = set(labels) - {-1}
  n_clusters = len(clusters)
  n_noise = np.sum(labels == -1)
  noise_pct = (n_noise/len(labels)) * 100
  mask = labels != -1
  X_scaled = pipeline[:-1].transform(X)
  X_filtered = X_scaled[mask]
  labels_filtered = labels[mask]
  n_clusters = len(set(labels_filtered))
  score = None
  if n_clusters >= 2:
    score = silhouette_score(X_filtered, labels_filtered)
    print(f"Silhouette score: {score:.2f}")
  else:
    print(f"Silhouette score cannot be calculated: fewer than 2 clusters")
  print(f"Number of cluster: {n_clusters}")
  print(f"Noise percentage: {noise_pct.round(2)} %")

  ## Returning dictionary with the results
  return {
    "n_clusters": n_clusters,
    "noise_pct": noise_pct,
    "silhouette_score": score
  }

## We use hyperparameter tuning using GridSearch
def tune_model(X):
  results_list = []
  param_grid = {
    "eps": [1.5, 2.0, 2.5, 3.0],
    "min_samples": [5, 10, 15]
  }

  for params in ParameterGrid(param_grid=param_grid):
    pipeline = train_model(
      X,
      eps=params["eps"],
      min_samples=params["min_samples"]
    )
    results = evaluate_model(pipeline, X)
    results_list.append({
      **params,
      **results
    }
    )
    print(f"Testing parameters: {params}")

  results_df = pd.DataFrame(results_list)
  results_df = results_df.sort_values(
    by="silhouette_score",
    ascending=False,
    na_position="last"
  )
  print(results_df)
  return results_df

def main():
  df = load_data()
  X = prepare_features(df)

  pipeline = train_model(X)
  labels = evaluate_model(pipeline, X)
  tune_model(X)

## Run
if __name__ == "__main__":
  main()