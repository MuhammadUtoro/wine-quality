## Import libraries or package
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

## Load the dataset - we start with white wine
df = pd.read_csv("data/winequality-white.csv", sep=";")

## Overview and Descriptive Statistics
print(f"Dataset shape: {df.shape}")
print(df.info())
print(df.describe())

## Rewrite the columns with snake_case
df.columns = df.columns.str.replace(" ", "_")

## Remove quality from the dataset
X = df.drop(columns=["quality"])
print(f"Features for clustering: {X.columns.tolist()}")

## Build the pipeline
pipeline = Pipeline([
 ( "scaler", StandardScaler()),
 ("pca", PCA(n_components=5)),
 ("kmeans", KMeans(n_clusters=2, random_state=42, n_init=10))
])
print(pipeline)


## Train the pipeline
pipeline.fit(X)

## Assign each wine to a cluster
labels = pipeline.named_steps["kmeans"].labels_
print(f"Cluster size: {pd.Series(labels).value_counts().sort_index()}")

## Save the pipeline
joblib.dump(pipeline, "models/wine_clustering.joblib")
print(f"Saving pipline success!")