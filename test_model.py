## We test the loaded pipeline
import joblib
import pandas as pd

## load model
def load_model():
  pipeline = joblib.load("models/wine_clustering.joblib")
  print(pipeline)
  return pipeline

## Load the dataset
def load_dataset():
  df = pd.read_csv("data/winequality-white.csv", sep=";")
  ## We select the features and drop the quality
  X = df.drop(columns=["quality"])
  ## Rename columns to snale_case to match the train file
  X.columns = X.columns.str.lower().str.replace(" ", "_")
  ## Take one sample as example and predict the cluster
  sample = X.iloc[[0]]
  return X, sample

def predict_cluster(pipeline, X, sample):
  ## Predict cluster - this just for one sample
  pipeline = load_model()
  X, sample =  load_dataset()
  prediction = pipeline.predict(sample)
  print(f"Predicted cluster: {prediction[0]}")
  print(f"Wine sample: {sample}")

  ## Now we predict all the samples
  predictions = pipeline.predict(X)
  print(f"\nCluster counts: ")
  print(pd.Series(predictions).value_counts().sort_index())

def main():
  X, sample = load_dataset()
  pipeline = load_model()
  predict_cluster(pipeline, X, sample)

if __name__ == "__main__":
  main()