## This module is responsible for extracting the data
import pandas as pd
import os

def extract_csv(filepath, filename, separator=";"):
  file = os.path.join(filepath, filename)

  if not os.path.isfile(file):
    raise FileNotFoundError(f"cannot locat file at: {file}")

  df = pd.read_csv(file, sep=separator)
  print(f"File loaded successfully!: {file}")

  return df