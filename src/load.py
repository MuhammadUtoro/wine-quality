import os
import pandas as pd

## Save transformed DF to csv-file
def save_csv(df, filepath, filename):
  ## Create directory if it is not exist yet
  os.makedirs(filepath, exist_ok=True)

  ## Build the output file path
  file = os.path.join(filepath, filename)

  ## Save to csv
  df.to_csv(file, index=False)

  print(f"File saved successfuly: {file}")
  return file