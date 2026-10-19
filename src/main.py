## This module is to orchestrate the ETL-Pipeline
from extract import extract_csv
from transform import transform_data
from load import save_csv

def main():
  ## Extract
  df = extract_csv(
      filepath="data",
      filename="winequality-red.csv",
      separator=";"
    )

  print(f"Original dataset shape: {df.shape}")

  ## Transform
  transformed_df = transform_data(df)

  ## Load
  output_path = save_csv(
    df=transformed_df,
    filepath="processed",
    filename="white_wine_processed.csv"
  )

  print(f"ETL pipeline completed successfully!")
  print(f"Saved to: {output_path}")

if __name__ == "__main__":
  main()