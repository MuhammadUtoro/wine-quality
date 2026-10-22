## We test the train module
import pandas as pd
from train import prepare_features

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