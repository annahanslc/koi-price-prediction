import json
import pandas as pd


def load_json_to_df(filepath: str) -> pd.DataFrame:
  """Loads data from json file into a DataFrame.

  Accepts a filepath of a json file to load the data, then returns a DataFrame with the data.

  Args:
    filepath(str): Path of json file to load.

  Returns:
    pd.DataFrame: DataFrame of data from the json file.
  """

  with open(filepath, 'r') as f:
    data = json.load(f)

  return pd.DataFrame(data)
