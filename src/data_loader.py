import pandas as pd
import numpy as np
import json
import torch
import os

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Dataset
from PIL import Image


######################### IMPORT JSON TO DATAFRAME #########################

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


######################### MULTI HOT ENCODE LABELS #########################

def mhot_encoder(row: list) -> list:
  """Multi-hot encode labels into a binary array.

  Loops over a list of labels to check if it is in the given row.
  Appending 1 if true and 0 if false.
  Labels are preset to ['kohaku', 'sanke', 'showa', 'tancho', 'gin'].
  The columns containing the labels must be named 'multi_label'.

  Args:
    row(list): A row of a DataFrame.

  Returns:
    list: A list of binary encoded labels.
  """

  labels = ['kohaku', 'sanke', 'showa', 'tancho', 'gin']
  mhot_encoded = []

  for label in labels:
    mhot_encoded.append(1 if label in [x.lower() for x in row['multi_label']] else 0)

  return mhot_encoded


######################### MULTI HOT ENCODE A ROW #########################

def multi_hot_encode_row(df:pd.DataFrame):
  """Create a new column in a DataFrame with multi-hot encoded labels

  Applies the mhot_encoder to the given DataFrame to create a new column
  called 'mhe' that contains the multi-hot encoded labels.

  Args:
    df(pd.DataFrame): A DataFrame with a 'multi_label' column.

  Returns:
    pd.DataFrame: A DataFrame that now contains the new column 'mhe'.
  """
  df['mhe'] = df.apply(mhot_encoder, axis=1)
  return df


################### STRATIFIED TRAIN TEST VAL SPLIT ########################

def stratified_train_test_val(image_paths:pd.Series, mhe:pd.Series, train_testval_size=0.3,
                              test_val_size=0.5, print_shapes=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Train_test_val split with stratification for multi-label classification.

  Uses MultilabelStratifiedShuffleSplit to split X and y columns two times.
  First split into train and testval, and second split testval into test and val.
  Then rejoins X and y for each split subset to create a DataFrame.
  Prints the shape of split data and rejoined data.

  Args:
    image_paths(pd.Series): A column of a DataFrame that contains the image filepaths.
    mhe(pd.Series): A column of a DataFrame that contains the multi-hot encoded labels.
    train_testval_size(float): The test size for the first split between train and test/val.
    test_val_size(float): The test size for the second split between test and val.
    print_shapes(bool): If True, the shapes will be printed.s

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of 3 dataframes containing train, test and val.s
  """

  # Define X and y
  X = np.array(image_paths.tolist())
  y = np.array(mhe.tolist())

  # Define stratified splitter for train and testval split
  msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=train_testval_size, random_state=42)

  # Apply stratified split to X and y
  for train_index, testval_index in msss.split(X, y):
    if print_shapes:
      print("TRAIN:", train_index)
    X_train, X_testval = X[train_index], X[testval_index]
    y_train, y_testval = y[train_index], y[testval_index]

  # Define stratified splitter for test/val
  msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_val_size, random_state=42)

  # Apply stratified split to X_testval and y_testval
  for test_index, val_index in msss2.split(X_testval, y_testval):
    if print_shapes:
      print("TEST:", test_index, "VAL:", val_index)
    X_test, X_val = X[test_index], X[val_index]
    y_test, y_val = y[test_index], y[val_index]

  # Print the shapes if print=True
  if print_shapes:
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

  # Recombine the X and y for loading into tensors
  train_df = pd.DataFrame({'image_path': X_train, 'mhe': list(y_train)})
  test_df = pd.DataFrame({'image_path': X_test, 'mhe': list(y_test)})
  val_df = pd.DataFrame({'image_path': X_val, 'mhe': list(y_val)})

  # Check the shape of the recombined DataFrames
  if print_shapes:
    print(f"train_df: {train_df.shape}, test_df: {test_df.shape}, val_df {val_df.shape}")

  return train_df, test_df, val_df


################### TENSOR DATASET CLASS ########################

# create a custom multi-label image dataset class

class CustomMultiLabelDataset(Dataset):
  """Custom Dataset for multi-label image classification

  Args:

  """
  def __init__(self, dataframe, image_col, label_col, dir_path, transforms=None):
    self.dataframe = dataframe
    self.image_col = image_col
    self.label_col = label_col
    self.dir_path = dir_path
    self.transforms = transforms

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    # load and transform image
    img_path = self.dataframe.iloc[idx][self.image_col]
    full_img_path = self.dir_path + img_path
    image = Image.open(full_img_path).convert('RGB')
    if self.transforms:
      image = self.transforms(image)

    # convert label to tensor
    label = torch.tensor(self.dataframe.iloc[idx][self.label_col], dtype=torch.float32)

    return image, label


################### TENSOR CHECKER ########################

def tensor_checker(train_loader, test_loader, val_loader):
  for images, labels in train_loader:
      print(type(images), images.shape)
      print(type(labels), labels.shape)
      break

  for images, labels in test_loader:
      print(type(images), images.shape)
      print(type(labels), labels.shape)
      break

  for images, labels in val_loader:
      print(type(images), images.shape)
      print(type(labels), labels.shape)
      break
