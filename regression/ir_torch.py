import json
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image

from ir_config import IRConfig

class ImageRegressionTorch:
  def __init__(
      self,
      filepath,
      image_col,
      target_col,
      config: IRConfig = IRConfig()
      ):

    self
    self.config = config
    self.filepath = filepath
    self.image_col = image_col
    self.target_col = target_col

    self.df = config.load_json_to_df(filepath)
    self.transforms = self.transform_img

  def transform_img(self):
    return v2.Compose(
      v2.Resize((224,224)),
      v2.ToTensor(),
      v2.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
    )

  class CustomDataset(Dataset):
    def __init__(self):
      self

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      dir_path = self.filepath

      # load and transform image
      img_path = self.df.iloc[idx][self.image_col]
      full_img_path = dir_path + img_path
      image = Image.open(full_img_path).convert('RGB')
      if self.transforms:
        image = self.transforms(image)

      # convert label to tensor
      label = torch.tensor(self.df.iloc[idx][self.target_col], dtype=torch.float32)

      return image, label
