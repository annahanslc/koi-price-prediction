import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

#################### RESNET18 MODEL #####################

def get_resnet18_model(num_classes:int) -> nn.Module:
  """Load pretrained ResNet18 and replaces the final layer output with the number of classes.

  Loads pretrained ResNet18 weights.
  Replaces the last fully connected layer with the number of inputs as ResNet18, but
  changes the number of outputs to the number of classes in the multi-label classification.

  Args:
    num_classes(int): Number of output labels.

  Returns:
    nn.Module: Modified ResNet18 model.
  """
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model.to(device)


################# FROZEN RESNET18 MODEL ##################

def get_resnet18_model_frozen(num_classes:int) -> nn.Module:
  """Load pretrained ResNet18, replaces the final layer output, freezes all other layers.

  Loads pretrained ResNet18 weights.
  Replaces the last fully connected layer with the number of inputs as ResNet18.
  Changes the number of outputs to the number of classes in the multi-label classification.
  Freezes all other layers.

  Args:
    num_classes(int): Number of output labels.

  Returns:
    nn.Module: Modified ResNet18 model.
  """
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model.to(device)
