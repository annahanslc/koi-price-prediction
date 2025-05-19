from torchvision import transforms

# define the basic transform

def transform_basic():
  """Basic transform including resize, tensor, and normalization using ImageNet mean and std.

  Returns:
    torchvision.transforms.Compose: Composed image transform.
  """
  return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
  ])
