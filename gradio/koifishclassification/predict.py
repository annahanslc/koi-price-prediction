import torch
import torch.nn as nn
from PIL import Image
from transforms import transform_basic
from torchvision import models
from torchvision.models import ResNet18_Weights


def predict_image(filepath):
    # Recreate the model architecture
    num_labels = 5
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_labels)

    # Load weights from best model
    model.load_state_dict(
        torch.load("best_model_unfrozen_ResNet18_augmentation.pth", map_location="cpu")
    )
    model.eval()

    # Get the transform
    transform = transform_basic()

    # Change image to tensor
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        output = model(input_batch)
        probs = torch.sigmoid(output)
        prediction = (probs > 0.5).int()

    # Get the labels
    LABELS = ["kohaku", "sanke", "showa", "tancho", "gin"]
    pred_list = prediction.squeeze(0).tolist()
    pred_labels = []

    for idx, pred in enumerate(pred_list):
        if pred == 1:
            pred_labels.append(LABELS[idx])

    return pred_labels
