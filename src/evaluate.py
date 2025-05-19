import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(model, val_loader, device):
  # calculate the validation metrics

  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
      for images, labels in val_loader:
          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
          probs = torch.sigmoid(outputs)
          preds = (probs > 0.5).int()

          all_preds.append(preds.cpu())   # moves predictions and labels from GPU to CPU
          all_labels.append(labels.cpu().int()) # convert labels to int (for metrics like precision/recall)

  # Stack all batches together (concatenate all mini-batch tensors into one big tensor)
  all_preds = torch.cat(all_preds).numpy()    # convert to numpy for metrics
  all_labels = torch.cat(all_labels).numpy()

  # Calculate accuracy (all labels must be correct)
  all_preds_tensor = torch.tensor(all_preds)
  all_labels_tensor = torch.tensor(all_labels)
  matches = (all_preds_tensor == all_labels_tensor).all(dim=1).float() # checks if all labels match
  accuracy = matches.mean().item()    # compute overall accuracy

  # Metrics
  f1 = f1_score(all_labels, all_preds, average='macro')
  precision = precision_score(all_labels, all_preds, average='macro')
  recall = recall_score(all_labels, all_preds, average='macro')

  print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

  return all_preds, all_labels, {'scores': {"accuracy": accuracy,
                                            "f1": f1,
                                            "precision": precision,
                                            "recall": recall}
                                }
