import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix

def evaluate_model_flare(model, val_loader, device, class_labels):
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

  # check the classification report

  print(classification_report(all_labels, all_preds, target_names=class_labels))

  # create the confusion matrix

  conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)

  # display the confusion matrix for each label

  num_rows = len(class_labels)//2 + 1

  fig, ax = plt.subplots(num_rows, 2, figsize=(10, num_rows*4))
  ax = ax.flatten()

  for i, cm in enumerate(conf_matrices):
      sns.heatmap(cm, annot=True, fmt='d', cmap='flare', cbar=False,
                  xticklabels=['Pred 0', 'Pred 1'],
                  yticklabels=['True 0', 'True 1'],
                  ax=ax[i])
      ax[i].set_title(f'Confusion Matrix for {class_labels[i]}')
      ax[i].set_ylabel('True Label')
      ax[i].set_xlabel('Predicted Label')

  # hide unused subplot
  for j in range(i+1, len(ax)):
      fig.delaxes(ax[j])

  plt.tight_layout()
  plt.show()

  return all_preds, all_labels, {'scores': {"accuracy": accuracy,
                                            "f1": f1,
                                            "precision": precision,
                                            "recall": recall}
                                }
