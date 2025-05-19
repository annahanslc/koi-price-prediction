import matplotlib.pyplot as plt

def plot_train_val_losses(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies):

  # plot the loss and accuracy over epochs

  epochs = range(1, num_epochs +1)
  plt.figure(figsize=(12,6))

  # plot the loss
  plt.subplot(1,2,1)
  plt.plot(epochs, train_losses, label='Train Loss', marker='o')
  plt.plot(epochs, val_losses, label='Val Loss', marker ='o')
  plt.title('Loss Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # plot the accuracy
  plt.subplot(1,2,2)
  plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
  plt.plot(epochs, val_accuracies, label='Val Acc', marker='o')
  plt.title('Accuracy Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout()
  plt.show()
