import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs: int=20) -> tuple[list, list, list, list]:

  # create the training loop

  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = []

  for epoch in range(num_epochs):
    # set the model to training mode
    model.train()
    # initialize loss tracking
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    # loop through training data in batches
    for images, labels in train_loader:
      images = images.to(device) # move data to MPS
      labels = labels.to(device) # move data to MPS

      # reset the gradients
      optimizer.zero_grad()
      # forward pass
      outputs = model(images)
      # compute the loss
      loss = criterion(outputs, labels)
      # backward pass
      loss.backward()
      # update model weights
      optimizer.step()

      # add batch loss to total
      total_train_loss += loss.item()

      # apply sigmoid and threshold to get multi-label predictions
      probs = torch.sigmoid(outputs)
      predicted = (probs > 0.5).float()

      # count the correct predictions per label
      train_correct += (predicted == labels).float().sum().item()
      train_total += labels.numel()

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct / train_total

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # VALIDATION
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()

        # apply sigmoid and threshold to get multi-label predictions
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()

        # count the correct predictions per label
        val_correct += (predicted == labels).float().sum().item()
        val_total += labels.numel()

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = val_correct / val_total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    # print loss for this epochs
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")

  return train_losses, train_accuracies, val_losses, val_accuracies
