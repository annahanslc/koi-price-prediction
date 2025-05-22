import torch

################## CALCULATE CLASS WEIGHTS ###################

def get_class_weights(train_df, device):
  # Use BCEWithLogitsLoss's pos_weights to balance class weights

  # step 1: calculate the weights
  label_counts = train_df['mhe'].sum(axis=0)
  print(f'label_counts: {label_counts}')

  total_samples = train_df['mhe'].shape[0]
  print(f'total_samples: {total_samples}')

  pos_weights = total_samples / label_counts
  print(f'pos_weights: {pos_weights}')

  pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
  return pos_weights.to(device)


######################### TRAIN MODEL #########################

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs: int=20, name_best_model: str='best_model.pth') -> tuple[list, list, list, list]:

  # create the training loop

  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = []
  best_val_loss = float('inf')

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

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(model.state_dict(), name_best_model)

    # print loss for this epochs
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")

  return train_losses, train_accuracies, val_losses, val_accuracies


############# TRAIN MODEL WITH EARLY STOPPING & REDUCELRONPLATEAU ###############

def train_model_earlystop_reducelronplateau(model,
                                            train_loader,
                                            val_loader,
                                            criterion,
                                            optimizer,
                                            device,
                                            scheduler,
                                            num_epochs: int=20,
                                            patience_early_stop=5,
                                            best_model_name: str='best_model.pth') -> tuple[list, list, list, list]:

  # define early stopping
  best_val_loss = float('inf')
  counter = 0

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

    current_lr = optimizer.param_groups[0]['lr']

    # print loss for this epochs
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%} | "
          f"LR: {current_lr:.6f}")

    # update the learning rate scheduler
    scheduler.step(avg_val_loss)

    # check for early stopping
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(model.state_dict(), best_model_name)
      counter = 0

    else:
      counter += 1
      if counter >= patience_early_stop:
        print("Early stopping")
        break


  return train_losses, train_accuracies, val_losses, val_accuracies
