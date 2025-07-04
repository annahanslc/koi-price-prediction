import torch
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ir_config import IRConfig


class ImageRegressionTorch:
    def __init__(
        self,
        filepath,
        img_dir,
        image_col,
        target_col,
        testval_size,
        val_size,
        model=torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        ),
        num_epochs=100,
        learning_rate=0.01,
        weight_decay=0.0001,
        scheduler_patience=5,
        scheduler_factor=0.1,
        config: IRConfig = IRConfig(),
    ):
        self.config = config
        self.filepath = filepath
        self.img_dir = img_dir
        self.image_col = image_col
        self.target_col = target_col
        self.testval_size = testval_size
        self.val_size = val_size
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        self.epoch = 0
        self.num_epochs = num_epochs
        self.device = "mps" if torch.mps.is_available else "cpu"
        self.df = config.load_json_to_df(filepath)

        self.transforms_train = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(10),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transforms_val = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.train_ds, self.val_ds, self.test_ds = self.get_split_datasets()
        self.train_loader, self.test_loader, self.val_loader = self.get_dataloaders()

        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last layer
        self.model.fc = torch.nn.Linear(model.fc.in_features, 1)
        # Define the loss funtion
        self.criterion = torch.nn.MSELoss()
        # Define the optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Move the model to device
        model.to(self.device)
        # Define the scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            verbose=True,
        )

    def get_split_datasets(self):
        df_train, df_testval = train_test_split(
            self.df, test_size=self.testval_size, shuffle=True, random_state=42
        )
        df_test, df_val = train_test_split(
            df_testval, test_size=self.val_size, shuffle=True, random_state=42
        )

        train_ds = self.CustomDataset(
            df=df_train,
            img_dir=self.img_dir,
            image_col=self.image_col,
            transforms_val=self.transforms_val,
            transforms_train=self.transforms_train,
            target_col=self.target_col,
            validation=False,
        )
        val_ds = self.CustomDataset(
            df=df_val,
            img_dir=self.img_dir,
            image_col=self.image_col,
            transforms_val=self.transforms_val,
            transforms_train=self.transforms_train,
            target_col=self.target_col,
            validation=True,
        )
        test_ds = self.CustomDataset(
            df=df_test,
            img_dir=self.img_dir,
            image_col=self.image_col,
            transforms_val=self.transforms_val,
            transforms_train=self.transforms_train,
            target_col=self.target_col,
            validation=True,
        )

        return train_ds, val_ds, test_ds

    def get_dataloaders(self):
        # Create data loaders
        train_loader = DataLoader(self.train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(self.test_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(self.val_ds, batch_size=32, shuffle=False)

        return train_loader, test_loader, val_loader

    def tensor_checker(self):
        for images, targets in self.train_loader:
            print(type(images), images.shape)
            print(type(targets), targets.shape)
            break

        for images, targets in self.test_loader:
            print(type(images), images.shape)
            print(type(targets), targets.shape)
            break

        for images, targets in self.val_loader:
            print(type(images), images.shape)
            print(type(targets), targets.shape)
            break

    def train(self, num_epochs):
        # Save class variables
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        # Save training metrics to list
        (
            train_mse_list,
            train_rmse_list,
            train_mae_list,
            val_mse_list,
            val_rmse_list,
            val_mae_list,
        ) = ([], [], [], [], [], [])

        start_time = datetime.now()

        for epoch in range(num_epochs):
            # Set the model to training model
            model.train()
            total_train_loss = 0
            total_train_mae = 0

            for images, targets in tqdm(self.train_loader):
                # Move to tensor
                images = images.to(self.device)
                targets = targets.to(self.device).unsqueeze(1).float()
                # Reset the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(images)
                # Compute the loss
                loss = criterion(outputs, targets)
                # Backwards pass
                loss.backward()
                # Update model weights
                optimizer.step()
                # Add batch loss to total loss
                total_train_loss += loss.item()
                # Add batch mae to total mae, first turn tensor to scalar
                mae = torch.nn.L1Loss(reduction="mean")(outputs, targets).item()
                total_train_mae += mae

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_train_rmse = (avg_train_loss) ** 0.5
            avg_train_mae = total_train_mae / len(self.train_loader)

            train_mse_list.append(avg_train_loss)
            train_rmse_list.append(avg_train_rmse)
            train_mae_list.append(avg_train_mae)

            model.eval()
            with torch.inference_mode():
                total_val_loss = 0
                total_val_mae = 0

                for images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1).float()

                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
                    mae = torch.nn.L1Loss(reduction="mean")(outputs, targets).item()
                    total_val_mae += mae

                avg_val_loss = total_val_loss / len(self.val_loader)
                avg_val_rmse = (avg_val_loss) ** 0.5
                avg_val_mae = total_val_mae / len(self.val_loader)

                val_mse_list.append(avg_val_loss)
                val_rmse_list.append(avg_val_rmse)
                val_mae_list.append(avg_val_mae)

            current_lr = optimizer.param_groups[0]["lr"]

            # Print loss for this epoch
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss(MSE): {avg_train_loss:.4f}, Train RMSE: {avg_train_rmse:.4f}, Train MAE: {avg_train_mae:.4f} | "
                f"Val Loss(MSE): {avg_val_loss:.4f}, Val RMSE: {avg_val_rmse:.4f}, Val MAE: {avg_val_mae:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        end_time = datetime.now()
        print(
            f"Start Training Time: {start_time}, End Training Time: {end_time}, Total Time: {end_time - start_time}"
        )

        # plot the loss and accuracy over epochs
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 6))

        # plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_rmse_list, label="Train RMSE", marker="o")
        plt.plot(epochs, val_rmse_list, label="Val RMSE", marker="o")
        plt.title("RMSE Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()

        # plot the rmse
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mae_list, label="Train MAE", marker="o")
        plt.plot(epochs, val_mae_list, label="Val MAE", marker="o")
        plt.title("MAE Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()

        plt.tight_layout()
        plt.show()

    class CustomDataset(Dataset):
        def __init__(
            self,
            df,
            img_dir,
            image_col,
            target_col,
            transforms_val,
            transforms_train,
            validation=True,
        ):
            self.df = df
            self.validation = validation
            self.img_dir = img_dir
            self.image_col = image_col
            self.target_col = target_col
            self.transforms_val = transforms_val
            self.transforms_train = transforms_train

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # load and transform images
            img_path = self.df.iloc[idx][self.image_col]
            full_img_path = self.img_dir + img_path
            image = Image.open(full_img_path).convert("RGB")
            if self.validation:
                image = self.transforms_val(image)
            else:
                image = self.transforms_train(image)

            # convert label to tensor
            label = torch.tensor(
                self.df.iloc[idx][self.target_col], dtype=torch.float32
            )

            return image, label
