import torch
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
        config: IRConfig = IRConfig(),
    ):
        self.config = config
        self.filepath = filepath
        self.img_dir = img_dir
        self.image_col = image_col
        self.target_col = target_col
        self.testval_size = testval_size
        self.val_size = val_size

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
        for images, labels in self.train_loader:
            print(type(images), images.shape)
            print(type(labels), labels.shape)
            break

        for images, labels in self.test_loader:
            print(type(images), images.shape)
            print(type(labels), labels.shape)
            break

        for images, labels in self.val_loader:
            print(type(images), images.shape)
            print(type(labels), labels.shape)
            break

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
