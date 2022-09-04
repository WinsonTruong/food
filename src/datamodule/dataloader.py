# this code is not my own it was adapted from Ayush Thukar

import torch
import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.datasets.utils import download_url

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.num_classes = 101 # found from docs

    def prepare_data(self):
        """
        Don't need to implement yet until I'm in a stage like this 
        model.prepare_data()
        initialize_distributed()
        model.setup(stage)
        model.train_dataloader()
        """
        pass

    def setup(self, stage = None):
        dataset = Food101(root = self.data_dir, download = True, split = "train") 
        self.train, self.val = torch.utils.data.random_split(dataset, [60600, 15150]) #60600, 15150 is specific to food101
        self.test = Food101(root=self.data_dir, download = True, split = "test")
        self.test = torch.utils.data.random_split(self.test, [len(self.test)])[0]
        
        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=16
            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val, batch_size=self.batch_size, num_workers=16
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, num_workers=16
            )

    
        