from typing import Callable, Literal, Optional

import lightning as pl
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from src.config import Config
from src.utils.logger import print, print_info


class BaseDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        labels: list[int],
        preprocess: None | Callable = None,
        augmentations: None | Callable = None,
        shuffle: bool = False,  # Shuffles the dataset once
        dataset2files: Optional[dict[str, list[str]]] = None,
    ):
        self.files = files
        self.labels = labels

        self.preprocess = preprocess
        self.augmentations = augmentations

        self.dataset2files = dataset2files

        if shuffle:
            self.shuffle()

    def shuffle(self):
        # create fixed seed for reproducibility
        idx = np.random.RandomState(42).permutation(len(self.files))
        self.files = [self.files[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        if self.augmentations is not None:
            image = self.augmentations(image)
        if self.preprocess is not None:
            image = self.preprocess(image)
        return {
            "image": image,
            "label": self.labels[idx],
            "path": path,
        }

    def print_statistics(self):
        print(f"Number of samples: {len(self.files)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution")
        names = self.get_class_names()
        for u, c in zip(unique, counts):
            print(f"Class {u} ({names[u]}): {c}")

    def get_class_names(self) -> dict[int, str]:
        raise NotImplementedError


def init_augmentations():
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.JPEG([40, 100]),
        ]
    )


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__()
        self.config = config
        self.preprocess = preprocess

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
