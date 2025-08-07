from typing import Callable

import numpy as np
from torchvision.datasets import MNIST

from ..config import Config
from ..dataset import BaseDataModule


class MNISTDataset(MNIST):
    def __init__(
        self,
        train: bool = True,
        preprocess: None | Callable = None,
        augmentations: None | Callable = None,
    ):
        super().__init__(root="datasets/other/", train=train, download=True)
        self.preprocess = preprocess
        self.augmentations = augmentations

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.augmentations is not None:
            image = self.augmentations(image)
        if self.preprocess is not None:
            image = self.preprocess(image)
        return {
            "image": image,
            "label": label,
            "path": f"{idx}: {label}",
            "idx": idx,
        }

    def print_statistics(self):
        print(f"Number of samples: {len(self)}")
        unique, counts = np.unique(self.targets, return_counts=True)
        print("Class distribution")
        names = self.get_class_names()
        for u, c in zip(unique, counts):
            print(f"Class {u} ({names[u]}): {c}")

    def get_class_names(self) -> dict[int, str]:
        return {i: str(i) for i in range(10)}


class MNISTDataModule(BaseDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__(config, preprocess)

    def setup(self, stage: str):
        # Initialize datasets
        if stage == "fit" or stage == "validate":
            self.train_dataset = MNISTDataset(train=True, preprocess=self.preprocess)
            self.val_dataset = MNISTDataset(train=False, preprocess=self.preprocess)

            print("\nTrain dataset")
            self.train_dataset.print_statistics()

            print("\nValidation dataset")
            self.val_dataset.print_statistics()

        if stage == "test":
            self.test_dataset = MNISTDataset(train=False, preprocess=self.preprocess)

            print("\nTest dataset")
            self.test_dataset.print_statistics()
