from typing import Callable

import numpy as np
from torchvision.datasets import CIFAR10

from ..config import Config
from ..dataset import BaseDataModule


class CIFAR10Dataset(CIFAR10):
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
        return {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__(config, preprocess)

    def setup(self, stage: str):
        # Initialize datasets
        if stage == "fit" or stage == "validate":
            self.train_dataset = CIFAR10Dataset(train=True, preprocess=self.preprocess)
            self.val_dataset = CIFAR10Dataset(train=False, preprocess=self.preprocess)

            print("\nTrain dataset")
            self.train_dataset.print_statistics()

            print("\nValidation dataset")
            self.val_dataset.print_statistics()

        if stage == "test":
            self.test_dataset = CIFAR10Dataset(train=False, preprocess=self.preprocess)

            print("\nTest dataset")
            self.test_dataset.print_statistics()
