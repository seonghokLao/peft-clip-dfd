import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.utils import logger
from src.utils.logger import print

from .base import BaseDataModule, BaseDataset, init_augmentations

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DeepfakeDataset(BaseDataset):
    """
    DeepfakeDataset is any dataset that follows this structure:
    <dataset_name> / <source_name> / <video_name> / <frame_name>

    <dataset_name> - Name of the dataset, e.g. FF, CDF, DFD, DFDC...
    <source_name> - Name of the source, e.g. real, fake, or any name of generator, e.g. FaceSwap, Face2Face...
    <video_name> - Name of the video, e.g. 000, 000_003, ...
    <frame_name> - Any name of the frame, e.g. 000001.jpg, 000002.jpg, ...

    Notice:
    if <video_name> has a structure <A>_<B>, it will be parsed as <video_id>_<identity_id>,
    otherwise <video_id> = <identity_id>

    """

    def __init__(
        self,
        files_with_paths: list[str] | dict[str, list[str]],
        preprocess: None | Callable = None,
        augmentations: None | Callable = None,
        shuffle: bool = False,  # Shuffles the dataset once
        binary: bool = False,
        limit_files: None | int = None,
    ):
        files = []
        labels = []
        logger.print_info("Loading files")

        if binary:
            label2name = {0: "real", 1: "fake"}
        else:
            raise NotImplementedError("Only binary labels are supported")

        source2label = {v: k for k, v in label2name.items()}

        self.label2name = label2name

        dataset2files = None

        if isinstance(files_with_paths, dict):
            dataset2files_with_paths = files_with_paths.copy()
            dataset2files = {dataset_name: [] for dataset_name in dataset2files_with_paths.keys()}
            files_with_paths = [item for sublist in files_with_paths.values() for item in sublist]

        for file_with_paths in set(files_with_paths):
            with open(file_with_paths, "r") as f:
                paths = f.readlines()
                paths = [path.strip() for path in paths]

                # If files do not exist, append root of 'file' to each path
                root = file_with_paths.rsplit("/", 1)[0]

                def process_path(root, path):
                    if not os.path.exists(path):
                        path = f"{root}/{path}"
                    assert os.path.exists(path), f"File not found: {path}"
                    return path

                with ThreadPoolExecutor() as executor:
                    process_with_root = partial(process_path, root)
                    paths = list(
                        tqdm(
                            executor.map(process_with_root, paths),
                            total=len(paths),
                            desc=f"Processing paths in {file_with_paths}",
                        )
                    )

                files.extend(paths)

                if dataset2files is not None:
                    for dataset_name, files_with_paths in dataset2files_with_paths.items():
                        if file_with_paths in files_with_paths:
                            dataset2files[dataset_name].extend(paths)

        # Remove duplicate paths
        files = np.unique(files).tolist()

        # Limit the number of files
        if limit_files is not None:
            files = self.limit_files(files, limit_files)

        # Get labels from paths
        for path in files:
            source = self.get_source_from_file(path)

            if binary:
                if "real" in source:
                    source = "real"
                else:
                    source = "fake"

            label = source2label[source]
            labels.append(label)

        logger.print_info("Files loaded")

        super().__init__(files, labels, preprocess, augmentations, shuffle, dataset2files)

        self.source2uid = self._source2uid()

        self.file2index = {f: i for i, f in enumerate(self.files)}

    def limit_files(self, files: list[str], limit: int) -> list[str]:
        """Limits number of files by considering unique videos"""
        # Select unique videos
        video_paths = [self.get_video_path(file) for file in files]
        unique_videos = list(np.unique(video_paths))

        # For each video, select files
        video2files = {video: [] for video in unique_videos}
        for file, video in zip(files, video_paths):
            video2files[video].append(file)

        # Shuffle videos with fixed seed
        np.random.RandomState(42).shuffle(unique_videos)

        # Select files from shuffled videos
        selected_files = []
        for video in unique_videos:
            selected_files.extend(video2files[video])

            if len(selected_files) >= limit:
                break

        return selected_files[:limit]

    def _source2uid(self) -> dict[str, int]:
        sources = [self.get_source_from_file(file) for file in self.files]
        sources = np.unique(sources)

        assert any("real" in g for g in sources), "No real source found"
        sources = [str(g) for g in sources]

        # Map all real sources to 0 and fake sources to 1, 2, 3, ...
        real_sources = [g for g in sources if "real" in g]
        fake_sources = [g for g in sources if "real" not in g]

        source2uid = {s: 0 for s in real_sources}
        for i, s in enumerate(fake_sources, start=1):
            source2uid[s] = i

        return source2uid

    def get_frame_from_file(self, file_path):
        # .../<source_name>/<video_name>/<frame_name>
        return file_path.split("/")[-1]

    def get_video_from_file(self, file_path):
        video = file_path.split("/")[-2]  # Extract video name (000, 000_003, ...)
        # <video_id>, <video_id>_<identity_id>, ...
        return video.split("_")[0] if "_" in video else video

    def get_identity_from_file(self, file_path):
        video = file_path.split("/")[-2]  # Extract video name (000, 000_003, ...)
        # <identity_id>, <video_id>_<identity_id>, ...
        return video.split("_")[1] if "_" in video else video

    def get_source_from_file(self, file_path):
        # .../<source_name>/<video_name>/<frame_name>
        return file_path.split("/")[-3]

    def get_dataset_from_file(self, file_path):
        # .../<dataset_name>/<source_name>/<video_name>/<frame_name>
        return file_path.split("/")[-4]

    def get_video_path(self, file_path):
        # file_path[::-1].find("/") finds the last occurrence of "/"
        return file_path[: -file_path[::-1].find("/")]

    def get_class_names(self) -> dict[int, str]:
        return self.label2name

    def print_statistics(self):
        super().print_statistics()

        video_paths = [self.get_video_path(file) for file in self.files]

        files_by_dataset = [self.get_dataset_from_file(file) for file in self.files]

        print(f"Total number of frames: {len(self.files)}")
        print(f"Total number of videos: {len(set(video_paths))}")

        # For each dataset, print number of frames and videos
        df = pd.DataFrame({"dataset": files_by_dataset, "video": video_paths})

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            videos_count = dataset_df["video"].nunique()
            frames_count = len(dataset_df)
            print(f"Dataset: {dataset}, videos: {videos_count}, frames: {frames_count}")

    def __getitem__(self, idx):
        # self.
        path = self.files[idx]
        image = Image.open(path)
        if self.augmentations is not None:
            image = self.augmentations(image)
        if self.preprocess is not None:
            image = self.preprocess(image)
        return {
            "idx": idx,
            "image": image,
            "label": self.labels[idx],
            "path": path,
            "video": self.get_video_from_file(path),
            "identity": self.get_identity_from_file(path),
            "source": self.source2uid[self.get_source_from_file(path)],
            "frame": self.get_frame_from_file(path),
        }


class DeepfakeDataModule(BaseDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__(config, preprocess)

    def setup(self, stage: str):
        # Initialize datasets
        if stage == "fit" or stage == "validate":
            augmentations = init_augmentations()
            logger.print("\n[blue]Creating training dataset")
            self.train_dataset = DeepfakeDataset(
                self.config.trn_files,
                self.preprocess,
                augmentations=augmentations,
                binary=self.config.binary_labels,
                limit_files=self.config.limit_trn_files,
            )
            self.train_dataset.print_statistics()

            logger.print("\n[blue]Creating validation dataset")
            self.val_dataset = DeepfakeDataset(
                self.config.val_files,
                self.preprocess,
                shuffle=True,
                binary=self.config.binary_labels,
                limit_files=self.config.limit_val_files,
            )
            self.val_dataset.print_statistics()

        if stage == "test":
            logger.print("\nCreating test dataset")
            self.test_dataset = DeepfakeDataset(
                self.config.tst_files,
                self.preprocess,
                binary=self.config.binary_labels,
                limit_files=self.config.limit_tst_files,
            )
            self.test_dataset.print_statistics()

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
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
