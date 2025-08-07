# extract_features.py


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.config import Config
from run import get_train_config
from src.dataset import DeepfakeDataModule
from src import model as models


@torch.no_grad()
def extract_features():
    config = get_train_config()
    model = models.DeepfakeDetectionModel(config)
    model.eval().cuda()


    datamodule = DeepfakeDataModule(config, model.get_preprocessing())
    datamodule.setup("fit")


    loader = DataLoader(datamodule.train_dataset, batch_size=64, num_workers=8)


    all_feats, all_labels = [], []


    for batch in tqdm(loader):
        images = batch["image"].cuda()
        labels = batch["label"]
        feats = model.feature_extractor(images)
        all_feats.append(feats.cpu())
        all_labels.append(labels)


    features = torch.cat(all_feats).numpy()
    labels = torch.cat(all_labels).numpy()


    np.savez("features_train.npz", features=features, labels=labels)


if __name__ == "__main__":
    extract_features()