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


    # Build model
    model = models.DeepfakeDetectionModel(config)
    model.eval().cuda()


    # Build datamodule
    dm = DeepfakeDataModule(config, model.get_preprocessing())
    dm.setup("fit")


    # IMPORTANT: turn off shuffle here so order is deterministic
    loader = DataLoader(dm.train_dataset, batch_size=64, num_workers=8, shuffle=False)


    all_feats, all_labels, all_paths = [], [], []


    for batch in tqdm(loader):
        images = batch["image"].cuda()
        labels = batch["label"]
        paths  = batch["path"]          # list of strings


        feats = model.feature_extractor(images)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
        all_paths.extend(paths)         # keep as a flat list


    features = torch.cat(all_feats).numpy()
    labels = torch.cat(all_labels).numpy()
    paths = np.array(all_paths)


    # Save ALL THREE
    np.savez("features_train.npz", features=features, labels=labels, paths=paths)


if __name__ == "__main__":
    extract_features()
