# extract_features.py


from dfdet import DeepfakeDetectionModel
from torch.utils.data import DataLoader
import torch


model.eval()
features = []
labels = []


for batch in DataLoader(train_dataset, batch_size=64):
    with torch.no_grad():
        feats = model.feature_extractor(batch["image"].to(device))
        features.append(feats.cpu())
        labels.append(batch["label"].cpu())


features = torch.cat(features)
labels = torch.cat(labels)
torch.save({"features": features, "labels": labels}, "train_feats.pt")