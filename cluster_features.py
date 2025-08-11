# cluster_features.py


import numpy as np
from sklearn.cluster import KMeans


data = np.load("features_train.npz", allow_pickle=True)
features = data["features"]
labels = data["labels"]
paths = data["paths"]


fake_idx = np.where(labels == 1)[0]
fake_features = features[fake_idx]


k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(fake_features)


fake_clusters = kmeans.labels_
cluster_labels = labels.copy()
cluster_labels[fake_idx] = fake_clusters + 1  # real=0, fake=1..k


centers = kmeans.cluster_centers_  # shape: (k, feature_dim)


np.savez(
    "cluster_labels.npz",
    paths=paths,
    cluster_labels=cluster_labels,
    centers=centers
)
