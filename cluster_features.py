import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
# from umap import UMAP   # alternative to TSNE, pip install umap-learn
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


# ====== Load features ======
data = np.load("features_train.npz")
features = data["features"]  # shape (N, D)
labels = data["labels"]      # 0 = real, 1 = fake


print(f"Loaded features: {features.shape}, labels: {np.unique(labels, return_counts=True)}")


# ====== Only cluster fake samples ======
fake_idx = np.where(labels == 1)[0]
fake_features = features[fake_idx]


# ====== Clustering with progress ======
n_clusters = 5
batch_size = 4096
max_epochs = 10


kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                          batch_size=batch_size,
                          init="k-means++",
                          random_state=42)


for epoch in trange(max_epochs, desc="Epochs"):
    pbar = tqdm(range(0, len(fake_idx), batch_size),
                desc=f"Clustering epoch {epoch+1}/{max_epochs}")
    for start in pbar:
        end = min(start + batch_size, len(fake_idx))
        mb = fake_features[start:end]
        kmeans.partial_fit(mb)
        pbar.set_postfix({"inertia": f"{kmeans.inertia_:.3e}"})


# ====== Assign cluster labels ======
fake_clusters = kmeans.predict(fake_features)


# Map back to full dataset labels: real stays 0, fake clusters are 1..K
cluster_labels = labels.copy()
cluster_labels[fake_idx] = fake_clusters + 1


np.save("cluster_labels.npy", cluster_labels)
print("Saved cluster_labels.npy")


# ====== Visualization ======
# Option 1: t-SNE
print("Running t-SNE projection (this can take a while)...")
proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(fake_features)


plt.figure(figsize=(8, 8))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=fake_clusters, cmap="tab10", s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(n_clusters), label="Cluster ID")
plt.title("Fake samples clustered (t-SNE projection)")
plt.savefig("clusters_tsne.png", dpi=300)
plt.show()


# ====== Optional: Visualize real+fake together ======
# If you want to see real samples too, you can sample a subset:
sample_real_idx = np.random.choice(np.where(labels==0)[0], size=2000, replace=False)
all_subset_idx = np.concatenate([sample_real_idx, fake_idx])
proj_all = TSNE(n_components=2, random_state=42).fit_transform(features[all_subset_idx])
color_all = cluster_labels[all_subset_idx]
plt.scatter(proj_all[:, 0], proj_all[:, 1], c=color_all, cmap="tab10", s=5)
plt.savefig("clusters_tsne_with_real.png", dpi=300)
plt.show()