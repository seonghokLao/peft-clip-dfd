import os
from pathlib import Path


ROOT = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/FaceForensics++")
CURR_ROOT = "config/datasets/FF"
SPLITS = ["train", "val", "test"]
METHODS = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def get_paths(source):
    return sorted([str(p) for p in ROOT.glob(f"{source}/*/*.png")])


for split in SPLITS:
    for method in METHODS:
        paths = get_paths(method)
        out_file = f"{CURR_ROOT}/{split}/{method}.txt"
        with open(out_file, "w") as f:
            for path in paths:
                f.write(path + "\n")
        print(f"Wrote {len(paths)} paths to {out_file}")

