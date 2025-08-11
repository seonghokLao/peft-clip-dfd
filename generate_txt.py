import os
from pathlib import Path


ROOT = Path("datasets/FF")  # Matches OUT_ROOT from symlink script
CURR_ROOT = "config/datasets/FF"
SPLITS = ["train", "val", "test"]
METHODS = ["real", "DF", "F2F", "FS", "NT"]  # Short labels used by dataset class


def get_paths(source):
    return sorted([str(p) for p in ROOT.glob(f"{source}/*/*.png")])


for split in SPLITS:
    for method in METHODS:
        paths = get_paths(method)
        out_dir = Path(CURR_ROOT) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{method}.txt"
        with open(out_file, "w") as f:
            for path in paths:
                f.write(path + "\n")
        print(f"Wrote {len(paths)} paths to {out_file}")
