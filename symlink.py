import os
import json
from pathlib import Path


DATA_ROOT = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/FaceForensics++")
OUT_ROOT = Path("datasets/FF")


# Mapping of method name to label directory
METHOD_MAP = {
    "Deepfakes": "DF",
    "Face2Face": "F2F",
    "FaceSwap": "FS",
    "NeuralTextures": "NT"
}
REAL_SOURCE = "youtube"


SPLITS = {
    "train": "config/datasets/FF/train.json",
    "val": "config/datasets/FF/val.json",
    "test": "config/datasets/FF/test.json",
}


def symlink_frames(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(src_dir.glob("*.png")):
        link = dst_dir / frame.name
        if not link.exists():
            os.symlink(frame.resolve(), link)


def process_split(json_file):
    with open(json_file, "r") as f:
        pairs = json.load(f)


    for src_id, tgt_id in pairs:
        fake_id = f"{src_id}_{tgt_id}"


        for method_name, short_label in METHOD_MAP.items():
            src = DATA_ROOT / f"manipulated_sequences/{method_name}/c23/frames/{fake_id}"
            dst = OUT_ROOT / short_label / fake_id
            if src.exists():
                symlink_frames(src, dst)


        for real_id in [src_id, tgt_id]:
            src = DATA_ROOT / f"original_sequences/{REAL_SOURCE}/c23/frames/{real_id}"
            dst = OUT_ROOT / "real" / real_id
            if src.exists():
                symlink_frames(src, dst)


# Run all splits (only structure depends on IDs)
for json_file in SPLITS.values():
    process_split(json_file)

