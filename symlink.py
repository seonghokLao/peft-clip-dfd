import os
import json
from pathlib import Path
import shutil


DATA_ROOT = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/FaceForensics++")
OUT_ROOT = Path("datasets/FF")


# Map each manipulation method to its "fake" label
MANIP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
REAL_SOURCE = "youtube"


SPLITS = {
    "train": "config/datasets/FF/train.json",
    "val": "config/datasets/FF/val.json",
    "test": "config/datasets/FF/test.json",
}


def symlink_video_frames(source_dir, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in sorted(source_dir.glob("*.png")):
        dst = dest_dir / frame_path.name
        if not dst.exists():
            os.symlink(frame_path.resolve(), dst)


def process_split(split_name, split_file):
    with open(split_file, "r") as f:
        pairs = json.load(f)


    for source_id, target_id in pairs:
        fake_id = f"{source_id}_{target_id}"


        # Process fakes for each method
        for method in MANIP_METHODS:
            src_path = DATA_ROOT / f"manipulated_sequences/{method}/c23/frames/{fake_id}"
            dst_path = OUT_ROOT / method / fake_id
            if src_path.exists():
                symlink_video_frames(src_path, dst_path)


        # Process reals
        for real_id in [source_id, target_id]:
            src_path = DATA_ROOT / f"original_sequences/{REAL_SOURCE}/c23/frames/{real_id}"
            dst_path = OUT_ROOT / "real" / real_id
            if src_path.exists():
                symlink_video_frames(src_path, dst_path)


# Run for all splits
for split_name, json_file in SPLITS.items():
    process_split(split_name, json_file)

