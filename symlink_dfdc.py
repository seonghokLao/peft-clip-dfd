import os
import json
from pathlib import Path


# Paths
METADATA_PATH = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/DFDC/test/metadata.json")
FRAMES_ROOT = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/DFDC/test/frames")
OUT_ROOT = Path("datasets/DFDC")  # Target: real/ and fake/


# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


def symlink_frames(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(src_dir.glob("*.png")):
        dst = dst_dir / frame.name
        if not dst.exists():
            os.symlink(frame.resolve(), dst)


# Process all videos
for video_name, info in metadata.items():
    video_id = video_name.replace(".mp4", "")
    label = "fake" if info["is_fake"] else "real"  # "real" or "fake"


    src_dir = FRAMES_ROOT / video_id
    dst_dir = OUT_ROOT / label / video_id


    if not src_dir.exists():
        print(f"[Warning] Frame dir missing: {src_dir}")
        continue


    symlink_frames(src_dir, dst_dir)
    print(f"✅ Linked {video_id} → {label}")


print("\nDone.")