import os
from pathlib import Path


# Original CDFv2 dataset root
SRC_ROOT = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/Celeb-DF-v2")


# Where to symlink the processed files
DST_ROOT = Path("datasets/CDFv2")


# Mapping source folders to class labels
SOURCES = [
    "Celeb-synthesis",
    "Celeb-real",
    "YouTube-real"
]


def symlink_video_frames(src_video_dir, dst_video_dir):
    dst_video_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(src_video_dir.glob("*.png")):
        dst = dst_video_dir / frame.name
        if not dst.exists():
            os.symlink(frame.resolve(), dst)


def process_all():
    for subdir_name in SOURCES:
        src_frames_root = SRC_ROOT / subdir_name / "frames"
        for video_dir in sorted(src_frames_root.iterdir()):
            if video_dir.is_dir():
                dst_video_dir = DST_ROOT / subdir_name / video_dir.name
                symlink_video_frames(video_dir, dst_video_dir)
                print(f"Linked {video_dir} → {dst_video_dir}")


process_all()
