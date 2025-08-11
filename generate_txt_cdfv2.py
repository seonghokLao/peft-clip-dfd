from pathlib import Path
from collections import defaultdict


# Input
video_list_file = Path("/home/laoseonghok/github/DeepfakeBench/datasets/rgb/Celeb-DF-v2/List_of_testing_videos.txt")  # Change to actual file path
frame_root = Path("datasets/CDFv2")        # Symlinked data root
out_dir = Path("config/datasets/CDFv2/test")
out_dir.mkdir(parents=True, exist_ok=True)


# Maps each source to its label used in symlink directory
source_to_label = {
    "Celeb-synthesis": "fake",
    "Celeb-real": "real",
    "YouTube-real": "real"
}


# Store frames grouped by source
frames_by_source = defaultdict(list)


# Read video list
with open(video_list_file, "r") as f:
    for line in f:
        try:
            _, video_path = line.strip().split()
        except ValueError:
            continue  # skip malformed lines


        source, video_name = video_path.split("/")
        video_id = video_name.replace(".mp4", "")
        label_dir = source_to_label[source]


        frame_dir = frame_root / source / video_id
        if not frame_dir.exists():
            print(f"[Warning] Missing frame dir: {frame_dir}")
            continue


        frame_paths = sorted(frame_dir.glob("*.png"))
        frames_by_source[source].extend([str(p) for p in frame_paths])


# Write output .txt for each source
for source, frame_list in frames_by_source.items():
    out_file = out_dir / f"{source}.txt"
    with open(out_file, "w") as f:
        for path in frame_list:
            f.write(path + "\n")
    print(f"✅ Wrote {len(frame_list)} frames to {out_file}")
