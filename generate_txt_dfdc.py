from pathlib import Path


ROOT = Path("datasets/DFDC")  # where symlinks live
OUT_DIR = Path("config/datasets/DFDC/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


for label in ["real", "fake"]:
    frame_paths = sorted(ROOT.glob(f"{label}/*/*.png"))
    out_file = OUT_DIR / f"{label}.txt"
    with open(out_file, "w") as f:
        for path in frame_paths:
            f.write(str(path) + "\n")
    print(f"✅ Wrote {len(frame_paths)} paths to {out_file}")
