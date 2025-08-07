import os

import torch
from lightning.fabric import Fabric
from PIL import Image

from src.config import Config
from src.model.dfdet import DeepfakeDetectionModel

DEVICES = [0]

torch.set_float32_matmul_precision("high")

# Check if weights/model.ckpt exists, if not, download it from huggingface
model_path = "weights/model.ckpt"
if not os.path.exists(model_path):
    print("Downloading model")
    os.makedirs("weights", exist_ok=True)
    os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O {model_path}")

# Load checkpoint
ckpt = torch.load(model_path, map_location="cpu")

run_name = ckpt["hyper_parameters"]["run_name"]
print(run_name)

# Initialize model from config
model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
model.eval()

# Load model state dict
model.load_state_dict(ckpt["state_dict"])

# Get preprocessing function
preprocessing = model.get_preprocessing()

# Load some images
paths = [
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
    "datasets/CDFv2/YouTube-real/00000/000.png",
    "datasets/CDFv2/YouTube-real/00000/014.png",
    "datasets/CDFv2/YouTube-real/00000/028.png",
    "datasets/CDFv2/YouTube-real/00000/043.png",
    "datasets/CDFv2/Celeb-real/id0_0000/045.png",
    "datasets/CDFv2/Celeb-real/id0_0000/030.png",
    "datasets/CDFv2/Celeb-real/id0_0000/015.png",
    "datasets/CDFv2/Celeb-real/id0_0000/000.png",
]

# To pillow images
pillow_images = [Image.open(image) for image in paths]

# To tensors
batch_images = torch.stack([preprocessing(image) for image in pillow_images])

precision = ckpt["hyper_parameters"]["precision"]
fabric = Fabric(accelerator="cuda", devices=DEVICES, precision=precision)
fabric.launch()
model = fabric.setup_module(model)

# perform inference
with torch.no_grad():
    # Move batch_images to the correct device and dtype
    batch_images = batch_images.to(fabric.device).to(model.dtype)

    # Forward pass
    output = model(batch_images)

# logits to probabilities
softmax_output = output.logits_labels.softmax(dim=1).cpu().numpy()

for path, (p_real, p_fake) in zip(paths, softmax_output):
    print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")
