import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Models:
        1. openai/clip-vit-base-patch16 | 768 features
        2. openai/clip-vit-base-patch32 | 768 features
        3. openai/clip-vit-large-patch14 | 1024 features

        See more in src/config.py
        """

        super().__init__()

        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        clip: CLIPModel = CLIPModel.from_pretrained(model_name)

        # take vision model from CLIP, maps image to vision_embed_dim
        self.vision_model = clip.vision_model

        self.model_name = model_name

        self.features_dim = self.vision_model.config.hidden_size

        # take visual_projection, maps vision_embed_dim to projection_dim
        # self.visual_projection = clip.visual_projection

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self):
        return self.features_dim


class CLIPEncoderPatches(CLIPEncoder):
    def __init__(self, model_name):
        """
        See CLIPEncoder
        """
        super().__init__(model_name)

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        embeddings = self.vision_model(preprocessed_images).last_hidden_state

        # for clip-large-patch14, we have [B, 257, 1024]
        # we want to reshape to take N by N patches, so that we have [B, N, N, 1024]
        embeddings = embeddings[:, 1:]
        B, T, _ = embeddings.shape
        N = int(np.sqrt(T))
        embeddings = embeddings.reshape(B, N, N, -1)

        # To [B, C, H, W]
        embeddings = embeddings.permute(0, 3, 1, 2)

        return embeddings

    def get_features_dim(self):
        config = self.vision_model.config
        hidden_size = config.hidden_size
        num_patches = config.image_size // config.patch_size
        return hidden_size, num_patches


if __name__ == "__main__":
    model = CLIPEncoder("openai/clip-vit-base-patch16")

    path1 = "datasets/FF/real/000/000.png"
    path2 = "datasets/FF/real/000/000.png"

    image1 = Image.open(path1)
    image2 = Image.open(path2)

    preprocessed = [model.preprocess(image) for image in [image1, image2]]
    preprocessed = torch.stack(preprocessed)
    outputs = model(preprocessed)

    print(outputs.shape)
