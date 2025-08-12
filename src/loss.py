from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.unifalign import alignment, uniformity
from src.utils import logger

from .config import Loss as LossConfig

import numpy as np


@dataclass
class LossInputs:
    logits_labels: None | torch.Tensor = None
    labels: None | torch.Tensor = None
    embeddings: None | torch.Tensor = None


@dataclass
class LossOutputs:
    ce_labels: None | float = None
    bce_labels: None | float = None
    uniformity: None | float = None
    alignment_labels: None | float = None
    total: int | torch.Tensor = 0


class Loss(nn.Module):
    def __init__(self, loss_config: LossConfig):
        super().__init__()
        self.config = loss_config

        centers = np.load("cluster_labels.npz")
        self.cluster_centers = torch.from_numpy(centers["centers"]).float()

    def forward(
        self,
        inputs: LossInputs,
    ) -> LossOutputs:
        loss_outputs = LossOutputs()
        print("labels", inputs.labels.unique())
        # print("fake target", target)

        if inputs.logits_labels is not None:
            if self.config.ce_labels:
                L = self.config.ce_labels * F.cross_entropy(
                    inputs.logits_labels, inputs.labels, label_smoothing=self.config.label_smoothing
                )
                loss_outputs.ce_labels = L.item()
                loss_outputs.total += L

        if inputs.embeddings is not None:
            # L2 normalize embeddings
            # See 3.1  https://arxiv.org/pdf/2004.11362
            # embeddings = F.normalize(inputs.embeddings, p=2, dim=1)
            embeddings = inputs.embeddings

            embeddings = F.normalize(embeddings, p=2, dim=1)

            # check that embeddings are normalized
            if not torch.allclose(
                embeddings.norm(p=2, dim=1), torch.ones(embeddings.size(0), device=embeddings.device)
            ):
                logger.print_warning_once("[yellow]Embeddings are not normalized")

            if inputs.labels is not None:
                if self.config.alignment_labels:
                    L = self.config.alignment_labels * alignment(embeddings, inputs.labels)
                    loss_outputs.alignment_labels = L.item()
                    loss_outputs.total += L
                
                Lp = self.prototype_contrastive_loss(embeddings, inputs.labels)
                loss_outputs.total += Lp

            if self.config.uniformity:
                L = self.config.uniformity * uniformity(embeddings)
                loss_outputs.uniformity = L.item()
                loss_outputs.total += L

        if isinstance(loss_outputs.total, int):
            logger.print_warning_once("[yellow]Total loss is 0. Check if loss coefficients are set correctly.")

        if isinstance(loss_outputs.total, torch.Tensor) and loss_outputs.total.isnan():
            logger.print_warning("[yellow]Total loss is nan")
            loss_outputs.total = inputs.logits_labels.sum() * 0

        return loss_outputs

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        return super().__call__(inputs)

    def prototype_contrastive_loss(self, embeddings, labels, temperature=0.07):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centers = F.normalize(self.cluster_centers.to(embeddings.device), p=2, dim=1)  # [K, D]


        # Only keep fake samples (label > 0)
        fake_mask = labels > 0
        if not fake_mask.any():
            print("[yellow]No fake samples found, skipping prototype contrastive loss")
            return torch.tensor(0.0, device=embeddings.device)


        emb = embeddings[fake_mask]
        # Remap 1..K → 0..K-1
        target = labels[fake_mask] - 1


        assert target.min() >= 0 and target.max() < centers.shape[0], \
            f"Target out of range: min={target.min()}, max={target.max()}, num_prototypes={centers.shape[0]}"


        logits = torch.matmul(emb, centers.T) / temperature  # [N_fake, K]
        return F.cross_entropy(logits, target)
