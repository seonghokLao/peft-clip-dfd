from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadOutput:
    logits_labels: None | torch.Tensor = None
    logits_source: None | torch.Tensor = None
    features: torch.Tensor = None


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x):
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=1)
        logits = self.linear(x)

        # Let features always be normalized
        features = x if self.normalize_inputs else F.normalize(x, p=2, dim=1)
        return HeadOutput(logits_labels=logits, features=features)
