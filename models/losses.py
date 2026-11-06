import torch
import torch.nn as nn


class BSIVLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert preds.shape == targets.shape
        assert preds.ndim == 2
        return self.loss_fn(preds, targets)
