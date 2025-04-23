import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, L):
        """
        Input:
            L: Tensor of shape (B, 1, H, W) — the grayscale channel
        Output:
            ab_pred: Tensor of shape (B, 2, H, W) — predicted ab color channels
        """
        raise NotImplementedError("Each model must implement forward()")
