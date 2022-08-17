from typing import Optional

import torch
from torch import nn


class Cascade(nn.Module):
    def __init__(self, coarse: nn.Module, fine: nn.Module):
        super(Cascade, self).__init__()
        self.coarse = coarse
        self.fine = fine

    def forward(self, use_coarse: bool, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if use_coarse:
            return self.coarse(x, sigma_only, sigma_noise)
        else:
            return self.fine(x, sigma_only, sigma_noise)
