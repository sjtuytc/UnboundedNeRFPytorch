from typing import List

import torch
from torch import nn


class MegaNeRFContainer(nn.Module):
    def __init__(self, sub_modules: List[nn.Module], bg_sub_modules: List[nn.Module], centroids: torch.Tensor,
                 grid_dim: torch.Tensor, min_position: torch.Tensor, max_position: torch.Tensor, need_viewdir: bool,
                 need_appearance_embedding: bool, cluster_2d: bool):
        super(MegaNeRFContainer, self).__init__()

        for i, sub_module in enumerate(sub_modules):
            setattr(self, 'sub_module_{}'.format(i), sub_module)

        for i, bg_sub_module in enumerate(bg_sub_modules):
            setattr(self, 'bg_sub_module_{}'.format(i), bg_sub_module)

        self.centroids = centroids
        self.grid_dim = grid_dim
        self.min_position = min_position
        self.max_position = max_position
        self.need_viewdir = need_viewdir
        self.need_appearance_embedding = need_appearance_embedding
        self.cluster_2d = cluster_2d
