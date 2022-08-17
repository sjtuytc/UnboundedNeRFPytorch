from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val

    def load_image(self) -> torch.Tensor:
        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> Optional[torch.Tensor]:
        if self._mask_path is None:
            return None

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
