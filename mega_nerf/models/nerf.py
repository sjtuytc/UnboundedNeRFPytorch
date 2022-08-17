from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int, pos_dir_dim: int, layers: int, skip_layers: List[int], layer_dim: int,
                 appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int,
                 sigma_activation: nn.Module):
        super(NeRF, self).__init__()
        self.xyz_dim = xyz_dim

        if rgb_dim > 3:
            assert pos_dir_dim == 0

        self.embedding_xyz = Embedding(pos_xyz_dim)
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2

        self.skip_layers = skip_layers

        xyz_encodings = []

        # xyz encoding layers
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            xyz_encodings.append(layer)

        self.xyz_encodings = nn.ModuleList(xyz_encodings)

        if pos_dir_dim > 0:
            self.embedding_dir = Embedding(pos_dir_dim)
            in_channels_dir = 3 + 3 * pos_dir_dim * 2
        else:
            self.embedding_dir = None
            in_channels_dir = 0

        if appearance_dim > 0:
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None

        if affine_appearance:
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None

        if pos_dir_dim > 0 or (appearance_dim > 0 and not affine_appearance):
            self.xyz_encoding_final = nn.Linear(layer_dim, layer_dim)
            # direction and appearance encoding layers
            self.dir_a_encoding = nn.Sequential(
                nn.Linear(layer_dim + in_channels_dir + (appearance_dim if not affine_appearance else 0),
                          layer_dim // 2),
                nn.ReLU(True))
        else:
            self.xyz_encoding_final = None

        # output layers
        self.sigma = nn.Linear(layer_dim, 1)
        self.sigma_activation = sigma_activation

        self.rgb = nn.Linear(
            layer_dim // 2 if (pos_dir_dim > 0 or (appearance_dim > 0 and not affine_appearance)) else layer_dim,
            rgb_dim)
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()  # = nn.Sequential(rgb, nn.Sigmoid())
        else:
            self.rgb_activation = None  # We're using spherical harmonics and will convert to sigmoid in rendering.py

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        expected = self.xyz_dim \
                   + (0 if (sigma_only or self.embedding_dir is None) else 3) \
                   + (0 if (sigma_only or self.embedding_a is None) else 1)

        if x.shape[1] != expected:
            raise Exception(
                'Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_noise is not None:
            sigma += sigma_noise

        sigma = self.sigma_activation(sigma)

        if sigma_only:
            return sigma

        if self.xyz_encoding_final is not None:
            xyz_encoding_final = self.xyz_encoding_final(xyz_)
            dir_a_encoding_input = [xyz_encoding_final]

            if self.embedding_dir is not None:
                dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))

            if self.embedding_a is not None and self.affine is None:
                dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))

            dir_a_encoding = self.dir_a_encoding(torch.cat(dir_a_encoding_input, -1))
            rgb = self.rgb(dir_a_encoding)
        else:
            rgb = self.rgb(xyz_)

        if self.affine is not None and self.embedding_a is not None:
            affine_transform = self.affine(self.embedding_a(x[:, -1].long())).view(-1, 3, 4)
            rgb = (affine_transform[:, :, :3] @ rgb.unsqueeze(-1) + affine_transform[:, :, 3:]).squeeze(-1)

        return torch.cat([self.rgb_activation(rgb) if self.rgb_activation is not None else rgb, sigma], -1)
