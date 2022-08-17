from typing import Dict

import torch
import torch.nn.functional as F
import lpips as plips


def psnr(rgbs: torch.Tensor, target_rgbs: torch.Tensor) -> float:
    mse = torch.mean((rgbs - target_rgbs) ** 2)
    return -10 * torch.log10(mse).item()


def lpips(rgbs: torch.Tensor, target_rgbs: torch.Tensor) -> Dict[str, float]:
    gt = target_rgbs.permute([2, 0, 1]).contiguous()
    pred = rgbs.permute([2, 0, 1]).contiguous()

    lpips_vgg = plips.LPIPS(net='vgg').eval().to(rgbs.device)
    lpips_vgg_i = lpips_vgg(gt, pred, normalize=True)

    lpips_alex = plips.LPIPS(net='alex').eval().to(rgbs.device)
    lpips_alex_i = lpips_alex(gt, pred, normalize=True)

    lpips_squeeze = plips.LPIPS(net='squeeze').eval().to(rgbs.device)
    lpips_squeeze_i = lpips_squeeze(gt, pred, normalize=True)

    return {'vgg': lpips_vgg_i.item(), 'alex': lpips_alex_i.item(), 'squeeze': lpips_squeeze_i.item()}


#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
def ssim(
        rgbs: torch.Tensor,
        target_rgbs: torch.Tensor,
        max_val: float,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
) -> float:
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.
    Args:
      rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      target_rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
    Returns:
      Each image's mean SSIM.
    """
    device = rgbs.device
    ori_shape = rgbs.size()
    width, height, num_channels = ori_shape[-3:]
    rgbs = rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    target_rgbs = target_rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(rgbs)
    mu1 = filt_fn(target_rgbs)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgbs ** 2) - mu00
    sigma11 = filt_fn(target_rgbs ** 2) - mu11
    sigma01 = filt_fn(rgbs * target_rgbs) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    return torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1).item()
