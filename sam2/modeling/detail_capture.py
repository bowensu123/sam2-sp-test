# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A ViTMatte-style detail-capture module for SAM 2, designed to be memory-friendly
at high input resolutions (e.g. 1024x1024).

A lightweight convolutional stream extracts image details at strides 2/4/8 and
projects them to the channel dims of the two high-resolution feature maps that
the SAM mask decoder consumes (stride 4 and stride 8). The projection heads are
zero-initialized, so at the start of finetuning the module is an exact no-op
and the model behaves identically to a pretrained SAM 2 checkpoint.

Memory notes:
- Intended to be called per frame inside the tracking loop and under
  `sam2.modeling.sam2_utils.checkpoint_module`, so none of its full-resolution
  intermediate activations persist until backward (they are recomputed).
- Uses GroupNorm by default: unlike BatchNorm, results do not depend on how
  many frames are batched together, so per-frame (lazy) computation is
  bit-identical to batched computation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    if norm == "GN":
        return nn.GroupNorm(num_groups=min(8, num_channels), num_channels=num_channels)
    if norm == "BN":
        # note: with BatchNorm, per-frame computation is NOT identical to
        # batched computation (batch statistics differ); prefer GN
        return nn.BatchNorm2d(num_channels)
    raise ValueError(f"unsupported norm type: {norm}")


class BasicConv3x3(nn.Sequential):
    def __init__(self, in_chans: int, out_chans: int, stride: int, norm: str):
        super().__init__(
            nn.Conv2d(in_chans, out_chans, 3, stride=stride, padding=1, bias=False),
            _make_norm(norm, out_chans),
            nn.ReLU(inplace=True),
        )


class DetailCapture(nn.Module):
    """
    Extract high-frequency image details and project them as residuals for the
    stride-4 and stride-8 high-resolution SAM features.

    Args:
        in_chans: input image channels.
        mid_chans: conv-stream channels at strides 2, 4, 8.
        out_chans: output channels at strides 4 and 8; must match the SAM mask
            decoder's high-res feature dims (transformer_dim // 8 and // 4).
        norm: "GN" (default, safe for per-frame computation) or "BN".
    """

    def __init__(
        self,
        in_chans: int = 3,
        mid_chans: Tuple[int, int, int] = (24, 48, 96),
        out_chans: Tuple[int, int] = (32, 64),
        norm: str = "GN",
    ):
        super().__init__()
        c2, c4, c8 = mid_chans
        self.stream = nn.ModuleList(
            [
                BasicConv3x3(in_chans, c2, stride=2, norm=norm),  # 1/1 -> 1/2
                BasicConv3x3(c2, c4, stride=2, norm=norm),  # 1/2 -> 1/4
                BasicConv3x3(c4, c8, stride=2, norm=norm),  # 1/4 -> 1/8
            ]
        )
        self.head_s4 = nn.Conv2d(c4, out_chans[0], kernel_size=1)
        self.head_s8 = nn.Conv2d(c8, out_chans[1], kernel_size=1)
        # zero-init the heads so the module starts as an exact no-op and the
        # pretrained SAM 2 behavior is unchanged at the beginning of finetuning
        for head in (self.head_s4, self.head_s8):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """images: (B, 3, H, W) -> [detail_s4 (B, C4, H/4, W/4), detail_s8 (B, C8, H/8, W/8)]"""
        x2 = self.stream[0](images)
        x4 = self.stream[1](x2)
        x8 = self.stream[2](x4)
        return [self.head_s4(x4), self.head_s8(x8)]
