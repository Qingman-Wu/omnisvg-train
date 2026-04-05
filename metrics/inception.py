"""
InceptionV3 feature extractor for FID computation.
Based on the widely used pytorch-fid implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps at selected layers."""

    BLOCK_INDEX_BY_DIM = {
        64:   0,
        192:  1,
        768:  2,
        2048: 3,
    }

    def __init__(self, output_blocks=(3,), resize_input=True, normalize_input=True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT,
        )

        self.blocks = nn.ModuleList()

        # Block 0: up to maxpool1 → 64-d
        block0 = [
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: up to maxpool2 → 192-d
        block1 = [
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block1))

        # Block 2: up to aux classifier → 768-d
        block2 = [
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))

        # Block 3: final pool → 2048-d
        block3 = [
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ]
        self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = []
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1  # [0,1] → [-1,1]

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                out.append(x)
            if idx == self.last_needed_block:
                break

        return out
