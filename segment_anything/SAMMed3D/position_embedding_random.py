# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn


class PositionEmbeddingRandom3D(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: Optional[float] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # self.register_buffer(
        #     "positional_encoding_gaussian_matrix",
        #     scale * torch.randn((3, num_pos_feats)),
        # )
        self.positional_encoding_gaussian_matrix = nn.Parameter(
            scale * torch.randn((3, num_pos_feats), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.dtype = dtype

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat(
            [torch.sin(coords), torch.cos(coords), torch.sin(coords)], dim=-1
        ).to(dtype=self.dtype, device=coords.device)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        x, y, z = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((x, y, z), device=device, dtype=self.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / y
        x_embed = x_embed / x
        z_embed = z_embed / z

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x X x Y x Z

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(self.dtype))  # B x N x C
