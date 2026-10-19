"""GPU version of the density splat in raster.py, for rasterising training batches.

Same maths as raster.density_layer: count people per cell, then blur with the same
truncated, normalised Gaussian (zero padding). tests/test_raster.py checks agreement.
"""

import torch
import torch.nn.functional as F

from crowdwm.raster import SIGMA_M, gaussian_kernel_1d


class TorchSplatter:
    def __init__(self, cell: float = 0.25, size: int = 64, sigma_m: float = SIGMA_M,
                 device: str | torch.device = "cuda"):
        self.cell, self.size, self.device = cell, size, torch.device(device)
        k = torch.tensor(gaussian_kernel_1d(sigma_m / cell), dtype=torch.float32, device=self.device)
        self.kx = k.view(1, 1, 1, -1)
        self.ky = k.view(1, 1, -1, 1)
        self.pad = (len(k) - 1) // 2

    def splat(self, xy: torch.Tensor, frame: torch.Tensor, n_frames: int, x0: torch.Tensor,
              y0: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Rasterise points into (n_frames, C, H, W).

        xy: (M, 2) world positions; frame: (M,) output frame index of each point;
        x0, y0: (n_frames,) grid origin per frame (so frames from different scenes
        can share one call); weights: (M, C) per-point weights (default: one channel of 1s).
        """
        if weights is None:
            weights = torch.ones(len(xy), 1, device=self.device)
        c = weights.shape[1]
        col = torch.floor((xy[:, 0] - x0[frame]) / self.cell).long()
        row = torch.floor((xy[:, 1] - y0[frame]) / self.cell).long()
        ok = (col >= 0) & (col < self.size) & (row >= 0) & (row < self.size)
        flat = ((frame[ok] * self.size + row[ok]) * self.size + col[ok])
        grid = torch.zeros(n_frames * self.size * self.size, c, device=self.device)
        grid.index_add_(0, flat, weights[ok].float())
        grid = grid.view(n_frames, self.size, self.size, c).permute(0, 3, 1, 2)
        g = grid.reshape(n_frames * c, 1, self.size, self.size)
        g = F.conv2d(F.pad(g, (self.pad, self.pad, 0, 0)), self.kx)
        g = F.conv2d(F.pad(g, (0, 0, self.pad, self.pad)), self.ky)
        return g.view(n_frames, c, self.size, self.size)
