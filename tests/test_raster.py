"""The GPU splat must match the reference rasteriser (same pictures for real and sim)."""

import numpy as np
import pytest
import torch

from crowdwm.raster import Grid, density_layer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_torch_splat_matches_numpy():
    from crowdwm.raster_torch import TorchSplatter

    rng = np.random.default_rng(0)
    grids = [Grid(-8.0, -7.5), Grid(-2.0, -3.0)]
    frames = [rng.uniform(-9, 9, size=(n, 2)) for n in (1, 50, 300)]

    xy, fidx, ref, x0, y0 = [], [], [], [], []
    for f, pts in enumerate(frames):
        grid = grids[f % 2]
        ref.append(density_layer(grid, pts[:, 0], pts[:, 1]))
        xy.append(pts)
        fidx.append(np.full(len(pts), f))
        x0.append(grid.x0)
        y0.append(grid.y0)

    s = TorchSplatter(device="cuda")
    out = s.splat(torch.tensor(np.concatenate(xy), dtype=torch.float32, device="cuda"),
                  torch.tensor(np.concatenate(fidx), device="cuda"), len(frames),
                  torch.tensor(x0, device="cuda"), torch.tensor(y0, device="cuda"))
    got = out[:, 0].cpu().numpy()
    np.testing.assert_allclose(got, np.stack(ref), atol=1e-5)


@cuda
def test_weighted_channels_sum_to_density():
    from crowdwm.raster_torch import TorchSplatter

    rng = np.random.default_rng(1)
    pts = torch.tensor(rng.uniform(-7, 7, (200, 2)), dtype=torch.float32, device="cuda")
    target = torch.tensor(rng.integers(0, 3, 200), device="cuda")
    s = TorchSplatter(device="cuda")
    z = torch.zeros(1, device="cuda") - 8
    fr = torch.zeros(200, dtype=torch.long, device="cuda")
    dens = s.splat(pts, fr, 1, z, z)
    per_exit = s.splat(pts, fr, 1, z, z, weights=torch.nn.functional.one_hot(target, 3))
    torch.testing.assert_close(per_exit.sum(1), dens[:, 0], atol=1e-5, rtol=0)
