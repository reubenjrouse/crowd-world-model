"""Shared rasteriser: turns any scene (real or simulated) into the grids the model sees.

Every scene goes through exactly the same code, so real and simulated crowds look
identical to the model. Output shape: (T, N_CHANNELS, H, W), float32.

Channels:
  0     DENSITY   people per cell, each person splatted as a small Gaussian
  1     WALLS     1 where you can't walk
  2..4  EXITS     one layer per exit slot (so the model can say WHICH exit people head to)
  5     GATES     1 where a gate is closed at that moment (0 when open or absent)
  6     ARRIVALS  1 where people are scheduled to arrive at that moment (e.g. train doors
                  while a train unloads); known in advance, like the gate plan

raster_torch.py implements the same density splat on the GPU for training; a test
checks the two agree.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import gaussian_filter

MAX_EXITS = 3
DENSITY, WALLS = 0, 1
EXITS = slice(2, 2 + MAX_EXITS)
GATES = 2 + MAX_EXITS
ARRIVALS = GATES + 1
N_CHANNELS = ARRIVALS + 1
CHANNEL_NAMES = ["density", "walls"] + [f"exit {i}" for i in range(MAX_EXITS)] + ["gates", "arrivals"]
SIGMA_M = 0.3           # Gaussian splat width [m]
TRUNCATE = 4.0          # kernel radius in sigmas (scipy's default)


@dataclass(frozen=True)
class Grid:
    x0: float               # world coordinate of the grid's lower-left corner [m]
    y0: float
    cell: float = 0.25      # cell size [m]
    size: int = 64          # cells per side

    @classmethod
    def centred_on(cls, geometry, cell=0.25, size=64) -> "Grid":
        minx, miny, maxx, maxy = geometry.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        half = cell * size / 2
        return cls(x0=cx - half, y0=cy - half, cell=cell, size=size)

    @property
    def edges(self):
        e = np.arange(self.size + 1) * self.cell
        return self.x0 + e, self.y0 + e

    def cell_boxes(self):
        """One shapely box per cell, indexed [row=y, col=x]."""
        xs, ys = self.edges
        x, y = np.meshgrid(xs[:-1], ys[:-1])
        return shapely.box(x, y, x + self.cell, y + self.cell)


def gaussian_kernel_1d(sigma_cells: float, truncate: float = TRUNCATE) -> np.ndarray:
    """The same normalised kernel scipy.ndimage.gaussian_filter uses."""
    radius = int(truncate * sigma_cells + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_cells) ** 2)
    return k / k.sum()


def static_layers(grid: Grid, walkable, exits=()) -> np.ndarray:
    """Walls plus one layer per exit slot, shape (1 + MAX_EXITS, H, W). Once per scene."""
    boxes = grid.cell_boxes()
    # A cell is wall if most of it is outside the walkable area, or if it touches an
    # obstacle at all (Jülich walls are only 5 cm thick, thinner than a cell)
    inside = shapely.area(shapely.intersection(boxes, walkable)) / grid.cell**2
    touches_obstacle = np.zeros_like(inside, dtype=bool)
    for hole in walkable.interiors:
        touches_obstacle |= shapely.intersects(boxes, shapely.Polygon(hole))
    out = np.zeros((1 + MAX_EXITS, grid.size, grid.size), dtype=np.float32)
    out[0] = (inside < 0.5) | touches_obstacle
    for i, ex in enumerate(exits):
        out[1 + i] = shapely.intersects(boxes, ex)
    return out


def gate_masks(grid: Grid, gates) -> np.ndarray:
    """One mask per gate polygon (None for exits without a gate), shape (len, H, W)."""
    boxes = grid.cell_boxes()
    return np.stack([shapely.intersects(boxes, g) if g is not None
                     else np.zeros((grid.size, grid.size), bool) for g in gates]).astype(np.float32)


def density_layer(grid: Grid, x: np.ndarray, y: np.ndarray, sigma_m: float = SIGMA_M) -> np.ndarray:
    """People per cell, shape (H, W). Sum ≈ number of people in the grid."""
    xs, ys = grid.edges
    h, _, _ = np.histogram2d(y, x, bins=[ys, xs])
    return gaussian_filter(h, sigma=sigma_m / grid.cell, mode="constant",
                           truncate=TRUNCATE).astype(np.float32)


def resample(traj: pd.DataFrame, fps_out: float = 5.0) -> pd.DataFrame:
    """Keep one sample per person per output frame (nearest in time). Adds column `k`."""
    k = np.round(traj["t"].to_numpy() * fps_out).astype(int)
    out = traj.assign(k=k, dt=np.abs(traj["t"] - k / fps_out))
    return out.sort_values("dt").drop_duplicates(["id", "k"]).drop(columns="dt")


def rasterise(traj: pd.DataFrame, grid: Grid, walkable, exits=(),
              gates_closed=None, fps_out: float = 5.0) -> np.ndarray:
    """Full scene -> (T, N_CHANNELS, H, W).

    traj needs columns id, t [s], x [m], y [m]. Frame k covers time k / fps_out.
    exits: up to MAX_EXITS polygons, one per exit slot.
    gates_closed: optional function k -> list of closed-gate polygons at frame k.
    """
    r = resample(traj, fps_out)
    k0, k1 = r["k"].min(), r["k"].max()
    static = static_layers(grid, walkable, exits)
    boxes = grid.cell_boxes() if gates_closed else None

    frames = np.zeros((k1 - k0 + 1, N_CHANNELS, grid.size, grid.size), dtype=np.float32)
    frames[:, WALLS:GATES] = static
    for k, g in r.groupby("k"):
        frames[k - k0, DENSITY] = density_layer(grid, g["x"].to_numpy(), g["y"].to_numpy())
    if gates_closed:
        for k in range(k0, k1 + 1):
            for gate in gates_closed(k):
                frames[k - k0, GATES] = np.maximum(
                    frames[k - k0, GATES], shapely.intersects(boxes, gate))
    return frames
