"""Training windows from synthetic scenes.

prepare_scene() turns a generated scene folder into cache.npz (done once):
  x0, y0          grid origin (grid centred on the scene)
  static          (1 + MAX_EXITS, H, W) walls + one layer per exit slot
  gate_masks      (MAX_EXITS, H, W) cells covered by each exit's gate
  gate_state      (T, MAX_EXITS) 1 where that gate is closed at frame k
  ptr             (T + 1,) points of frame k are rows ptr[k]:ptr[k+1]
  xy, target      (M, 2) positions and (M,) current target exit of every point
  hurry, n_exits  hidden crowd speed [m/s], number of exits

Frames are at 5 fps: frame k is time k / 5 s. The Dataset returns raw points for a
window; make_batch() rasterises them on the GPU with TorchSplatter.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from torch.utils.data import Dataset

from crowdwm.raster import MAX_EXITS, Grid, gate_masks, static_layers
from crowdwm.raster_torch import TorchSplatter

FPS = 5.0


def prepare_scene(scene_dir: Path, force: bool = False) -> Path | None:
    """Returns the cache path, or None for a scene with nobody in it (no cache written)."""
    from crowdwm.sim.scenes import SceneSpec, exit_parts, walkable_area

    out = scene_dir / "cache.npz"
    if out.exists() and not force:
        return out
    spec = SceneSpec.from_dict(json.loads((scene_dir / "spec.json").read_text()))
    traj = pd.read_parquet(scene_dir / "traj.parquet")
    if traj.empty:
        return None
    gates = pd.read_parquet(scene_dir / "gates.parquet")

    area = walkable_area(spec)
    grid = Grid.centred_on(area)
    parts = [exit_parts(spec, e) for e in spec.exits]
    static = static_layers(grid, area, [p["stage"] for p in parts])
    masks = np.zeros((MAX_EXITS, grid.size, grid.size), np.uint8)
    masks[:len(parts)] = gate_masks(grid, [p["gate"] if e.gated else None
                                           for p, e in zip(parts, spec.exits)])

    k = np.round(traj["t"].to_numpy() * FPS).astype(np.int64)
    n_frames = int(k.max()) + 1
    order = np.argsort(k, kind="stable")
    ptr = np.searchsorted(k[order], np.arange(n_frames + 1))
    state = np.zeros((n_frames, MAX_EXITS), np.uint8)
    gk = np.round(gates["t"].to_numpy() * FPS).astype(np.int64)
    cols = [c for c in gates.columns if c.startswith("closed_")]
    state[gk[gk < n_frames], :len(cols)] = gates[cols].to_numpy()[gk < n_frames]

    # Scheduled arrivals (train doors): a burst group's zone is active from its first to its
    # last person stepping out. Steady streams and waiting crowds are not scheduled.
    arrivals = np.zeros((n_frames, grid.size, grid.size), bool)
    people = pd.read_parquet(scene_dir / "people.parquet")
    boxes = grid.cell_boxes()
    for gi, g in enumerate(spec.groups):
        if g.kind != "burst":
            continue
        t = people.loc[people["group"] == gi, "spawn_t"]
        if t.empty:
            continue
        k0, k1 = int(round(t.min() * FPS)), int(round(t.max() * FPS))
        arrivals[k0:k1 + 1] |= shapely.intersects(boxes, shapely.box(*g.zone))

    np.savez_compressed(
        out, x0=grid.x0, y0=grid.y0, static=static.astype(np.uint8), gate_masks=masks,
        gate_state=state, ptr=ptr, xy=traj[["x", "y"]].to_numpy(np.float32)[order],
        target=traj["target"].to_numpy(np.int8)[order], pid=traj["id"].to_numpy(np.int64)[order],
        hurry=np.float32(spec.hurry),
        n_exits=len(spec.exits), arrivals=pack_arrivals(arrivals),
        inflow=pack_arrivals(first_appearances(k, traj["id"].to_numpy(), traj[["x", "y"]].to_numpy(),
                                               n_frames, grid)))
    return out


def pack_arrivals(a: np.ndarray) -> np.ndarray:
    """(T, H, W) bool -> bit-packed uint8 (T, H, W/8), to keep caches small."""
    return np.packbits(a, axis=-1)


def unpack_arrivals(s: dict, k: int, length: int, size: int = 64, key: str = "arrivals") -> np.ndarray:
    """Packed masks (scheduled 'arrivals' or observed 'inflow') for frames k..k+length-1
    (zeros if the scene has none)."""
    if key not in s:
        return np.zeros((length, size, size), np.uint8)
    a = np.unpackbits(s[key][k:k + length], axis=-1)[..., :size]
    if len(a) < length:   # windows running past the last recorded frame
        a = np.concatenate([a, np.zeros((length - len(a), size, size), np.uint8)])
    return a


def first_appearances(k: np.ndarray, ids: np.ndarray, xy: np.ndarray, n_frames: int, grid) -> np.ndarray:
    """(T, H, W) bool: cells where a person is seen for the first time at each frame
    (spawning in a simulation, or walking into view in a recording)."""
    out = np.zeros((n_frames, grid.size, grid.size), bool)
    order = np.argsort(k, kind="stable")
    _, first = np.unique(ids[order], return_index=True)
    idx = order[first]
    col = np.clip(((xy[idx, 0] - grid.x0) / grid.cell).astype(int), 0, grid.size - 1)
    row = np.clip(((xy[idx, 1] - grid.y0) / grid.cell).astype(int), 0, grid.size - 1)
    out[k[idx], row, col] = True
    return out


def window_arrivals(s: dict, k: int, length: int, context: int) -> np.ndarray:
    """The ARRIVALS input for a window: scheduled arrivals (train doors; known in advance)
    for every frame, plus where people were observed walking in during the watched frames
    only. Future walk-ins are not given: the model has to continue the observed trend."""
    a = unpack_arrivals(s, k, length).copy()
    a[:context] |= unpack_arrivals(s, k, context, key="inflow")
    return a


def make_item(s: dict, start: int, length: int, context: int = 10, family: int = 0,
              gate_state: np.ndarray | None = None) -> dict:
    """One model-input window (frames start..start+length-1) from a scene cache. Every
    script builds inputs through this, so training and evaluation see the same thing."""
    lo, hi = s["ptr"][start], s["ptr"][start + length]
    # Static layers: walls, exit slots, and the open edge of the camera view (where people
    # walk in or out away from marked exits; empty for closed synthetic rooms)
    boundary = s["boundary"][None] if "boundary" in s else np.zeros((1,) + s["static"].shape[1:], np.uint8)
    return dict(xy=s["xy"][lo:hi], target=s["target"][lo:hi].astype(np.int64),
                frame=np.repeat(np.arange(length), np.diff(s["ptr"][start:start + length + 1])),
                static=np.concatenate([s["static"], boundary]), gate_masks=s["gate_masks"],
                gate_state=s["gate_state"][start:start + length] if gate_state is None else gate_state,
                x0=float(s["x0"]), y0=float(s["y0"]), hurry=float(s["hurry"]), n_exits=int(s["n_exits"]),
                family=family, arrivals=window_arrivals(s, start, length, context))


FAMILY_NAMES = ["hall", "platform", "bottleneck"]


def split_of(seed: int) -> str:
    return {18: "val", 19: "test"}.get(seed % 20, "train")


class SimWindows(Dataset):
    """Every (scene, start frame) window of `length` frames, stepping by `stride`.

    root: one scene folder root, or a list of them (e.g. halls + platforms + bottlenecks).
    """

    def __init__(self, root=None, split: str = "train", length: int = 30, stride: int = 5,
                 max_scenes: int | None = None, dirs: list | None = None, context: int = 10):
        """Pass `dirs` to use an explicit list of scene folders (e.g. real runs) instead.
        context: number of watched frames (observed walk-ins are only shown for those)."""
        self.context = context
        if dirs is None:
            roots = [root] if isinstance(root, (str, Path)) else list(root)
            dirs = sorted(d for r in roots for d in Path(r).glob("scene_*") if (d / "cache.npz").exists()
                          and split_of(int(d.name.split("_")[1])) == split)
        dirs = [Path(d) for d in dirs]
        self.dirs = dirs[:max_scenes] if max_scenes else dirs
        self.length = length
        fams = [json.loads((d / "spec.json").read_text()).get("family", "hall") for d in self.dirs]
        self.family = [FAMILY_NAMES.index(f) if f in FAMILY_NAMES else 0 for f in fams]
        self.index = []
        for i, d in enumerate(self.dirs):
            n_frames = len(np.load(d / "cache.npz")["ptr"]) - 1
            self.index += [(i, k) for k in range(0, n_frames - length + 1, stride)]
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _scene(self, i):
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = dict(np.load(self.dirs[i] / "cache.npz"))
        return self._cache[i]

    def __getitem__(self, j):
        i, k = self.index[j]
        return make_item(self._scene(i), k, self.length, self.context, family=self.family[i])


def collate(items):
    """Stack windows; points are concatenated with a batch index."""
    out = {key: torch.as_tensor(np.stack([it[key] for it in items]))
           for key in ["static", "gate_masks", "gate_state", "x0", "y0", "hurry", "n_exits", "family"]}
    out["xy"] = torch.as_tensor(np.concatenate([it["xy"] for it in items]))
    out["target"] = torch.as_tensor(np.concatenate([it["target"] for it in items]))
    out["frame"] = torch.as_tensor(np.concatenate([it["frame"] for it in items]))
    out["batch"] = torch.as_tensor(np.concatenate([np.full(len(it["xy"]), b)
                                                   for b, it in enumerate(items)]))
    out["arrivals"] = torch.as_tensor(np.stack([   # scheduled arrivals; none if not given
        it["arrivals"] if "arrivals" in it else
        np.zeros((len(it["gate_state"]),) + it["static"].shape[1:], np.uint8) for it in items]))
    return out


def make_batch(b: dict, splatter: TorchSplatter, length: int) -> dict:
    """Rasterise a collated batch on the GPU.

    Returns density (B, L, 1, H, W), dest (B, L, MAX_EXITS, H, W) = density split by
    target exit, static (B, 1 + MAX_EXITS, H, W), gates (B, L, 1, H, W), hurry (B,),
    exit_mask (B, MAX_EXITS) marking exit slots that exist.
    """
    dev = splatter.device
    bsz = b["static"].shape[0]
    fidx = (b["batch"] * length + b["frame"]).to(dev)
    x0 = b["x0"].float().to(dev).repeat_interleave(length)
    y0 = b["y0"].float().to(dev).repeat_interleave(length)
    onehot = torch.nn.functional.one_hot(b["target"].to(dev), MAX_EXITS).float()
    dest = splatter.splat(b["xy"].to(dev), fidx, bsz * length, x0, y0, weights=onehot)
    size = dest.shape[-1]
    dest = dest.view(bsz, length, MAX_EXITS, size, size)
    gates = torch.einsum("ble,behw->blhw", b["gate_state"].float().to(dev),
                         b["gate_masks"].float().to(dev)).clamp(max=1).unsqueeze(2)
    exit_mask = torch.arange(MAX_EXITS, device=dev)[None] < b["n_exits"].to(dev)[:, None]
    arrivals = b["arrivals"].float().to(dev).unsqueeze(2) if "arrivals" in b else torch.zeros_like(gates)
    return dict(density=dest.sum(2, keepdim=True), dest=dest,
                static=b["static"].float().to(dev), gates=gates, arrivals=arrivals,
                hurry=b["hurry"].float().to(dev), exit_mask=exit_mask)


def augment(B: dict, generator: torch.Generator | None = None) -> dict:
    """Random rotation by 90-degree steps and mirroring, independently per window.

    Every grid in a window gets the same transform, so the scene stays consistent; the
    model learns that crowd behaviour doesn't depend on which way the scene faces.
    """
    out = {k: v.clone() for k, v in B.items()}
    bsz = B["static"].shape[0]
    ks = torch.randint(0, 4, (bsz,), generator=generator).tolist()
    flips = torch.randint(0, 2, (bsz,), generator=generator).tolist()
    for i, (k, f) in enumerate(zip(ks, flips)):
        for key in [k for k in ["density", "dest", "static", "gates", "arrivals"] if k in out]:
            x = out[key][i]
            if f:
                x = x.flip(-1)
            out[key][i] = torch.rot90(x, k, dims=(-2, -1))
    return out
