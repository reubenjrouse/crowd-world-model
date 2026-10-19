"""Does the final model work? One table per test set, against simple competitors.

For held-out windows (2 s watched, then 4 s / 10 s ahead), people misplaced on the
map (lower is better) for:
  nothing moves      last watched frame, frozen
  constant velocity  every person keeps the speed and direction of their last 1 s
                     (stops at walls); the standard trajectory-forecasting baseline
  model, most likely the model's most likely future (prior mean): its forecast
  model, one future  one sampled imagination
  model, mean of N   average of N sampled futures (the samples are used for the odds)
plus headcount error, sharpness of single futures, and danger odds (Brier score of
"some 1 m^2 holds >= the 75th-percentile crowding within 10 s").

Run:  python scripts/final_check.py --run final_real
"""

import argparse
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crowdwm.data.real_dataset import REAL_DIR
from crowdwm.data.sim_dataset import collate, make_batch, make_item, split_of
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster import Grid, density_layer
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, AHEAD = 10, 50, {"4s": 20, "10s": 50}
SETS = {
    "synthetic test": ("sim", "v3", None),
    "Jülich bottlenecks (held out)": ("real", "juelich", "caserne_bottleneck_2006__b120,caserne_bottleneck_2006__b200,hermes_bottleneck_2009__ao-360-400"),
    "crossings (held out)": ("real", "crossing", "crossing_90_d_2,crossing_90_d_8,crossing_90_e_3"),
    "Eindhoven days 7-9": ("real", "eindhoven", "day7_*,day8_*,day9_*"),
}

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--samples", type=int, default=16)
p.add_argument("--max_windows", type=int, default=300)
a = p.parse_args()

model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
splatter = TorchSplatter(device="cuda")
rng = np.random.default_rng(0)


def constant_velocity(s, k0, grid, walls):
    """Positions of everyone seen at frame k0, moved at their velocity over the last 1 s."""
    lo, hi = s["ptr"][k0], s["ptr"][k0 + 1]
    xy, pid = s["xy"][lo:hi].astype(float), s["pid"][lo:hi]
    plo, phi = s["ptr"][k0 - 5], s["ptr"][k0 - 4]
    prev = dict(zip(s["pid"][plo:phi], s["xy"][plo:phi]))
    vel = np.zeros((len(xy), 2))
    for i, q in enumerate(pid):
        if q in prev:
            vel[i] = (xy[i] - prev[q]) / 5
    pos, out = xy.copy(), {}
    for n in range(1, HORIZON + 1):
        new = pos + vel
        col = np.clip(((new[:, 0] - grid.x0) / grid.cell).astype(int), 0, 63)
        row = np.clip(((new[:, 1] - grid.y0) / grid.cell).astype(int), 0, 63)
        stop = walls[row, col] > 0
        vel[stop] = 0
        new[stop] = pos[stop]
        pos = new
        if n in AHEAD.values():
            out[n] = density_layer(grid, pos[:, 0], pos[:, 1])
    return out


def busiest_m2(d):   # (..., H, W) -> max people in any 1 m x 1 m area
    t = torch.as_tensor(d, dtype=torch.float32)
    sh = t.shape
    return (F.avg_pool2d(t.reshape(-1, 1, *sh[-2:]), 4, stride=1) * 16).reshape(*sh[:-2], -1).amax(-1).numpy()


for title, (kind, name, only) in SETS.items():
    if kind == "sim":
        dirs = [d for d in sorted((ROOT / "data/processed/sim" / name).glob("scene_*"))
                if split_of(int(d.name.split("_")[1])) == "test" and (d / "cache.npz").exists()]
    else:
        dirs = [d for d in sorted((REAL_DIR / name).iterdir())
                if (d / "cache.npz").exists() and any(fnmatch(d.name, o) for o in only.split(","))]
    windows = []
    for d in dirs:
        n_frames = len(np.load(d / "cache.npz")["ptr"]) - 1
        windows += [(d, k) for k in range(5, n_frames - CONTEXT - HORIZON, 25)]
    if len(windows) > a.max_windows:
        windows = [windows[i] for i in rng.choice(len(windows), a.max_windows, replace=False)]

    rec = {k: {lbl: [] for lbl in AHEAD} for k in ["persist", "cv", "likely", "single", "mean"]}
    cnt = {k: [] for k in ["persist", "likely", "mean"]}
    sharp, busy_true, busy_samp = [], [], []
    for d, start in windows:
        s = dict(np.load(d / "cache.npz"))
        k0 = start + CONTEXT - 1
        grid = Grid(float(s["x0"]), float(s["y0"]))
        B = make_batch(collate([make_item(s, start, CONTEXT + HORIZON, CONTEXT)]), splatter, CONTEXT + HORIZON)
        rep = {k: v.repeat_interleave(a.samples, 0) for k, v in B.items() if torch.is_tensor(v) and v.dim() > 0}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            samp, _, _ = model(rep["density"], rep["static"], rep["gates"], rep["exit_mask"], CONTEXT,
                               arrivals=rep["arrivals"], sample=True)
            likely, _, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT,
                                 arrivals=B["arrivals"], sample=False)
        samp = samp.float()[:, :, 0].cpu().numpy()            # (N, L-1, H, W)
        likely = likely.float()[0, :, 0].cpu().numpy()
        true = B["density"][0, :, 0].cpu().numpy()            # (L, H, W)
        last = true[CONTEXT - 1]
        cv = constant_velocity(s, k0, grid, s["static"][0])
        for lbl, n in AHEAD.items():
            j, tr = CONTEXT - 2 + n, true[CONTEXT - 1 + n]
            rec["persist"][lbl].append(np.abs(last - tr).sum())
            rec["cv"][lbl].append(np.abs(cv[n] - tr).sum())
            rec["likely"][lbl].append(np.abs(likely[j] - tr).sum())
            rec["single"][lbl].append(np.abs(samp[0, j] - tr).sum())
            rec["mean"][lbl].append(np.abs(samp[:, j].mean(0) - tr).sum())
        tr10 = true[-1]
        cnt["persist"].append(abs(last.sum() - tr10.sum()))
        cnt["mean"].append(abs(samp[:, -1].sum((-1, -2)).mean() - tr10.sum()))
        cnt["likely"].append(abs(likely[-1].sum() - tr10.sum()))
        if tr10.max() > 0.05:
            sharp.append(samp[:, -1].max((-1, -2)).mean() / tr10.max())
        busy_true.append(busiest_m2(true[CONTEXT:]).max())
        busy_samp.append(busiest_m2(samp[:, CONTEXT - 1:]).max(-1))
    busy_true, busy_samp = np.array(busy_true), np.array(busy_samp)
    thr = np.quantile(busy_true, 0.75)
    event, prob = busy_true >= thr, (busy_samp >= thr).mean(1)

    print(f"\n=== {title}: {len(windows)} windows from {len(dirs)} scenes/runs")
    print(f"  people misplaced          4 s ahead   10 s ahead   (score vs nothing moves at 10 s)")
    for k, label in [("persist", "nothing moves"), ("cv", "constant velocity"), ("likely", "MODEL, most likely"),
                     ("single", "model, one future"),
                     ("mean", f"model, mean of {a.samples}")]:
        m4, m10 = np.mean(rec[k]["4s"]), np.mean(rec[k]["10s"])
        print(f"  {label:24s}{m4:10.1f}{m10:13.1f}      ({m10 / np.mean(rec['persist']['10s']):.2f})")
    print(f"  headcount error at 10 s: nothing moves {np.mean(cnt['persist']):.1f}, model most likely "
          f"{np.mean(cnt['likely']):.1f}, mean of samples {np.mean(cnt['mean']):.1f}")
    print(f"  sharpness of one future at 10 s (peak / real peak): {np.mean(sharp):.2f}")
    print(f"  danger odds (>= {thr:.2f} people in 1 m^2 within 10 s; happens in {event.mean():.0%}): "
          f"Brier {np.mean((prob - event) ** 2):.3f} vs base rate {np.mean((event.mean() - event) ** 2):.3f}")
