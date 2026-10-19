"""Destination inference on real people (crossing experiments, parts D and E).

At moments every 2 s, each person in view is assigned an exit by:
  model     the world model's destination map at the person's cell, after 2 s of watching
  nearest   the nearest exit (what a blind simulator assumes)
  heading   the exit most in line with the person's direction of motion over the last 1 s
and compared with the exit where the person's path really ends. Reported overall and
for people in the middle of the crossing (where the two streams mix).

Run:  python scripts/eval_destinations.py wm_v2 wm_v2_real
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely
import torch
from shapely.ops import nearest_points

from crowdwm.data.real_dataset import REAL_DIR
from crowdwm.data.sim_dataset import collate, make_batch, make_item
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 10

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--dataset", default="crossing")
p.add_argument("--stride", type=int, default=10)
p.add_argument("--only", default=None, help="comma-separated run folders to include")
a = p.parse_args()

splatter = TorchSplatter(device="cuda")
models = {}
for run in a.runs:
    ckpt = "last.pt" if "_real" in run else "best.pt"
    m = load_checkpoint(ROOT / "outputs/runs" / run / ckpt)
    models[run] = m

hits = {k: [] for k in ["nearest", "heading", *a.runs]}
all_targets = []
mid_hits = {k: [] for k in hits}
per_run = []
for d in sorted((REAL_DIR / a.dataset).iterdir()):
    if a.only and d.name not in a.only.split(","):
        continue
    meta = json.loads((d / "spec.json").read_text())
    s = dict(np.load(d / "cache.npz"))
    exits = [shapely.from_wkt(w) for w in meta["exits_wkt"]]
    walk = shapely.from_wkt(meta["walkable_wkt"])
    mid = shapely.box(*walk.centroid.buffer(2.5).bounds)          # the crossing's centre square
    run_hits = {k: [] for k in hits}
    n_frames = len(s["ptr"]) - 1
    for k0 in range(CONTEXT, n_frames, a.stride):
        start = k0 - CONTEXT + 1
        lo, hi = s["ptr"][k0], s["ptr"][k0 + 1]
        if hi - lo < 5:
            continue
        xy, target, pid = s["xy"][lo:hi], s["target"][lo:hi], s["pid"][lo:hi]
        # Positions 1 s earlier, for the heading rule
        plo, phi = s["ptr"][k0 - 5], s["ptr"][k0 - 4]
        prev = dict(zip(s["pid"][plo:phi], s["xy"][plo:phi]))
        guesses = {"nearest": [], "heading": []}
        for (x, y), i in zip(xy, pid):
            pt = shapely.Point(x, y)
            # Nearest point of each exit (exits can be several pieces, e.g. both track edges)
            near = np.array([[q.x, q.y] for q in (nearest_points(e, pt)[0] for e in exits)])
            dist = np.hypot(near[:, 0] - x, near[:, 1] - y)
            guesses["nearest"].append(int(np.argmin(dist)))
            if i in prev and np.hypot(*(np.array([x, y]) - prev[i])) > 0.05:
                v = np.array([x, y]) - prev[i]
                to = near - np.array([x, y])
                cos = to @ v / (np.linalg.norm(to, axis=1) * np.linalg.norm(v) + 1e-9)
                guesses["heading"].append(int(np.argmax(cos)))
            else:
                guesses["heading"].append(int(np.argmin(dist)))   # standing still: fall back
        all_targets += list(target)
        lo_w, hi_w = s["ptr"][start], s["ptr"][k0 + 1]
        item = make_item(s, start, CONTEXT, CONTEXT)
        B = make_batch(collate([item]), splatter, CONTEXT)
        col = np.clip(((xy[:, 0] - s["x0"]) / 0.25).astype(int), 0, 63)
        row = np.clip(((xy[:, 1] - s["y0"]) / 0.25).astype(int), 0, 63)
        for name, model in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, dest, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT)
            logits = dest[0, CONTEXT - 1].float().cpu().numpy()
            guesses[name] = list(logits[:, row, col].argmax(0))
        in_mid = shapely.contains_xy(mid, xy[:, 0], xy[:, 1])
        for name, g in guesses.items():
            ok = (np.array(g) == target)
            hits[name] += list(ok)
            mid_hits[name] += list(ok[in_mid])
            run_hits[name] += list(ok)
    per_run.append((meta["run"], {k: float(np.mean(v)) for k, v in run_hits.items()}, len(run_hits["nearest"])))

print(f"\nReal destination inference: {a.dataset} runs, after 2 s of watching\n")
print(f"{'run':18s}{'people-moments':>15s}" + "".join(f"{k:>12s}" for k in hits))
for name, acc, n in per_run:
    print(f"{name:18s}{n:15d}" + "".join(f"{acc[k]:12.3f}" for k in hits))
print(f"\n{'ALL':18s}{len(hits['nearest']):15d}" + "".join(f"{np.mean(v):12.3f}" for v in hits.values()))
print(f"{'centre region':18s}{len(mid_hits['nearest']):15d}" + "".join(f"{np.mean(v):12.3f}" for v in mid_hits.values()))
share = np.bincount(all_targets) / len(all_targets)
print(f"\nDestination mix: " + ", ".join(f"exit {i} {s:.0%}" for i, s in enumerate(share))
      + f"  ->  'everyone goes to the most common exit' would score {share.max():.3f}")
