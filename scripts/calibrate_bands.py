"""Honest headcount ranges for the model's forecasts (conformal calibration).

The model's own imagined futures vary too little in headcount, so instead: on runs used
for training (synthetic: the validation split), measure how far the real headcount ends
up from the model's most likely forecast at every step ahead. The 10th-90th percentile
of those misses, added to a new forecast, is its 80% range. Then check on the held-out
test sets how often reality really falls inside.

Saves outputs/runs/<run>/count_bands.json (used by show_people.py).
Run:  python scripts/calibrate_bands.py --run final_real_c
"""

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import torch

from crowdwm.data.real_dataset import REAL_DIR
from crowdwm.data.sim_dataset import collate, make_batch, make_item, split_of
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON = 10, 50
HOLDOUT = ("caserne_bottleneck_2006__b120,caserne_bottleneck_2006__b200,hermes_bottleneck_2009__ao-360-400,"
           "crossing_90_d_2,crossing_90_d_8,crossing_90_e_3,day7_*,day8_*,day9_*").split(",")

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real_c")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--max_windows", type=int, default=400)
a = p.parse_args()

model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
splatter = TorchSplatter(device="cuda")
rng = np.random.default_rng(0)


def scene_dirs(name, which):
    if name == "synthetic":
        split = "val" if which == "calib" else "test"
        return [d for d in sorted((ROOT / "data/processed/sim/v3").glob("scene_*"))
                if split_of(int(d.name.split("_")[1])) == split and (d / "cache.npz").exists()]
    held = lambda d: any(fnmatch(d.name, h) for h in HOLDOUT)
    return [d for d in sorted((REAL_DIR / name).iterdir())
            if (d / "cache.npz").exists() and held(d) == (which == "test")]


def misses(dirs):
    """(windows, HORIZON) real headcount minus most likely forecast, each step ahead."""
    wins = []
    for d in dirs:
        n = len(np.load(d / "cache.npz")["ptr"]) - 1
        wins += [(d, k) for k in range(5, n - CONTEXT - HORIZON, 25)]
    if len(wins) > a.max_windows:
        wins = [wins[i] for i in rng.choice(len(wins), a.max_windows, replace=False)]
    out = []
    for d, k in wins:
        s = dict(np.load(d / "cache.npz"))
        B = make_batch(collate([make_item(s, k, CONTEXT + HORIZON, CONTEXT)]), splatter, CONTEXT + HORIZON)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT,
                               arrivals=B["arrivals"], sample=False)
        pc = pred[0, CONTEXT - 1:, 0].float().sum((-1, -2))
        rc = B["density"][0, CONTEXT:, 0].sum((-1, -2))
        out.append((rc - pc).cpu().numpy())
    return np.array(out)


bands = {}
print("80% headcount range 10 s ahead: built on training runs, checked on held-out test runs")
for name in ["synthetic", "juelich", "crossing", "eindhoven"]:
    cal, test = misses(scene_dirs(name, "calib")), misses(scene_dirs(name, "test"))
    lo, hi = np.quantile(cal, 0.1, axis=0), np.quantile(cal, 0.9, axis=0)
    bands[name] = {"lo": lo.round(2).tolist(), "hi": hi.round(2).tolist()}
    inside = ((test >= lo) & (test <= hi)).mean(0)
    print(f"  {name:10s} range {lo[-1]:+5.1f} .. {hi[-1]:+5.1f} people around the forecast | "
          f"reality inside on test: {inside[-1]:.0%} at 10 s, {inside.mean():.0%} over all steps "
          f"({len(cal)} calibration / {len(test)} test windows)")
(ROOT / "outputs/runs" / a.run / "count_bands.json").write_text(json.dumps(bands))
