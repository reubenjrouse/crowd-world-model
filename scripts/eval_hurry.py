"""Does the model's hurry estimate tell high-motivation crowds from low-motivation ones?

CrowdQueue runs come in pairs with the same corridor width and priming, differing only
in instructions: high motivation (h0) vs low (h-). For each run the model's hurry
estimate is averaged over every 2 s window; within each pair we check whether the
high-motivation run gets the higher estimate. The model never saw these runs or any
motivation labels. For reference, the same check is done with simple measured
quantities (median walking speed, and time for the whole group to get through).

Run:  python scripts/eval_hurry.py wm_v2 wm_v3_real
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import torch

from crowdwm.data.real_dataset import REAL_DIR
from crowdwm.data.sim_dataset import collate, make_batch, make_item
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 10
PAIRS = [("030", "040"), ("050", "060"), ("070", "080"), ("090", "100"), ("110", "120"),
         ("150", "160"), ("170", "180"), ("190", "200"), ("230", "240"), ("250", "260"), ("270", "280")]

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
a = p.parse_args()

splatter = TorchSplatter(device="cuda")
models = {}
for run in a.runs:
    ckpt = "last.pt" if ("_real" in run or run.startswith("wm_s")) else "best.pt"
    m = load_checkpoint(ROOT / "outputs/runs" / run / ckpt)
    models[run] = m

folders = {d.name[:3]: d for d in (REAL_DIR / "crowdqueue").iterdir()}
stats = {}
for code, d in folders.items():
    s = dict(np.load(d / "cache.npz"))
    meta = json.loads((d / "spec.json").read_text())
    n_frames = len(s["ptr"]) - 1
    est = {name: [] for name in models}
    for k0 in range(CONTEXT - 1, n_frames, 5):
        start = k0 - CONTEXT + 1
        if np.diff(s["ptr"][start:k0 + 1]).min() < 3:
            continue
        lo, hi = s["ptr"][start], s["ptr"][k0 + 1]
        item = make_item(s, start, CONTEXT, CONTEXT, family=2)
        B = make_batch(collate([item]), splatter, CONTEXT)
        for name, model in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, h = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT)
            est[name].append(h.item())
    stats[code] = dict(run=d.name, motivation=meta["motivation"], width=meta["width"],
                       speed=float(s["hurry"]), clear_time=n_frames / 5,
                       **{name: float(np.mean(v)) if v else np.nan for name, v in est.items()})

cols = ["speed", "clear_time", *models]
print(f"\nHurry estimate per run (model: mean over 2 s windows, m/s); speed = measured median walking speed\n")
print(f"{'pair':10s}{'width':>6s}" + "".join(f"{c + ' high':>16s}{c + ' low':>15s}" for c in cols))
wins = {c: 0 for c in cols}
for hi, lo in PAIRS:
    H, L = stats[hi], stats[lo]
    row = f"{hi}/{lo}   {H['width']:6.1f}"
    for c in cols:
        row += f"{H[c]:16.3f}{L[c]:15.3f}"
        # high motivation should be faster (higher speed / hurry) and clear sooner
        wins[c] += (H[c] < L[c]) if c == "clear_time" else (H[c] > L[c])
    print(row)
n = len(PAIRS)


def sign_test_p(k, n):
    """One-sided probability of k or more successes out of n by chance (p = 0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


print(f"\nPairs where the high-motivation run comes out 'more hurried' (of {n}):")
for c in cols:
    label = {"speed": "measured walking speed", "clear_time": "clears sooner"}.get(c, f"model {c}")
    print(f"  {label:28s} {wins[c]:2d}/{n}   (chance of doing this well by luck: p = {sign_test_p(wins[c], n):.3f})")
