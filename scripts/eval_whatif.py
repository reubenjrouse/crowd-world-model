"""How well does the world model answer "what if?" (see make_whatif.py for the data).

For every held-out moment the model watches the same 2 s, then imagines 10 s under each
gate plan. Compared with the re-simulated truth (report: crowdwm.whatif.summarize):
  exit counts   people through each exit within 10 s, per plan
  effect        change caused by the action (plan minus keep), per exit
  planning      pick the plan with the lowest peak crowding at 10 s
  crowd map     people misplaced on the map at 10 s (vs nothing moves)
Simulator baselines with guessed destinations: scripts/baseline_whatif.py.

Run:  python scripts/eval_whatif.py --run final_real_c --save
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crowdwm.data.sim_dataset import collate, make_batch, make_item
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster import Grid, density_layer
from crowdwm.raster_torch import TorchSplatter
from crowdwm.whatif import busiest_m2, summarize

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, ACT_AFTER = 10, 50, 5

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real_c")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--split", default="test")
p.add_argument("--save", action="store_true", help="save per-moment results (used by show_planner.py)")
a = p.parse_args()

model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
model.arrivals, model.track_outflow = False, True       # truth has no new arrivals either
splatter = TorchSplatter(device="cuda")


def true_maps_at_end(truth: pd.DataFrame, grid: Grid, n_plans: int) -> np.ndarray:
    end = truth[truth.n == HORIZON - 1]
    return np.stack([density_layer(grid, end[end.plan == j].x.to_numpy(), end[end.plan == j].y.to_numpy())
                     for j in range(n_plans)])


rows = []
for m in sorted((ROOT / "data/processed/whatif" / a.split).glob("*/info.json")):
    info = json.loads(m.read_text())
    s = dict(np.load(ROOT / "data/processed/sim/v3" / info["scene"] / "cache.npz"))
    k0, n_ex, P = info["k0"], len(info["exit_counts"][0]), len(info["plans"])
    start, length = k0 - CONTEXT + 1, CONTEXT + HORIZON
    base = make_item(s, start, length, CONTEXT)
    items = []
    for plan in info["plans"]:
        state = np.repeat(s["gate_state"][k0][None], length, 0).copy()
        for _, i, what in plan["events"]:
            state[CONTEXT - 1 + ACT_AFTER:, i] = 1 if what == "close" else 0
        items.append(dict(base, gate_state=state))
    B = make_batch(collate(items), splatter, length)
    model.outflow = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred, _, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT, sample=False)
    ai = torch.stack(model.outflow[CONTEXT - 1:]).sum(0).float().cpu().numpy()[:, :n_ex]
    true_maps = true_maps_at_end(pd.read_parquet(m.parent / "truth.parquet"), Grid(float(s["x0"]), float(s["y0"])), P)
    ai_maps = pred[:, -1, 0].float().cpu().numpy()
    last = B["density"][0, CONTEXT - 1, 0].cpu().numpy()
    rows.append(dict(name=m.parent.name, plans=[q["name"] for q in info["plans"]],
                     true=np.array(info["exit_counts"], float), ai=ai,
                     true_peak=busiest_m2(true_maps), ai_peak=busiest_m2(ai_maps),
                     map_err=np.abs(ai_maps - true_maps).sum((1, 2)),
                     map_persist=np.abs(last[None] - true_maps).sum((1, 2))))

summarize(rows, f"world model ({a.run})")
if a.save:
    out = ROOT / "outputs" / f"whatif_{a.run}.json"
    out.write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                               for r in rows]))
    print("saved", out)
