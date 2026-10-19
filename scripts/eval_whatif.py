"""How well does the world model answer "what if?" (see make_whatif.py for the data).

For every held-out moment the model watches the same 2 s, then imagines 10 s under each
gate plan. Compared with the re-simulated truth:
  exit counts   people through each exit within 10 s, per plan
  effect        change caused by the action (plan minus keep), per exit: does the model
                get its direction and size right? Versus ignoring the action entirely.
  planning      pick the plan that gets the most people out within 10 s (or the lowest
                peak crowding): how often is the model's pick the true best, and how many
                people does a wrong pick cost? Versus always keep / random.
  crowd map     people misplaced on the map at 10 s, per plan (vs nothing moves)

Run:  python scripts/eval_whatif.py --run final_real_c
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from crowdwm.data.sim_dataset import collate, make_batch, make_item
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster import Grid, density_layer
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, ACT_AFTER = 10, 50, 5

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real_c")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--split", default="test")
p.add_argument("--save", action="store_true", help="save per-moment results for the planner demo")
a = p.parse_args()

model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
model.arrivals, model.track_outflow = False, True       # truth has no new arrivals either
splatter = TorchSplatter(device="cuda")


def busiest_m2(d):   # (N, H, W) -> max people in any 1 m x 1 m area, per map
    t = torch.as_tensor(d, dtype=torch.float32)[:, None]
    return (F.avg_pool2d(t, 4, stride=1) * 16).flatten(1).amax(1).numpy()


rows, eff_true, eff_ai, map_err, map_persist = [], [], [], [], []
for m in sorted((ROOT / "data/processed/whatif" / a.split).glob("*/info.json")):
    info = json.loads(m.read_text())
    truth = pd.read_parquet(m.parent / "truth.parquet")
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
    ai = torch.stack(model.outflow[CONTEXT - 1:]).sum(0).float().cpu().numpy()[:, :n_ex]   # (P, exits)
    true = np.array(info["exit_counts"], float)                                           # (P, exits)

    grid = Grid(float(s["x0"]), float(s["y0"]))
    end = truth[truth.n == HORIZON - 1]
    true_maps = np.stack([density_layer(grid, end[end.plan == j].x.to_numpy(), end[end.plan == j].y.to_numpy())
                          for j in range(P)])
    ai_maps = pred[:, -1, 0].float().cpu().numpy()
    last = B["density"][0, CONTEXT - 1, 0].cpu().numpy()
    map_err += list(np.abs(ai_maps - true_maps).sum((1, 2)))
    map_persist += list(np.abs(last[None] - true_maps).sum((1, 2)))
    for j in range(1, P):
        eff_true.append(true[j] - true[0])
        eff_ai.append(ai[j] - ai[0])
    rows.append(dict(name=m.parent.name, plans=[q["name"] for q in info["plans"]], true=true, ai=ai,
                     true_peak=busiest_m2(true_maps), ai_peak=busiest_m2(ai_maps)))

eff_true, eff_ai = np.concatenate(eff_true), np.concatenate(eff_ai)
print(f"{len(rows)} held-out moments, {sum(len(r['plans']) for r in rows)} plan outcomes ({a.run})\n")

err_ai = np.mean([np.abs(r["ai"] - r["true"]).mean() for r in rows])
err_ignore = np.mean([np.abs(r["ai"][0][None] - r["true"]).mean() for r in rows])
print("EXIT COUNTS (people through each exit in 10 s)")
print(f"  average miss per exit: model {err_ai:.1f} people | model ignoring the gate action {err_ignore:.1f}")
tot_t = np.concatenate([r["true"].sum(1) for r in rows])
tot_a = np.concatenate([r["ai"].sum(1) for r in rows])
print(f"  people out in 10 s, all exits: truth {tot_t.mean():.1f}, model {tot_a.mean():.1f} on average")

big = np.abs(eff_true) >= 3
print("\nEFFECT OF THE ACTION (plan minus keep, per exit)")
print(f"  exits where the action moves >= 3 people: {big.sum()}")
print(f"  model gets the direction right: {np.mean(np.sign(eff_ai[big]) == np.sign(eff_true[big])):.0%}")
print(f"  correlation of predicted vs true change: {np.corrcoef(eff_ai, eff_true)[0, 1]:.2f}")
print(f"  model's change / true change (size, median over big ones): {np.median(eff_ai[big] / eff_true[big]):.2f}")

rng = np.random.default_rng(0)
print("\nPLANNING (pick one plan per moment)")
for goal, key, better in [("most people out in 10 s", "out", max), ("lowest peak crowding at 10 s", "peak", min)]:
    hits, cost, keep_cost, rand_cost, n = 0, [], [], [], 0
    for r in rows:
        tv = r["true"].sum(1) if key == "out" else r["true_peak"]
        av = r["ai"].sum(1) if key == "out" else r["ai_peak"]
        if np.ptp(tv) < (3 if key == "out" else 0.5):     # plans make no real difference
            continue
        n += 1
        best = int(np.argmax(tv) if better is max else np.argmin(tv))
        pick = int(np.argmax(av) if better is max else np.argmin(av))
        hits += tv[pick] == tv[best]
        sign = 1 if better is max else -1
        cost.append(sign * (tv[best] - tv[pick]))
        keep_cost.append(sign * (tv[best] - tv[0]))
        rand_cost.append(sign * (tv[best] - tv[rng.integers(len(tv))]))
    unit = "people" if key == "out" else "people/m^2"
    print(f"  {goal}: {n} moments where the plan matters")
    print(f"    model picks the true best plan {hits / n:.0%} of the time "
          f"(random pick {np.mean(np.array(rand_cost) == 0):.0%}, always keep {np.mean(np.array(keep_cost) == 0):.0%})")
    print(f"    average cost of its pick vs the best: model {np.mean(cost):.2f} {unit} | "
          f"random {np.mean(rand_cost):.2f} | always keep {np.mean(keep_cost):.2f}")

print("\nCROWD MAP AT 10 S (people misplaced, per plan)")
print(f"  model {np.mean(map_err):.1f} | nothing moves {np.mean(map_persist):.1f} "
      f"(score {np.mean(map_err) / np.mean(map_persist):.2f})")

if a.save:
    out = ROOT / "outputs" / f"whatif_{a.run}.json"
    out.write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                               for r in rows]))
    print("saved", out)
