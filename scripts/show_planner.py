"""Planner demo: the AI imagines every gate plan and recommends the safest one.

One held-out what-if moment (make_whatif.py). Top row: the AI's imagined crowd under each
plan (crowding heat map, people per m^2) after watching the same 2 s. Bottom row: the
truth for each plan, re-simulated with everyone's real destination and speed. The AI
recommends the plan with the lowest peak crowding (busiest 1 m^2 after 10 s); the
title says whether that matches the truly best plan.

Without --moment, picks the held-out moment where the choice matters most (largest true
difference in peak crowding between plans, 3 plans), among those the AI gets right
(it picks the true best in ~80% of moments where the plan matters; eval_whatif.py).

Run:  python scripts/eval_whatif.py --run final_real_c --save
      python scripts/show_planner.py --run final_real_c
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter

from crowdwm.data.sim_dataset import collate, make_batch, make_item
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster import Grid, density_layer
from crowdwm.raster_torch import TorchSplatter
from crowdwm.sim.scenes import SceneSpec, exit_parts, walkable_area

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, ACT_AFTER = 10, 50, 5
COLOURS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]])
CMAP = plt.get_cmap("Reds").copy()
CMAP.set_bad(alpha=0)                   # empty floor stays visible

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real_c")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--moment", default=None, help="folder name under data/processed/whatif/test")
a = p.parse_args()

if a.moment is None:
    rows = json.loads((ROOT / "outputs" / f"whatif_{a.run}.json").read_text())
    good = [r for r in rows if len(r["plans"]) == 3 and np.argmin(r["ai_peak"]) == np.argmin(r["true_peak"])]
    a.moment = max(good, key=lambda r: np.ptp(r["true_peak"]))["name"]
m = ROOT / "data/processed/whatif/test" / a.moment
info = json.loads((m / "info.json").read_text())
truth = pd.read_parquet(m / "truth.parquet")
d = ROOT / "data/processed/sim/v3" / info["scene"]
spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
s = dict(np.load(d / "cache.npz"))
k0, plans = info["k0"], info["plans"]
P, start, length = len(plans), k0 - CONTEXT + 1, CONTEXT + HORIZON

# AI: same watched frames, one imagined future per plan
model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
model.arrivals = False                  # the truth has no new arrivals either
base = make_item(s, start, length, CONTEXT)
items, states = [], []
for plan in plans:
    state = np.repeat(s["gate_state"][k0][None], length, 0).copy()
    for _, i, what in plan["events"]:
        state[CONTEXT - 1 + ACT_AFTER:, i] = 1 if what == "close" else 0
    items.append(dict(base, gate_state=state))
    states.append(state)
B = make_batch(collate(items), TorchSplatter(device="cuda"), length)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred, _, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT, sample=False)
watched = B["density"][0, :CONTEXT, 0].cpu().numpy()
ai = np.concatenate([np.repeat(watched[None], P, 0), pred[:, CONTEXT - 1:, 0].float().cpu().numpy()], 1)

# Truth: watched frames from the scene, then each plan's re-simulation
grid = Grid(float(s["x0"]), float(s["y0"]))
true_pts = [[s["xy"][s["ptr"][start + i]:s["ptr"][start + i + 1]] for i in range(CONTEXT)] for _ in range(P)]
for j in range(P):
    tj = truth[truth.plan == j]
    for n in range(1, HORIZON + 1):
        f = tj[tj.n == min(n, HORIZON - 1)]
        true_pts[j].append(f[["x", "y"]].to_numpy())
true_maps = np.array([[density_layer(grid, q[:, 0], q[:, 1]) if len(q) else np.zeros((64, 64)) for q in pts]
                      for pts in true_pts])


def busiest(maps):   # (..., H, W) -> people in the busiest 1 m^2
    t = torch.as_tensor(maps, dtype=torch.float32)
    sh = t.shape
    return (F.avg_pool2d(t.reshape(-1, 1, *sh[-2:]), 4, stride=1) * 16).reshape(*sh[:-2], -1).amax(-1).numpy()


ai_peak = busiest(ai[:, -1])            # crowding at the end of the 10 s (as in eval_whatif.py)
true_peak = busiest(true_maps[:, -1])
pick, best = int(np.argmin(ai_peak)), int(np.argmin(true_peak))

area = walkable_area(spec)
parts = [exit_parts(spec, e) for e in spec.exits]
extent = [s["x0"], s["x0"] + 16, s["y0"], s["y0"] + 16]
fig, axes = plt.subplots(2, P, figsize=(4.4 * P, 8.6), constrained_layout=True)
heat, dots, gate_art = [], [], []
for r in range(2):
    for j in range(P):
        ax = axes[r, j]
        ax.fill(*area.exterior.xy, color="0.95")
        for hole in area.interiors:
            ax.fill(*hole.xy, color="0.35")
        ax.plot(*area.exterior.xy, color="k", lw=1, zorder=4)
        for i, q in enumerate(parts):
            ax.fill(*q["stage"].exterior.xy, color=COLOURS[i], zorder=6)
            ax.text(q["stage"].centroid.x, q["stage"].centroid.y, "ABC"[i], weight="bold", ha="center",
                    va="center", color="white", zorder=7)
        heat.append(ax.imshow(np.ma.masked_all((64, 64)), origin="lower", extent=extent, cmap=CMAP, vmin=0,
                              vmax=4, alpha=0.9, zorder=3))
        dots.append(ax.scatter([], [], s=6, c="k", zorder=5) if r == 1 else None)
        gate_art.append([ax.fill(*q["gate"].exterior.xy, color="red", zorder=6, visible=False)[0]
                         if q["gate"] is not None else None for q in parts])
        minx, miny, maxx, maxy = area.bounds
        ax.set_xlim(minx - 0.6, maxx + 0.6)
        ax.set_ylim(miny - 0.6, maxy + 0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            star = "  <- AI's pick" if j == pick else ""
            ax.set_title(f"plan: {plans[j]['name']}{star}\nAI imagines: peak {ai_peak[j]:.1f} people/m² at 10 s",
                         fontsize=10, color="tab:red" if j == pick else "k")
        else:
            star = "  <- truly best" if j == best else ""
            ax.set_title(f"TRUTH: peak {true_peak[j]:.1f} people/m² at 10 s{star}", fontsize=10)
verdict = "matches the truly best plan" if pick == best else f"truly best was '{plans[best]['name']}'"
title = fig.suptitle("")
fig.colorbar(heat[0], ax=axes, shrink=0.5, label="people per m²")


def update(i):
    for r in range(2):
        for j in range(P):
            k = r * P + j
            heat[k].set_data(np.ma.masked_less((ai[j, i] if r == 0 else true_maps[j, i]) * 16, 0.15))
            if r == 1:
                q = true_pts[j][i]
                dots[k].set_offsets(q if len(q) else np.empty((0, 2)))
            for g, art in enumerate(gate_art[k]):
                if art is not None:
                    art.set_visible(bool(states[j][i, g]))
    phase = "WATCHING the real crowd (2 s)" if i < CONTEXT else f"IMAGINING each plan: +{(i - CONTEXT + 1) / 5:.1f} s"
    title.set_text(f"{phase}\nAI recommends '{plans[pick]['name']}' (lowest crowding): {verdict}")
    return heat + [x for x in dots if x is not None]


anim = FuncAnimation(fig, update, frames=length, blit=False)
out = ROOT / "outputs/figures" / f"planner_{a.run}_{a.moment}.gif"
anim.save(out, writer=PillowWriter(fps=5), dpi=60)
print("saved", out, "| plans", [q["name"] for q in plans], "| AI peaks", ai_peak.round(1),
      "| true peaks", true_peak.round(1))
