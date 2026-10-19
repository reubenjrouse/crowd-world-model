"""Pictures for the dataset card, drawn straight from the released Parquet files.

  whatif_example.gif   one what-if moment: the 2 s a model watches, then the true outcome
                       of each gate plan side by side (people coloured by their hidden
                       intended exit; a closed gate is drawn in red)
  scene_families.png   one hall, one platform and one bottleneck scene

Run:  python scripts/show_dataset.py --moment scene_001319_k225
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
COLOURS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]])
WATCH, HORIZON = 10, 50

p = argparse.ArgumentParser()
p.add_argument("--release", default="data/release/crowd-whatif-v1")
p.add_argument("--moment", default="scene_001319_k225")
p.add_argument("--out", default="data/release/crowd-whatif-v1/assets")
a = p.parse_args()
rel, out = ROOT / a.release, ROOT / a.out
out.mkdir(parents=True, exist_ok=True)
scenes = pd.read_parquet(rel / "scenes.parquet").set_index("scene_id")


def draw_room(ax, sc, closed=()):
    """Floor, walls, exits A/B/C; returns the gate patches (shown when closed)."""
    floor = shapely.from_wkt(sc.walkable_wkt)
    ax.fill(*floor.exterior.xy, color="0.95")
    for hole in floor.interiors:
        ax.fill(*hole.xy, color="0.4")
    ax.plot(*floor.exterior.xy, color="k", lw=1)
    for i, w in enumerate(json.loads(sc.exits_wkt)):
        e = shapely.from_wkt(w)
        ax.fill(*e.exterior.xy, color=COLOURS[i], zorder=3)
        ax.text(e.centroid.x, e.centroid.y, "ABC"[i], color="white", weight="bold", ha="center", va="center",
                fontsize=8, zorder=4)
    gates = []
    for w in json.loads(sc.gates_wkt):
        g = ax.fill(*shapely.from_wkt(w).exterior.xy, color="red", zorder=5, visible=False)[0] if w else None
        gates.append(g)
    minx, miny, maxx, maxy = floor.bounds
    ax.set_xlim(minx - 0.6, maxx + 0.6)
    ax.set_ylim(miny - 0.6, maxy + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return gates


# ---------------------------------------------------------------- what-if GIF
split = "test"
mo = pd.read_parquet(rel / "whatif" / f"{split}.parquet").set_index("moment_id").loc[a.moment]
sc = scenes.loc[mo.scene_id]
plans, counts = json.loads(mo.plans), json.loads(mo.exit_counts)
truth = pd.read_parquet(rel / "whatif_truth" / f"{split}.parquet", filters=[("moment_id", "==", a.moment)])
shard = next(f for f in sorted((rel / "trajectories").glob(f"{split}-*.parquet"))
             if len(pd.read_parquet(f, columns=["scene_id"], filters=[("scene_id", "==", mo.scene_id)])))
traj = pd.read_parquet(shard, filters=[("scene_id", "==", mo.scene_id)])
traj["k"] = (traj.t * 5).round().astype(int)
people = pd.read_parquet(rel / "people" / f"{split}.parquet", filters=[("scene_id", "==", mo.scene_id)])
intended = dict(zip(people.person_id, people.intended))
init_closed = {i for t, i, w in json.loads(sc.gate_events) if t <= mo.t0 and w == "close"} - \
              {i for t, i, w in json.loads(sc.gate_events) if t <= mo.t0 and w == "open"}

P = len(plans)
fig, axes = plt.subplots(1, P, figsize=(4.0 * P, 4.6))
dots, gates = [], []
for j, ax in enumerate(axes):
    gates.append(draw_room(ax, sc))
    dots.append(ax.scatter([], [], s=9, zorder=6))
    flow = ", ".join(f"{'ABC'[i]} {c}" for i, c in enumerate(counts[j]))
    ax.set_title(f"plan: {plans[j]['name']}\nleft within 10 s: {flow}", fontsize=10)
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.9))


def frame_points(j, i):
    if i < WATCH:
        f = traj[traj.k == mo.k0 - WATCH + 1 + i]
    else:
        f = truth[(truth.plan == j) & (truth.n == min(i - WATCH + 1, HORIZON - 1))]
    return f[["x", "y"]].to_numpy(), f.person_id.to_numpy()


def update(i):
    for j in range(P):
        xy, pid = frame_points(j, i)
        dots[j].set_offsets(xy if len(xy) else np.empty((0, 2)))
        dots[j].set_color(COLOURS[[intended.get(q, 0) for q in pid]] if len(pid) else "k")
        closed = set(init_closed)
        if i >= WATCH + 5:                                  # the action happens 1 s after watching
            for _, g, what in plans[j]["events"]:
                closed = closed | {g} if what == "close" else closed - {g}
        for g, patch in enumerate(gates[j]):
            if patch is not None:
                patch.set_visible(g in closed)
    if i < WATCH:
        title.set_text(f"The 2 s a model watches ({i / 5 + 0.2:.1f} s)\ndots = people, coloured by the exit they secretly want")
    else:
        title.set_text(f"What really happens under each gate plan: +{(i - WATCH + 1) / 5:.1f} s\n"
                       "(red bar = closed gate; plans change 1 s after watching)")
    return dots


anim = FuncAnimation(fig, update, frames=WATCH + HORIZON, blit=False)
anim.save(out / "whatif_example.gif", writer=PillowWriter(fps=5), dpi=60)
print("saved", out / "whatif_example.gif")

# ---------------------------------------------------------------- scene families
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, fam in zip(axes, ["hall", "platform", "bottleneck"]):
    cand = scenes[(scenes.family == fam) & (scenes.split == "test")].sort_values("n_people", ascending=False)
    sid = cand.index[len(cand) // 4]
    sc = scenes.loc[sid]
    f = next(f for f in sorted((rel / "trajectories").glob("test-*.parquet"))
             if len(pd.read_parquet(f, columns=["scene_id"], filters=[("scene_id", "==", sid)])))
    t = pd.read_parquet(f, filters=[("scene_id", "==", sid)])
    busiest = t.groupby("t").size().idxmax()
    snap = t[t.t == busiest]
    pp = pd.read_parquet(rel / "people/test.parquet", filters=[("scene_id", "==", sid)])
    want = dict(zip(pp.person_id, pp.intended))
    draw_room(ax, sc)
    ax.scatter(snap.x, snap.y, s=8, c=COLOURS[[want[q] for q in snap.person_id]], zorder=6)
    ax.set_title(f"{fam}: {len(snap)} people at t = {busiest:.0f} s", fontsize=10)
fig.suptitle("Three kinds of scene (people coloured by the exit they want)", fontsize=11)
fig.tight_layout()
fig.savefig(out / "scene_families.png", dpi=80)
print("saved", out / "scene_families.png")
