"""What-if demo: the same crowd and the same 2 s of watching, two different gate plans.

Top row: plan A (gates stay as they are). Bottom row: plan B (a gate closes ~1 s after
watching ends). Left: the truth for each plan, re-simulated from the same people with
their true hidden destinations and speeds. Right: the world model's imagined crowd for
each plan, coloured by the exit it thinks people are heading to. Text: people through
each exit within 10 s, truth vs model.

Run:  python scripts/show_whatif.py --run wm_s3 [--scene data/processed/sim/v3/scene_XXXXXX --k0 F]
      (without --scene, a suitable held-out scene is picked automatically)
"""

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from crowdwm.data.sim_dataset import collate, make_batch, make_item, split_of
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster_torch import TorchSplatter
from crowdwm.sim.restart import simulate_from
from crowdwm.sim.scenes import SceneSpec, exit_parts, walkable_area
from crowdwm.viz import move_people

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, CLOSE_AFTER = 10, 50, 5     # gate closes 5 frames (1 s) after watching ends
COLOURS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]])

p = argparse.ArgumentParser()
p.add_argument("--run", default="wm_s3")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--scene", default=None)
p.add_argument("--k0", type=int, default=None, help="last watched frame")
a = p.parse_args()


def find_scene():
    """A held-out scene with a gated exit, a decent crowd heading to it, no gate change and
    no scheduled arrivals in the window."""
    for d in sorted((ROOT / "data/processed/sim/v3").glob("scene_*")):
        if split_of(int(d.name.split("_")[1])) != "test" or not (d / "cache.npz").exists():
            continue
        spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
        gated = [i for i, e in enumerate(spec.exits) if e.gated]
        if not gated or spec.family == "bottleneck":
            continue
        s = np.load(d / "cache.npz")
        counts, ptr = np.diff(s["ptr"]), s["ptr"]
        for k in range(CONTEXT, len(counts) - HORIZON, 5):
            t0 = k / 5
            if counts[k] < 40 or any(t0 - 2 <= t <= t0 + HORIZON / 5 for t, _, _ in spec.gate_events):
                continue
            if s["gate_state"][k][gated[0]]:
                continue
            tgt = s["target"][ptr[k]:ptr[k + 1]]
            share = np.mean(tgt == gated[0])
            if 0.25 <= share <= 0.7:
                return d, k, gated[0]
    raise SystemExit("no suitable scene found")


if a.scene:
    d, k0 = Path(a.scene), a.k0
    spec0 = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
    gate = next(i for i, e in enumerate(spec0.exits) if e.gated)
else:
    d, k0, gate = find_scene()
spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
t0 = k0 / 5
s = dict(np.load(d / "cache.npz"))
traj = pd.read_parquet(d / "traj.parquet")
traj["k"] = np.round(traj["t"] * 5).astype(int)
ppl = pd.read_parquet(d / "people.parquet").set_index("id")
now = traj[traj.k == k0]
people = [(r.id, r.x, r.y, ppl.loc[r.id, "intended"], ppl.loc[r.id, "speed"]) for r in now.itertuples()]

# Truth for both plans: same people, true hidden state, different gate events
plans = {"A: all gates open": [], f"B: gate {'ABC'[gate]} closes at +1 s": [(t0 + CLOSE_AFTER / 5, gate, "close")]}
truth = {}
for name, events in plans.items():
    sp = dataclasses.replace(spec, gate_events=[e for e in spec.gate_events if e[0] <= t0] + events)
    truth[name], _ = simulate_from(sp, t0, people, HORIZON / 5)

def _old_move_people(xy, flux, leave, blocked, x0, y0, cell=0.25, seed=0):
    """Moving-people view of the model's forecast: start from the real positions at the
    last watched frame and move each person with the model's own per-cell move
    probabilities (5x5 neighbourhood), with a little momentum for smooth paths. People
    never enter walls or closed gates and leave at exits with the model's leaving chance.
    Returns a list over imagined frames of (N, 2) positions (NaN = left the scene)."""
    rng = np.random.default_rng(seed)
    k = int(round(np.sqrt(flux[0].shape[0])))
    offs = np.array([(j % k - k // 2, j // k - k // 2) for j in range(k * k)], float) * cell   # (dx, dy)
    pos, vel = xy.astype(float).copy(), np.zeros_like(xy, dtype=float)
    alive = np.ones(len(xy), bool)
    out = []
    for p, lv, blk in zip(flux, leave, blocked):
        col = np.clip(((pos[:, 0] - x0) / cell).astype(int), 0, 63)
        row = np.clip(((pos[:, 1] - y0) / cell).astype(int), 0, 63)
        probs = p[:, row, col].T                                    # (N, k*k)
        probs = probs / probs.sum(1, keepdims=True)
        choice = np.array([rng.choice(k * k, p=q) for q in probs])
        step = offs[choice]
        vel = 0.5 * vel + 0.5 * step
        new = pos + vel + rng.normal(0, 0.02, pos.shape)
        ncol = np.clip(((new[:, 0] - x0) / cell).astype(int), 0, 63)
        nrow = np.clip(((new[:, 1] - y0) / cell).astype(int), 0, 63)
        bad = blk[nrow, ncol]
        new[bad] = pos[bad] + step[bad]          # the sampled move itself always has a clear path
        vel[bad] = step[bad]
        leaving = rng.random(len(pos)) < lv[row, col]
        alive &= ~leaving
        pos = new
        snap = pos.copy()
        snap[~alive] = np.nan
        out.append(snap)
    return out


# Model: same watched frames, different future gate channel
model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
model.arrivals, model.track_outflow, model.record_flux = False, True, True
start, length = k0 - CONTEXT + 1, CONTEXT + HORIZON
lo, hi = s["ptr"][start], s["ptr"][start + CONTEXT]
base = make_item(s, start, length, CONTEXT)
pred, dest, flows, particles = {}, {}, {}, {}
for name, events in plans.items():
    state = np.repeat(s["gate_state"][k0][None], length, 0).copy()
    if events:
        state[CONTEXT - 1 + CLOSE_AFTER:, gate] = 1
    B = make_batch(collate([dict(base, gate_state=state)]), TorchSplatter(device="cuda"), length)
    model.outflow, model.flux, model.leave = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pr, de, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], CONTEXT, sample=False)
    blocked = ((B["static"][0, 0] + B["gates"][0, :, 0]) > 0.5).cpu().numpy()       # (L, H, W)
    particles[name] = move_people(now[["x", "y"]].to_numpy(),
                                  [f[0].cpu().numpy() for f in model.flux[CONTEXT - 1:]],
                                  [lv[0, 0].cpu().numpy() for lv in model.leave[CONTEXT - 1:]],
                                  blocked[CONTEXT:], float(s["x0"]), float(s["y0"]))
    particles[name] = [q if len(q) else np.empty((0, 2)) for q in particles[name]]
    pred[name] = pr[0, :, 0].float().cpu().numpy()
    dest[name] = torch.softmax(de[0].float(), 1).cpu().numpy()
    flows[name] = torch.stack(model.outflow[CONTEXT - 1:]).sum(0)[0].float().cpu().numpy()


def truth_flows(f):
    parts = [exit_parts(spec, e) for e in spec.exits]
    cent = np.array([[q["stage"].centroid.x, q["stage"].centroid.y] for q in parts])
    last = f.sort_values("n").groupby("id").last()
    gone = last[last.n < HORIZON]
    out = np.zeros(len(spec.exits))
    for x, y in gone[["x", "y"]].to_numpy():
        out[np.argmin(np.hypot(cent[:, 0] - x, cent[:, 1] - y))] += 1
    return out


area = walkable_area(spec)
parts = [exit_parts(spec, e) for e in spec.exits]
extent = [s["x0"], s["x0"] + 16, s["y0"], s["y0"] + 16]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
artists = []
for r, name in enumerate(plans):
    tf, mf = truth_flows(truth[name]), flows[name][:len(spec.exits)]
    for c, label in enumerate(["TRUTH (re-simulated)", f"AI ({a.run}), moving-people view"]):
        ax = axes[r, c]
        ax.fill(*area.exterior.xy, color="0.94")
        ax.plot(*area.exterior.xy, color="k", lw=1)
        for hole in area.interiors:
            ax.fill(*hole.xy, color="0.35")
        for i, q in enumerate(parts):
            ax.fill(*q["stage"].exterior.xy, color=COLOURS[i])
            ax.text(q["stage"].centroid.x, q["stage"].centroid.y, "ABC"[i], weight="bold", ha="center",
                    va="center", color="white")
        flow = tf if c == 0 else mf
        ax.set_title(f"{name}\n{label}: through " + ", ".join(f"{'ABC'[i]} {flow[i]:.0f}" for i in range(len(flow))),
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        minx, miny, maxx, maxy = area.bounds
        ax.set_xlim(minx - 0.5, maxx + 0.5)
        ax.set_ylim(miny - 0.5, maxy + 0.5)
        gp = ax.fill(*parts[gate]["gate"].exterior.xy, color="red", visible=False, zorder=6)[0]
        if c == 0:
            art = ax.scatter([], [], s=10, c="tab:blue", zorder=5)
        else:
            art = ax.scatter([], [], s=10, zorder=5)
        artists.append((name, c, art, gp))
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.95))
true_tgt = dict(zip(now["id"], now["target"]))


def update(i):
    n = i - CONTEXT + 1                       # frames after the last watched frame
    out = [title]
    for name, c, art, gp in artists:
        closed = bool(plans[name]) and n >= CLOSE_AFTER
        gp.set_visible(closed)
        if c == 0:
            if n <= 0:
                pts = traj[traj.k == k0 + n]
            else:
                f = truth[name]
                pts = f[f.n == n]
            art.set_offsets(pts[["x", "y"]].to_numpy() if len(pts) else np.empty((0, 2)))
        else:
            pts = (traj[traj.k == k0 + n][["x", "y"]].to_numpy() if n <= 0
                   else particles[name][n - 1])
            pts = pts[~np.isnan(pts).any(1)] if len(pts) else np.empty((0, 2))
            art.set_offsets(pts if len(pts) else np.empty((0, 2)))
            if len(pts):   # colour = the model's destination guess where each person stands
                pr = dest[name][min(i, length - 1)][:len(spec.exits)]
                col = np.clip(((pts[:, 0] - s["x0"]) / 0.25).astype(int), 0, 63)
                row = np.clip(((pts[:, 1] - s["y0"]) / 0.25).astype(int), 0, 63)
                art.set_color(COLOURS[pr[:, row, col].argmax(0)])
        out += [art, gp]
    phase = "watching" if n <= 0 else f"imagining +{n / 5:.1f} s"
    title.set_text(f"What if? Same crowd, same 2 s of watching, two gate plans ({phase})   "
                   f"colour = exit the AI thinks people head to")
    return out


anim = FuncAnimation(fig, update, frames=length, blit=False)
out = ROOT / "outputs/figures" / f"whatif_{a.run}_{d.name}.gif"
anim.save(out, writer=PillowWriter(fps=5), dpi=65)
print(f"saved {out}  (scene {d.name}, k0 {k0}, gate {'ABC'[gate]})")
for name in plans:
    print(f"  {name}: truth through exits {truth_flows(truth[name]).astype(int).tolist()}, "
          f"AI {np.round(flows[name][:len(spec.exits)], 1).tolist()}")
