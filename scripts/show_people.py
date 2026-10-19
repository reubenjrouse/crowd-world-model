"""Reality vs the model's most likely future, both drawn as moving people.

Left: real people. Right: people moved by the model's predicted flow (see crowdwm.viz),
coloured by the exit the model thinks each is heading to. Bottom: people in view over
time, real vs model. Works for any cached scene (synthetic or real).
Saves outputs/figures/people_<run>_<scene>.gif

Run:  python scripts/show_people.py --run final_real --dir data/processed/real/juelich/caserne_bottleneck_2006__b120
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from crowdwm.data.sim_dataset import collate, make_batch, make_item, unpack_arrivals
from crowdwm.model.world_model import load_checkpoint
from crowdwm.raster_torch import TorchSplatter
from crowdwm.viz import move_people

ROOT = Path(__file__).resolve().parents[1]
COLOURS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]])

p = argparse.ArgumentParser()
p.add_argument("--run", default="final_real")
p.add_argument("--ckpt", default="last.pt")
p.add_argument("--dir", required=True)
p.add_argument("--start", default="auto")
p.add_argument("--context", type=int, default=10)
p.add_argument("--horizon", type=int, default=50)
p.add_argument("--title", default=None)
a = p.parse_args()

d = Path(a.dir)
s = dict(np.load(d / "cache.npz"))
length = a.context + a.horizon
n_frames = len(s["ptr"]) - 1
if a.start == "auto":   # just before a train unloads if there is one, else a busy moment
    arr = np.flatnonzero(unpack_arrivals(s, 0, n_frames).reshape(n_frames, -1).any(1))
    counts = np.diff(s["ptr"])
    k0 = int(arr[0] - a.context) if len(arr) else int(np.argmax(counts[:max(1, n_frames - length)]))
else:
    k0 = int(a.start)
k0 = int(np.clip(k0, 0, n_frames - length - 1))

model = load_checkpoint(ROOT / "outputs/runs" / a.run / a.ckpt)
model.record_flux = True
B = make_batch(collate([make_item(s, k0, length, a.context)]), TorchSplatter(device="cuda"), length)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred, dest, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], a.context,
                          arrivals=B["arrivals"], sample=False)
dest = torch.softmax(dest[0].float(), 1).cpu().numpy()
blocked = ((B["static"][0, 0] + B["gates"][0, :, 0]) > 0.5).cpu().numpy()
c0 = a.context - 1
last = s["xy"][s["ptr"][k0 + c0]:s["ptr"][k0 + c0 + 1]]
ai = move_people(last, [f[0].cpu().numpy() for f in model.flux[c0:]],
                 [lv[0, 0].cpu().numpy() for lv in model.leave[c0:]], blocked[a.context:],
                 float(s["x0"]), float(s["y0"]),
                 source=[src[0, 0].cpu().numpy() for src in model.source[c0:]] if model.source else None)
pred_count = np.concatenate([B["density"][0, :a.context, 0].sum((-1, -2)).cpu().numpy(),
                             pred[0, c0:, 0].float().sum((-1, -2)).cpu().numpy()])
real_count = B["density"][0, :, 0].sum((-1, -2)).cpu().numpy()

extent = [s["x0"], s["x0"] + 16, s["y0"], s["y0"] + 16]
walls = s["static"][0].astype(float)
n_ex = int(s["n_exits"])
exits = np.zeros((64, 64, 4))
for i in range(n_ex):
    exits[s["static"][1 + i] > 0] = [*COLOURS[i], 0.55]
arrivals = B["arrivals"][0, :, 0].cpu().numpy()

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
dots = []
for ax, t in zip(axes, ["REALITY", f"AI ({a.run}): most likely future, drawn as people"]):
    ax.imshow(np.ones((64, 64, 3)) * 0.96, origin="lower", extent=extent)
    w = np.zeros((64, 64, 4))
    w[..., :3] = 0.35
    w[..., 3] = walls * 0.9
    ax.imshow(w, origin="lower", extent=extent, zorder=2)
    ax.imshow(exits, origin="lower", extent=extent, zorder=2)
    ax.set_title(t, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    dots.append(ax.scatter([], [], s=9, zorder=5))
arr_imgs = [ax.imshow(np.zeros((64, 64, 4)), origin="lower", extent=extent, zorder=3) for ax in axes]
axc = fig.add_subplot(gs[1, :])
tt = np.arange(length) / 5
axc.plot(tt, real_count, "k-", lw=2, label="real")
axc.plot(tt, pred_count, color="tab:red", lw=2, label="AI (most likely future)")
axc.axvspan(0, c0 / 5, color="0.9")
on = np.flatnonzero(arrivals[a.context:].reshape(a.horizon, -1).any(1))
if len(on):
    axc.axvspan((a.context + on[0]) / 5, (a.context + on[-1]) / 5, color="tab:green", alpha=0.12,
                label="scheduled arrivals (known)")
cursor = axc.axvline(0, color="tab:blue")
axc.set(xlabel="seconds", ylabel="people in view")
axc.legend(loc="upper left", fontsize=8, ncol=3)
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.95))


def colour_for(pts, i):
    if not len(pts):
        return "k"
    pr = dest[min(i, length - 1)][:n_ex]
    col = np.clip(((pts[:, 0] - s["x0"]) / 0.25).astype(int), 0, 63)
    row = np.clip(((pts[:, 1] - s["y0"]) / 0.25).astype(int), 0, 63)
    return COLOURS[pr[:, row, col].argmax(0)]


def update(i):
    real = s["xy"][s["ptr"][k0 + i]:s["ptr"][k0 + i + 1]]
    dots[0].set_offsets(real if len(real) else np.empty((0, 2)))
    dots[0].set_color("tab:blue" if n_ex == 1 else "0.2")
    pts = real if i <= c0 else ai[i - c0 - 1]
    dots[1].set_offsets(pts if len(pts) else np.empty((0, 2)))
    dots[1].set_color(colour_for(pts, i))
    g = np.zeros((64, 64, 4))
    g[..., 1] = 0.6
    g[..., 3] = arrivals[i] * 0.3
    for im in arr_imgs:
        im.set_data(g)
    cursor.set_xdata([i / 5, i / 5])
    phase = "WATCHING (real frames go in)" if i <= c0 else f"IMAGINING +{(i - c0) / 5:.1f} s"
    title.set_text(f"{a.title or d.name}   {phase}   (AI dot colour = exit it thinks they head to)")
    return [*dots, *arr_imgs, cursor, title]


anim = FuncAnimation(fig, update, frames=length, blit=False)
out = ROOT / "outputs/figures" / f"people_{a.run}_{d.name}.gif"
anim.save(out, writer=PillowWriter(fps=5), dpi=65)
print("saved", out, "| start frame", k0, "| real people at end", int(round(real_count[-1])),
      "| AI expects", int(round(pred_count[-1])))
