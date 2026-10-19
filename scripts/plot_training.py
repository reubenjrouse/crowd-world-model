"""Plot a training run's log (blog-ready figures).

Saves outputs/runs/<name>/training_curves.png with four panels:
  1. training losses (smoothed)
  2. held-out crowd error at 1 s and 4 s ahead vs the "nothing moves" baseline
  3. destination accuracy on held-out scenes
  4. hurry error on held-out scenes

Run:  python scripts/plot_training.py wm_v1 [other_run ...]   (several runs are overlaid)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    rows = [json.loads(line) for line in open(ROOT / "outputs" / "runs" / name / "log.jsonl")]
    df = pd.DataFrame(rows)
    if "kind" not in df:          # older logs: one combined row per evaluation
        df["kind"] = "val"
        df = df.rename(columns={c: c[4:] for c in df.columns if c.startswith("val_")})
    return df[df.kind == "train"], df[df.kind == "val"]


names = sys.argv[1:] or ["wm_v1"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for name in names:
    tr, va = load(name)
    label = name if len(names) > 1 else None
    ax = axes[0, 0]
    for k, style in [("rollout", "-"), ("teacher", "--"), ("dest", ":")]:
        if len(tr) and k in tr:
            ax.plot(tr.step, tr[k].rolling(20, min_periods=1).mean(), style,
                    label=f"{name} {k}" if label else k)
    ax = axes[0, 1]
    for k, style in [("mae_1s", "-"), ("mae_4s", "-")]:
        line, = ax.plot(va.step, va[k], style, marker="o", ms=3,
                        label=f"{name} AI, {k[4:]} ahead" if label else f"AI, {k[4:]} ahead")
        ax.plot(va.step, va[k.replace("mae", "persist")], ":", color=line.get_color(),
                label=f'"nothing moves", {k[4:]} ahead' if not label else None)
    axes[1, 0].plot(va.step, 100 * va["dest_acc"], marker="o", ms=3, label=label)
    axes[1, 1].plot(va.step, va["hurry_mae"], marker="o", ms=3, label=label)

axes[0, 0].set(title="Training losses (smoothed)", xlabel="step", yscale="log")
axes[0, 1].set(title="Held-out crowd error (lower is better)", xlabel="step",
               ylabel="people misplaced per frame")
axes[1, 0].set(title="Held-out destination accuracy", xlabel="step", ylabel="% of people")
axes[1, 1].set(title="Held-out hurry error", xlabel="step", ylabel="m/s")
for ax in axes.flat:
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
fig.suptitle("Crowd world model training: " + ", ".join(names))
fig.tight_layout()
out = ROOT / "outputs" / "runs" / names[0] / "training_curves.png"
fig.savefig(out, dpi=130)
print("saved", out)
