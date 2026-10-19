"""Simulator baselines for "what if?": run the crowd simulator on what can be observed.

The truth (make_whatif.py) is the simulator with everyone's real destination and speed,
which a camera can't see. These baselines restart the same simulator from the positions
at the last watched frame, but with guessed hidden state, and the calibrated body
settings for an average crowd (the real ones depend on the hidden hurry):
  nearest   everyone heads for the nearest exit at the average desired speed (1.2 m/s)
  heading   everyone heads for the exit they are walking towards over the last 1 s, at
            their observed speed (people barely moving, e.g. queueing, get the nearest
            exit and the average speed: a queue doesn't reveal how fast they'd walk)
Scored with the same report as the world model (crowdwm.whatif.summarize).

Run:  python scripts/baseline_whatif.py --workers 10
"""

import argparse
import dataclasses
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from crowdwm.raster import Grid, density_layer
from crowdwm.sim.restart import simulate_from
from crowdwm.sim.scenes import CONFIG, SceneSpec, exit_parts

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, ACT_AFTER = 10, 50, 5
MEAN_SPEED = 1.2                  # middle of the hurry range used to make the scenes (0.8-1.6)


def guess_people(s, k0, centres, how):
    """[(id, x, y, exit, speed)] from what a camera sees at frame k0 (and 1 s before)."""
    lo, hi = s["ptr"][k0], s["ptr"][k0 + 1]
    xy, pid = s["xy"][lo:hi].astype(float), s["pid"][lo:hi]
    plo, phi = s["ptr"][k0 - 5], s["ptr"][k0 - 4]
    prev = dict(zip(s["pid"][plo:phi], s["xy"][plo:phi].astype(float)))
    out = []
    for q, p in zip(pid, xy):
        to_exit = centres - p
        nearest = int(np.argmin(np.hypot(*to_exit.T)))
        ex, v0 = nearest, MEAN_SPEED
        if how == "heading" and q in prev:
            v = p - prev[q]                                   # metres moved in the last 1 s
            speed = float(np.hypot(*v))
            if speed > 0.3:
                cos = to_exit @ v / (np.hypot(*to_exit.T) * speed + 1e-9)
                ex, v0 = int(np.argmax(cos)), float(np.clip(speed, 0.5, 1.8))
        out.append((int(q), float(p[0]), float(p[1]), ex, v0))
    return out


def run(job):
    m, how = job
    info = json.loads((m / "info.json").read_text())
    d = ROOT / "data/processed/sim/v3" / info["scene"]
    spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
    body = yaml.safe_load(CONFIG.read_text())
    spec = dataclasses.replace(spec, radius=body["radius"], time_gap=body["time_gap"])   # average crowd
    s = dict(np.load(d / "cache.npz"))
    k0, t0 = info["k0"], info["k0"] / 5
    parts = [exit_parts(spec, e) for e in spec.exits]
    centres = np.array([[q["stage"].centroid.x, q["stage"].centroid.y] for q in parts])
    people = guess_people(s, k0, centres, how)
    grid = Grid(float(s["x0"]), float(s["y0"]))
    counts, maps = [], []
    for plan in info["plans"]:
        events = [e for e in spec.gate_events if e[0] <= t0] + [tuple(e) for e in plan["events"]]
        f, _ = simulate_from(dataclasses.replace(spec, gate_events=events), t0, people, HORIZON / 5)
        last = f.sort_values("n").groupby("id").last()
        gone = last[last.n < HORIZON]
        c = np.zeros(len(spec.exits))
        for x, y in gone[["x", "y"]].to_numpy():
            c[np.argmin(np.hypot(centres[:, 0] - x, centres[:, 1] - y))] += 1
        counts.append(c)
        end = f[f.n == HORIZON - 1]
        maps.append(density_layer(grid, end.x.to_numpy(), end.y.to_numpy()))
    lo, hi = s["ptr"][k0], s["ptr"][k0 + 1]
    watched = density_layer(grid, s["xy"][lo:hi, 0], s["xy"][lo:hi, 1])
    tr = __import__("pandas").read_parquet(m / "truth.parquet")
    tr = tr[tr.n == HORIZON - 1]
    true_maps = [density_layer(grid, tr[tr.plan == j].x.to_numpy(), tr[tr.plan == j].y.to_numpy())
                 for j in range(len(info["plans"]))]
    return dict(name=m.name, true=info["exit_counts"], ai=np.array(counts).tolist(),
                maps=np.array(maps), true_maps=np.array(true_maps), watched=watched)


def main():
    from crowdwm.whatif import busiest_m2, summarize   # loads torch: main process only

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max", type=int, default=None)
    a = p.parse_args()
    moments = sorted(m.parent for m in (ROOT / "data/processed/whatif" / a.split).glob("*/info.json"))[:a.max]
    for how, label in [("nearest", "simulator, everyone to the nearest exit"),
                       ("heading", "simulator, exit and speed guessed from heading")]:
        with Pool(a.workers) as pool:
            res = pool.map(run, [(m, how) for m in moments], chunksize=4)
        rows = [dict(name=r["name"], true=r["true"], ai=r["ai"],
                     true_peak=busiest_m2(r["true_maps"]), ai_peak=busiest_m2(r["maps"]),
                     map_err=np.abs(r["maps"] - r["true_maps"]).sum((1, 2)),
                     map_persist=np.abs(r["watched"][None] - r["true_maps"]).sum((1, 2))) for r in res]
        summarize(rows, label)


if __name__ == "__main__":
    main()
