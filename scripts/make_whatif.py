"""What-if dataset: the same crowd moment, several gate plans, the true outcome of each.

For one or more moments per synthetic scene (>= `--min_people` people, >= 2 exits, a gated exit,
no gate change in the 2 s before or the 10 s after), the simulator is restarted from
everyone's exact position with their true hidden destination and speed, once per plan:
  keep        gates stay as they are
  close X     gated exit X closes 1 s after the moment (one plan per open gated exit)
  open X      closed gated exit X opens 1 s after the moment
Only people present at the moment are simulated (no new arrivals), so plans differ only
by the gate action.

Each moment becomes data/processed/whatif/<split>/<scene>_k<frame>/ with
  info.json     scene, frame k0 (last watched frame), plans [{name, events}], people left
                through each exit within 10 s per plan, simulator wall time
  truth.parquet plan, id, n (frames after k0 at 5 fps), x, y

Run:  python scripts/make_whatif.py --split test --per_scene 4 --workers 10
"""

import argparse
import dataclasses
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from crowdwm.sim.restart import simulate_from
from crowdwm.sim.scenes import SceneSpec, exit_parts

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, HORIZON, ACT_AFTER = 10, 50, 5        # watch 2 s, imagine 10 s, act 1 s after watching
SIM_DIR = ROOT / "data/processed/sim/v3"
OUT_DIR = ROOT / "data/processed/whatif"


def split_of(seed: int) -> str:   # same rule as crowdwm.data.sim_dataset (which loads torch)
    return {18: "val", 19: "test"}.get(seed % 20, "train")


def pick_moments(d: Path, rng: np.random.Generator, min_people: int, per_scene: int = 1):
    """[(spec, k0, plans)] for up to `per_scene` random suitable moments of this scene,
    at least 5 s apart."""
    spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
    gated = [i for i, e in enumerate(spec.exits) if e.gated]
    if len(spec.exits) < 2 or not gated or not (d / "cache.npz").exists():
        return []
    s = np.load(d / "cache.npz")
    counts = np.diff(s["ptr"])
    ok = [k for k in range(CONTEXT, len(counts) - HORIZON)
          if counts[k] >= min_people
          and not any(k / 5 - 2 <= t <= (k + HORIZON) / 5 for t, _, _ in spec.gate_events)]
    picked = []
    for k0 in rng.permutation(ok):
        if len(picked) == per_scene:
            break
        if all(abs(int(k0) - k) >= 25 for k in picked):
            picked.append(int(k0))
    out = []
    for k0 in picked:
        t_act = (k0 + ACT_AFTER) / 5
        closed = s["gate_state"][k0]
        plans = [{"name": "keep", "events": []}]
        for i in gated:
            what = "open" if closed[i] else "close"
            plans.append({"name": f"{what} {'ABC'[i]}", "events": [[t_act, i, what]]})
        out.append((spec, k0, plans))
    return out


def exit_counts(spec: SceneSpec, f: pd.DataFrame) -> list:
    """People who left within the horizon, by the exit nearest their last position."""
    cent = np.array([[p["stage"].centroid.x, p["stage"].centroid.y]
                     for p in (exit_parts(spec, e) for e in spec.exits)])
    last = f.sort_values("n").groupby("id").last()
    gone = last[last.n < HORIZON]
    out = np.zeros(len(spec.exits), int)
    for x, y in gone[["x", "y"]].to_numpy():
        out[np.argmin(np.hypot(cent[:, 0] - x, cent[:, 1] - y))] += 1
    return out.tolist()


def make(job):
    d, split, seed, min_people, per_scene = job
    rng = np.random.default_rng(seed)
    return sum(make_one(d, split, seed, spec, k0, plans)
               for spec, k0, plans in pick_moments(d, rng, min_people, per_scene))


def make_one(d, split, seed, spec, k0, plans) -> int:
    out = OUT_DIR / split / f"{d.name}_k{k0}"
    if (out / "info.json").exists():
        return 1
    t0 = k0 / 5
    traj = pd.read_parquet(d / "traj.parquet")
    now = traj[np.round(traj["t"] * 5).astype(int) == k0]
    ppl = pd.read_parquet(d / "people.parquet").set_index("id")
    people = [(r.id, r.x, r.y, ppl.loc[r.id, "intended"], ppl.loc[r.id, "speed"]) for r in now.itertuples()]
    frames, flows, wall = [], [], 0.0
    for j, plan in enumerate(plans):
        events = [e for e in spec.gate_events if e[0] <= t0] + [tuple(e) for e in plan["events"]]
        f, w = simulate_from(dataclasses.replace(spec, gate_events=events), t0, people, HORIZON / 5, seed=seed)
        frames.append(f.assign(plan=j))
        flows.append(exit_counts(spec, f))
        wall += w
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames)[["plan", "id", "n", "x", "y"]].to_parquet(out / "truth.parquet")
    info = dict(scene=d.name, k0=k0, plans=plans, exit_counts=flows, people=len(people), sim_seconds=round(wall, 2))
    tmp = out / "info.json.tmp"
    tmp.write_text(json.dumps(info))
    tmp.replace(out / "info.json")            # written last: a folder with info.json is complete
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", help="test (evaluation) or train (fine-tuning)")
    p.add_argument("--n", type=int, default=None, help="max scenes to try")
    p.add_argument("--min_people", type=int, default=20)
    p.add_argument("--per_scene", type=int, default=1, help="moments per scene (at least 5 s apart)")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    dirs = [d for d in sorted(SIM_DIR.glob("scene_*")) if split_of(int(d.name.split("_")[1])) == a.split]
    if a.n:
        dirs = dirs[:a.n]
    jobs = [(d, a.split, int(d.name.split("_")[1]), a.min_people, a.per_scene) for d in dirs]
    done = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(make, jobs)):
            done += r
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} scenes tried, {done} what-if moments", flush=True)
    print(f"done: {done} what-if moments in {OUT_DIR / a.split}")


if __name__ == "__main__":
    main()
