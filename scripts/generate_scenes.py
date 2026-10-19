"""Stage 1c: generate many random synthetic scenes in parallel and save them.

Each scene is saved to data/processed/sim/<version>/scene_<seed>/:
  traj.parquet    id, t, x, y, target (exit the person is heading to at that moment)
  people.parquet  per person: intended exit, desired speed, group, spawn time (hidden)
  gates.parquet   t, closed_0..closed_k (the actions)
  spec.json       the scene recipe (re-runnable, editable for counterfactuals)
  info.json       run statistics

Run:  python scripts/generate_scenes.py --n 200 --version v1 [--start 0] [--workers 12]
      python scripts/generate_scenes.py --n 4000 --version v2 --family platform
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make(args):
    seed, out_dir, family = args
    from crowdwm.sim.scenes import run_scene, sample_scene

    d = Path(out_dir) / f"scene_{seed:06d}"
    if (d / "info.json").exists():
        try:
            return json.loads((d / "info.json").read_text())   # already done (resumable)
        except json.JSONDecodeError:
            pass   # half-written by an interrupted run: generate the scene again
    spec = sample_scene(np.random.default_rng(seed), seed=seed, family=family)
    out = run_scene(spec)
    d.mkdir(parents=True, exist_ok=True)
    out["traj"].astype({"x": "float32", "y": "float32", "t": "float32"}).to_parquet(d / "traj.parquet")
    out["people"].to_parquet(d / "people.parquet")
    out["gates"].to_parquet(d / "gates.parquet")
    (d / "spec.json").write_text(json.dumps(spec.to_dict()))
    info = dict(out["info"], seed=seed)
    # Written last, via a temp file + rename, so it only exists once the scene is complete
    tmp = d / "info.json.tmp"
    tmp.write_text(json.dumps(info))
    tmp.replace(d / "info.json")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--version", default="v1")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--family", default=None, help="hall, platform or bottleneck (default: mixed)")
    a = p.parse_args()

    out_dir = ROOT / "data" / "processed" / "sim" / a.version
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(out_dir), a.family) for s in range(a.start, a.start + a.n)]
    t0 = time.perf_counter()
    infos = []
    with Pool(a.workers) as pool:
        for i, info in enumerate(pool.imap_unordered(make, jobs)):
            infos.append(info)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} scenes, {time.perf_counter() - t0:.0f} s")

    wall = time.perf_counter() - t0
    people = sum(i["people"] for i in infos)
    stuck = sum(i["still_inside"] for i in infos)
    unspawned = sum(i["never_spawned"] for i in infos)
    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"\n{len(infos)} scenes, {people:,} people, {sum(i['sim_seconds'] for i in infos) / 3600:.1f} h "
          f"of crowd simulated in {wall / 60:.1f} min ({wall / len(infos):.2f} s per scene with "
          f"{a.workers} workers)")
    print(f"stuck at time limit: {stuck} people, never spawned: {unspawned}")
    print(f"disk: {size_mb:.0f} MB in {out_dir} ({size_mb / len(infos):.2f} MB per scene)")


if __name__ == "__main__":
    main()
