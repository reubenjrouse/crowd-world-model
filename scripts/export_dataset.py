"""Package the synthetic crowd scenes and what-if pairs as a dataset release (Parquet).

Writes data/release/<name>/:
  scenes.parquet                 one row per scene: split, family, layout (walkable area and
                                 exits as WKT polygons, gates), hidden crowd settings, gate events
  trajectories/<split>-NNN.parquet  scene_id, person_id, t [s], x, y [m], target (exit the person
                                 is currently heading to; hidden), 5 frames per second
  people/<split>.parquet         scene_id, person_id, intended exit, desired speed (hidden),
                                 group, spawn / exit time, exit used
  whatif/<split>.parquet         moment_id, scene_id, k0 (last watched frame), plans (JSON),
                                 people through each exit within 10 s per plan (JSON)
  whatif_truth/<split>.parquet   moment_id, plan, person_id, n (frames after k0), x, y
The data card (README.md) is written separately (docs/DATASET_CARD.md is copied in).

Run:  python scripts/export_dataset.py --name crowd-whatif-v1
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from crowdwm.sim.scenes import SceneSpec, exit_parts, walkable_area

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "data/processed/sim/v3"
WHATIF = ROOT / "data/processed/whatif"
SPLITS = ("train", "val", "test")


def split_of(seed: int) -> str:
    return {18: "val", 19: "test"}.get(seed % 20, "train")


def scene_row(d: Path) -> dict:
    spec = SceneSpec.from_dict(json.loads((d / "spec.json").read_text()))
    parts = [exit_parts(spec, e) for e in spec.exits]
    info = json.loads((d / "info.json").read_text()) if (d / "info.json").exists() else {}
    return dict(scene_id=d.name, split=split_of(spec.seed), family=spec.family,
                n_people=info.get("people"), n_exits=len(spec.exits),
                walkable_wkt=walkable_area(spec).wkt,
                exits_wkt=json.dumps([q["stage"].wkt for q in parts]),
                gates_wkt=json.dumps([q["gate"].wkt if q["gate"] is not None else None for q in parts]),
                gate_events=json.dumps(spec.gate_events),
                hurry=spec.hurry, radius=spec.radius, time_gap=spec.time_gap,
                spec_json=json.dumps(spec.to_dict()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="crowd-whatif-v1")
    p.add_argument("--shard", type=int, default=1500, help="scenes per trajectory file")
    a = p.parse_args()
    out = ROOT / "data/release" / a.name
    for sub in ["trajectories", "people", "whatif", "whatif_truth"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    dirs = [d for d in sorted(SIM.glob("scene_*")) if (d / "traj.parquet").exists()]
    scenes = pd.DataFrame([scene_row(d) for d in dirs])
    scenes.to_parquet(out / "scenes.parquet", index=False)
    print(f"scenes: {len(scenes)} ({scenes.split.value_counts().to_dict()})")

    for split in SPLITS:
        ids = scenes.scene_id[scenes.split == split].tolist()
        people = []
        for i in range(0, len(ids), a.shard):
            traj = []
            for sid in ids[i:i + a.shard]:
                t = pd.read_parquet(SIM / sid / "traj.parquet").rename(columns={"id": "person_id"})
                traj.append(t.assign(scene_id=sid)[["scene_id", "person_id", "t", "x", "y", "target"]])
                pp = pd.read_parquet(SIM / sid / "people.parquet").rename(columns={"id": "person_id"})
                people.append(pp.assign(scene_id=sid))
            pd.concat(traj).to_parquet(out / "trajectories" / f"{split}-{i // a.shard:03d}.parquet", index=False)
        cols = ["scene_id", "person_id", "intended", "speed", "group", "spawn_t", "exit_t", "exited_via"]
        pd.concat(people)[cols].to_parquet(out / "people" / f"{split}.parquet", index=False)
        print(f"  {split}: trajectories and people for {len(ids)} scenes")

        moments, truth = [], []
        for m in sorted((WHATIF / split).glob("*/info.json")):
            info = json.loads(m.read_text())
            mid = m.parent.name
            moments.append(dict(moment_id=mid, scene_id=info["scene"], k0=info["k0"], t0=info["k0"] / 5,
                                n_people=info["people"], plans=json.dumps(info["plans"]),
                                exit_counts=json.dumps(info["exit_counts"])))
            truth.append(pd.read_parquet(m.parent / "truth.parquet").rename(columns={"id": "person_id"})
                         .assign(moment_id=mid)[["moment_id", "plan", "person_id", "n", "x", "y"]])
        if moments:
            pd.DataFrame(moments).to_parquet(out / "whatif" / f"{split}.parquet", index=False)
            pd.concat(truth).to_parquet(out / "whatif_truth" / f"{split}.parquet", index=False)
        print(f"  {split}: {len(moments)} what-if moments")

    card = ROOT / "docs/DATASET_CARD.md"
    if card.exists():
        shutil.copy(card, out / "README.md")
    print("wrote", out)


if __name__ == "__main__":
    main()
