---
license: cc-by-4.0
pretty_name: Crowd What-If
tags:
- crowd-simulation
- pedestrian-dynamics
- world-model
- counterfactual
- synthetic
task_categories:
- time-series-forecasting
size_categories:
- 100M<n<1B
configs:
- config_name: scenes
  data_files: scenes.parquet
- config_name: trajectories
  data_files:
  - split: train
    path: trajectories/train-*.parquet
  - split: validation
    path: trajectories/val-*.parquet
  - split: test
    path: trajectories/test-*.parquet
- config_name: people
  data_files:
  - split: train
    path: people/train.parquet
  - split: validation
    path: people/val.parquet
  - split: test
    path: people/test.parquet
- config_name: whatif
  data_files:
  - split: train
    path: whatif/train.parquet
  - split: validation
    path: whatif/val.parquet
  - split: test
    path: whatif/test.parquet
- config_name: whatif_truth
  data_files:
  - split: train
    path: whatif_truth/train.parquet
  - split: validation
    path: whatif_truth/val.parquet
  - split: test
    path: whatif_truth/test.parquet
---

# Crowd What-If

Simulated crowds where you know what everyone *wants*, and **"what if?" pairs**: the same crowd moment replayed under different gate plans, with the true outcome of each.

Use it to train and test models that watch a crowd, guess where people are heading, and predict what happens if you close or open a gate.

![what-if example](https://huggingface.co/datasets/reubenrouse/crowd-whatif/resolve/main/assets/whatif_example.gif)

*One test moment. First the 2 s a model gets to watch, then what really happens under each gate plan. Everyone here wants exit B (orange). Closing B sends them to exit A, which breaks up the queue but gets fewer people out in 10 s.*

- **15,000 scenes**: 6,000 halls, 5,000 train platforms, 4,000 bottlenecks. 1.85M people, 5 frames per second.
- **Hidden labels**: each person's intended exit, current target and desired speed.
- **Actions**: gates closing and opening, train arrivals.
- **9,145 what-if moments**, each replayed under 2–3 gate plans ("keep", "close B", "open C").
- **Realistic simulator**: [JuPedSim](https://www.jupedsim.org), tuned to real crowd experiments (10.6% flow error).

Real recordings only show what actually happened. Here you also get what *would* have happened.

## How it was made

![scene types](https://huggingface.co/datasets/reubenrouse/crowd-whatif/resolve/main/assets/scene_families.png)

1. **Scenes**: random rooms with 1–3 exits, waiting crowds, steady streams and train doors. Each crowd has its own hurry level (0.8–1.6 m/s). Rushed crowds also stand closer together.
2. **Gates** close and reopen during scenes. People heading for a closed exit walk to the nearest open one.
3. **What-if moments**: the simulator restarts from everyone's exact position and replays 10 s once per plan: keep the gates as they are, or change one gate 1 s in. Nobody new enters during the replay.

**Splits** are by scene: 13,500 train, 750 validation, 750 test. What-if moments: 7,072 train, 1,039 validation, 1,034 test.

## Files

| Config | One row per | Main columns |
|---|---|---|
| `scenes` | scene | `scene_id`, `split`, `family`, layout as polygons (`walkable_wkt`, `exits_wkt`, `gates_wkt`), `gate_events`, hidden `hurry` |
| `trajectories` | person per frame | `scene_id`, `person_id`, `t` (s), `x`, `y` (m), `target` (current exit, hidden) |
| `people` | person | `scene_id`, `person_id`, `intended` exit, desired `speed` (hidden) |
| `whatif` | moment | `moment_id`, `scene_id`, `k0` (last watched frame), `plans`, `exit_counts` (people out per exit in 10 s, per plan) |
| `whatif_truth` | person per frame per plan | `moment_id`, `plan`, `person_id`, `n` (frames after `k0`), `x`, `y` |

Frame = round(t × 5), counted from the start of the scene. The 2 s a model "watches" are frames `k0-9 … k0`.

```python
from datasets import load_dataset
import json

moments = load_dataset("reubenrouse/crowd-whatif", "whatif", split="test").to_pandas()
traj = load_dataset("reubenrouse/crowd-whatif", "trajectories", split="test").to_pandas()

m = moments.iloc[0]
watched = traj[(traj.scene_id == m.scene_id) & (traj.t.mul(5).round().between(m.k0 - 9, m.k0))]
for j, plan in enumerate(json.loads(m.plans)):
    print(plan["name"], "-> people out per exit:", json.loads(m.exit_counts)[j])
```

## Benchmark (test set)

Every method sees only 2 s of positions, never the hidden labels.

| Method | Exit-count error (people) | Gets the effect's direction right | Finds the right gate change when one helps | Picks the least crowded plan |
|---|---|---|---|---|
| Random | – | – | – | 48% |
| Never touch the gates | – | – | 0% | 72% |
| Simulator, everyone to the nearest exit | 6.0 | 83% | 47% | 77% |
| Simulator, exit guessed from walking direction | 4.7 | 79% | 58% | 74% |
| [Crowd World Model](https://github.com/reubenjrouse/crowd-world-model) | **3.7** | **93%** | **80%** | **80%** |

*Crowding = people in the busiest 1 m² after 10 s.*

## Limitations

- It's simulated: no pushing, simple route choice, rooms up to 16 × 16 m.
- Nobody new enters during a what-if replay.
- The simulator is deterministic, so each plan has one true outcome.

## Licence

CC BY 4.0. Made with JuPedSim. Code to rebuild everything: [github.com/reubenjrouse/crowd-world-model](https://github.com/reubenjrouse/crowd-world-model).
