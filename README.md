# Crowd World Model

An AI that watches 2 seconds of a crowd from above, guesses where everyone is heading, and imagines the next 10 seconds. Ask it "what if I close this gate?" and it recommends the safest plan.

![planner](assets/planner.gif)

*Top: the AI imagines each gate plan and picks the least crowded one. Bottom: what really happens. Closing gate B breaks up the queue, and the AI saw it coming.*

| Synthetic station | Real lab bottleneck (Jülich) | Real train platform (Eindhoven) |
|---|---|---|
| ![synthetic](assets/synthetic.gif) | ![juelich](assets/juelich.gif) | ![eindhoven](assets/eindhoven.gif) |

*Left: reality. Right: the AI's forecast (dots coloured by the exit it thinks each person wants). Bottom: headcount, real (black) vs AI (red) with its 80% range. Typical moments, not hand-picked.*

**Dataset:** [reubenrouse/crowd-whatif](https://huggingface.co/datasets/reubenrouse/crowd-whatif) on Hugging Face. It has 15,000 simulated scenes with hidden destinations, plus 9,145 "what if?" moments, each replayed under every gate plan.

## How it works

1. **Simulator**: [JuPedSim](https://www.jupedsim.org), tuned to real crowd experiments.
2. **Training data**: 15,000 simulated halls, platforms and bottlenecks with gates and train arrivals.
3. **Model**: watches 64×64 crowd maps and predicts how people move between neighbouring cells. People can't vanish or walk through walls.
4. **Real data**: fine-tuned on Jülich lab experiments and the Eindhoven Centraal station platform. Always tested on runs and days it never saw.

## Results

**"What if?"** (1,034 test moments; every method sees only 2 s of positions):

| Method | Effect of a gate change: direction right | Finds the right gate change when one helps | Picks the least crowded plan |
|---|---|---|---|
| Never touch the gates | – | 0% | 72% |
| Simulator, everyone to the nearest exit | 83% | 47% | 77% |
| Simulator, exit guessed from walking direction | 79% | 58% | 74% |
| **This model** | **93%** | **80%** | **80%** |

**Forecasting**, error 10 s ahead relative to "nothing moves" (lower is better):

| Test set | Keep walking straight | **This model** |
|---|---|---|
| Synthetic | 1.25 | **0.55** |
| Jülich bottlenecks (real) | 1.52 | **0.51** |
| Lab crossings (real) | 1.39 | **0.79** |
| Eindhoven platform (real) | 1.30 | **0.86** |

**Also:**
- It guesses destinations correctly 85–88% of the time on real crowds.
- It ranks which crowd is more rushed correctly in 10 of 11 real experiments.
- Its 80% headcount range holds 76–83% of the time.

**Limitations:**
- It predicts density, not individuals.
- It can't see surges coming (e.g. a group still downstairs), so on real platforms it's only modestly better than "nothing moves".
- It underestimates how big a gate's effect is (about 73% of the true size).
- The what-if results are measured in simulation.

## Run it

Setup: [docs/SETUP.md](docs/SETUP.md) (one 8 GB GPU is enough).

```bash
python scripts/calibrate_sim.py                      # tune the simulator
python scripts/generate_scenes.py --n 6000 --start 0     --version v3 --family hall
python scripts/generate_scenes.py --n 5000 --start 6000  --version v3 --family platform
python scripts/generate_scenes.py --n 4000 --start 11000 --version v3 --family bottleneck
python scripts/prepare_dataset.py --version v3
python scripts/prepare_real.py                       # real data (raw files in data/raw)

python scripts/train_wm.py --data v3 --name final_pre --z_dim 8 --beta 0.15 --prior_every 2 --prior_mean \
    --horizon 30 --batch 8 --steps 40000 --lr 3e-4
# then fine-tune on real data three times (5000 steps each, --init_from the previous run,
# --real juelich,crossing,eindhoven; the last one with --crop 0.5): final_real -> _b -> _c

python scripts/final_check.py --run final_real_c     # forecasting results
python scripts/make_whatif.py --split test --per_scene 4
python scripts/eval_whatif.py --run final_real_c --save
python scripts/baseline_whatif.py                    # simulator baselines
python scripts/show_planner.py --run final_real_c --moment scene_001319_k225
```

| Folder | Contents |
|---|---|
| `src/crowdwm/sim/` | Scene generator, simulator helpers |
| `src/crowdwm/data/` | Real-data loaders, training windows |
| `src/crowdwm/model/` | The world model |
| `scripts/` | The steps above |
