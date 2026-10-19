"""Scoring "what if?" answers against re-simulated truth (see scripts/make_whatif.py).

Every predictor (the world model, simulator baselines) produces, per held-out moment:
  true, ai          (plans, exits) people through each exit within 10 s
  true_peak, ai_peak (plans,) people in the busiest 1 m^2 at 10 s
  map_err, map_persist  per plan: people misplaced on the map at 10 s, by the predictor
                    and by "nothing moves"
and summarize() turns a list of those into the same report for all of them.
"""

import numpy as np
import torch
import torch.nn.functional as F


def busiest_m2(maps) -> np.ndarray:
    """(N, H, W) density maps (people per 0.25 m cell) -> people in the busiest 1 m^2, per map."""
    t = torch.as_tensor(np.asarray(maps), dtype=torch.float32)[:, None]
    return (F.avg_pool2d(t, 4, stride=1) * 16).flatten(1).amax(1).numpy()


def summarize(rows: list, label: str, seed: int = 0) -> dict:
    """Print the what-if report for one predictor; returns the headline numbers."""
    rng = np.random.default_rng(seed)
    out = {}
    eff_t = np.concatenate([(np.asarray(r["true"])[1:] - np.asarray(r["true"])[0]).ravel() for r in rows])
    eff_a = np.concatenate([(np.asarray(r["ai"])[1:] - np.asarray(r["ai"])[0]).ravel() for r in rows])
    big = np.abs(eff_t) >= 3
    out["exit_miss"] = float(np.mean([np.abs(np.asarray(r["ai"]) - np.asarray(r["true"])).mean() for r in rows]))
    out["direction"] = float(np.mean(np.sign(eff_a[big]) == np.sign(eff_t[big])))
    out["correlation"] = float(np.corrcoef(eff_a, eff_t)[0, 1])
    out["effect_size"] = float(np.median(eff_a[big] / eff_t[big]))

    def picks(subset_keep: bool):
        """Plan choice by lowest peak crowding. subset_keep=False: only the gate actions
        (one gate must change); True: all plans including keep."""
        hit, n, cost, keep_hit, rand_hit = 0, 0, [], 0, []
        for r in rows:
            tv, av = np.asarray(r["true_peak"]), np.asarray(r["ai_peak"])
            if not subset_keep:
                if len(tv) < 3:
                    continue
                tv, av = tv[1:], av[1:]
            if np.ptp(tv) < 0.5:          # plans make no real difference
                continue
            n += 1
            hit += tv[np.argmin(av)] == tv.min()
            cost.append(tv[np.argmin(av)] - tv.min())
            keep_hit += tv[0] == tv.min()
            rand_hit.append(np.mean(tv == tv.min()))
        return hit / n, n, float(np.mean(cost)), keep_hit / n, float(np.mean(rand_hit))

    out["pick_safest"], n_all, out["pick_cost"], keep, rand = picks(True)
    out["pick_gate"], n_gate, _, _, rand_gate = picks(False)
    act = [r for r in rows if np.ptp(r["true_peak"]) >= 0.5 and np.argmin(r["true_peak"]) != 0]
    out["find_action"] = float(np.mean([np.argmin(r["ai_peak"]) == np.argmin(r["true_peak"]) for r in act]))
    out["map_score"] = float(np.mean(np.concatenate([r["map_err"] for r in rows])) /
                             np.mean(np.concatenate([r["map_persist"] for r in rows])))

    print(f"\n=== {label}: {len(rows)} moments")
    print(f"  exit counts: average miss per exit {out['exit_miss']:.1f} people")
    print(f"  effect of the action (exits moved by >= 3 people: {big.sum()}): direction right {out['direction']:.0%}, "
          f"correlation {out['correlation']:.2f}, size {out['effect_size']:.2f} x true")
    print(f"  safest plan (lowest peak crowding at 10 s; {n_all} moments where it matters): picks the best "
          f"{out['pick_safest']:.0%} (always keep {keep:.0%}, random {rand:.0%}), cost {out['pick_cost']:.2f} people/m^2")
    print(f"  when acting is truly safer ({len(act)} moments): finds the right action {out['find_action']:.0%}")
    print(f"  which gate, if one must change ({n_gate} moments): {out['pick_gate']:.0%} (random {rand_gate:.0%})")
    print(f"  crowd map at 10 s vs nothing moves: {out['map_score']:.2f}")
    return out
