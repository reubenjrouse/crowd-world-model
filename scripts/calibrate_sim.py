"""Stage 1a: tune JuPedSim so simulated bottleneck flow matches the real Jülich runs.

Per-crowd "hurry level" (mean desired speed) is MEASURED from the real data: the
median free walking speed past the bottleneck, one value per series.
Shared "body physics" (radius, time_gap) is FITTED: for every setting on the grid,
replay all 15 width runs and compare the steady flow through the bottleneck.

(A first pass that also fitted the speed picked a slower speed for the crowd that
was measured walking faster, trading speed off against flow. Measuring it removes
that freedom.) Writes:
  outputs/calibration/grid.csv           every simulation result
  configs/sim_calibrated.yaml            the chosen settings (tracked in git)
  outputs/figures/calibration_flow.png   real vs default vs calibrated flow

Run:  python scripts/calibrate_sim.py
"""

import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RADII = [0.15, 0.18, 0.20, 0.22]            # realistic body half-widths [m]
TIME_GAPS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SPEED_SD = 0.15
DEFAULT = dict(radius=0.20, time_gap=1.0, speed=1.2)


def simulate(job):
    from crowdwm.data.juelich import SERIES, load_run
    from crowdwm.metrics import crossing_times, free_speed, steady_flow
    from crowdwm.sim.replica import replay_arrivals

    series, name, radius, time_gap, speed = job
    run = load_run(series, name)
    real = run.traj.assign(t=run.traj["t"] - run.traj["t"].min())
    sim, info = replay_arrivals(real, run.walkable_area.polygon, run.exit_polygon,
                                desired_speed=speed, speed_sd=SPEED_SD, radius=radius,
                                time_gap=time_gap)
    region = SERIES[series]["free_region"]
    return dict(series=series, run=name, width=run.width, radius=radius, time_gap=time_gap,
                speed=speed, sim_flow=steady_flow(crossing_times(sim)),
                real_flow=steady_flow(crossing_times(real)),
                sim_free_speed=free_speed(sim, region), real_free_speed=free_speed(real, region),
                stuck=info["left_in_scene"], compute_s=info["wall_seconds"])


def measured_speeds(runs):
    """Hurry level per series = median over its runs of the real free walking speed."""
    from crowdwm.data.juelich import SERIES, load_run
    from crowdwm.metrics import free_speed

    per_run = {}
    for s, n in runs:
        per_run.setdefault(s, []).append(free_speed(load_run(s, n).traj, SERIES[s]["speed_region"]))
    return {s: round(float(np.median(v)), 2) for s, v in per_run.items()}


def main():
    from crowdwm.data.juelich import width_runs

    import sys
    # --exclude series__run,... keeps held-out test runs out of the calibration
    excl = set(sys.argv[sys.argv.index("--exclude") + 1].split(",")) if "--exclude" in sys.argv else set()
    runs = [(s, n) for s, n in width_runs() if f"{s}__{n}" not in excl]
    print(f"calibrating on {len(runs)} runs (excluded: {sorted(excl) or 'none'})")
    speeds = measured_speeds(runs)
    print("measured hurry level (free walking speed) per series:", speeds)
    grid = list(itertools.product(RADII, TIME_GAPS))
    jobs = [(s, n, r, tg, speeds[s]) for s, n in runs for r, tg in grid]
    jobs += [(s, n, DEFAULT["radius"], DEFAULT["time_gap"], DEFAULT["speed"]) for s, n in runs]
    print(f"{len(runs)} runs x {len(grid)} settings = {len(jobs)} simulations on 12 processes")

    t0 = time.perf_counter()
    with Pool(12) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(simulate, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} done, {time.perf_counter() - t0:.0f} s")
    df = pd.DataFrame(rows).drop_duplicates(["series", "run", "radius", "time_gap", "speed"])
    df["rel_err"] = (df["sim_flow"] - df["real_flow"]) / df["real_flow"]
    out_dir = ROOT / "outputs" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "grid.csv", index=False)
    print(f"all done in {time.perf_counter() - t0:.0f} s")

    # Error per shared (radius, time_gap), averaged over the two series
    fitted = df[df.apply(lambda r: r.speed == speeds[r.series], axis=1)
                & ~((df.radius == DEFAULT["radius"]) & (df.time_gap == DEFAULT["time_gap"])
                    & (df.speed == DEFAULT["speed"]))]
    per_series = (fitted.groupby(["radius", "time_gap", "series"])["rel_err"]
                  .apply(lambda e: np.sqrt(np.mean(e**2))).unstack("series"))
    per_series["mean"] = per_series.mean(axis=1)
    per_series = per_series.sort_values("mean")
    print("\nBest shared body settings (RMS relative flow error per series and mean):")
    print(per_series.head(6).round(3).to_string())

    radius, time_gap = per_series.index[0]
    best = fitted[(fitted.radius == radius) & (fitted.time_gap == time_gap)]
    print("\nfree walking speed, real vs simulated (median over runs):")
    print(best.groupby("series")[["real_free_speed", "sim_free_speed"]].median().round(2).to_string())
    config = {"model": "CollisionFreeSpeedModel", "excluded_runs": sorted(excl), "radius": float(radius),
              "time_gap": float(time_gap), "speed_sd": SPEED_SD,
              "desired_speed_by_series": speeds,
              "rms_relative_flow_error": float(per_series["mean"].iloc[0]),
              "rms_relative_flow_error_by_series": {
                  k: float(v) for k, v in per_series.iloc[0].drop("mean").items()}}
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "sim_calibrated.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print("\nchosen:", config)

    # Figure: real vs default vs calibrated flow per run
    sel = pd.concat([df[(df.radius == radius) & (df.time_gap == time_gap)
                        & (df.series == s) & (df.speed == v)] for s, v in speeds.items()])
    dflt = df[(df.radius == DEFAULT["radius"]) & (df.time_gap == DEFAULT["time_gap"])
              & (df.speed == DEFAULT["speed"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, marker in [("caserne_bottleneck_2006", "o"), ("hermes_bottleneck_2009", "s")]:
        r = sel[sel.series == s].sort_values("width")
        d = dflt[dflt.series == s].sort_values("width")
        ax.plot(r.width, r.real_flow, marker, color="black", ms=7,
                label="real" if s.startswith("c") else None)
        ax.plot(d.width, d.sim_flow, marker, color="tab:gray", mfc="none",
                label="simulator, default settings" if s.startswith("c") else None)
        ax.plot(r.width, r.sim_flow, marker, color="tab:red", mfc="none", ms=9,
                label="simulator, calibrated" if s.startswith("c") else None)
    ax.set_xlabel("Bottleneck width [m]   (circles: Caserne, squares: Hermes)")
    ax.set_ylabel("Steady flow [people / s]")
    ax.set_title(f"Calibration: radius {radius} m, time gap {time_gap} s,\n"
                 f"hurry level {speeds}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / "calibration_flow.png", dpi=150)
    print("saved outputs/figures/calibration_flow.png")


if __name__ == "__main__":
    main()
