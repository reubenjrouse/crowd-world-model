"""Crowd measurements shared by exploration, calibration and evaluation."""

import numpy as np
import pandas as pd


def crossing_times(traj: pd.DataFrame, line_y: float = 0.0) -> np.ndarray:
    """Time each person first crosses the horizontal line y = line_y, in either direction."""
    times = []
    for _, p in traj.groupby("id"):
        above = p["y"].to_numpy() >= line_y
        idx = np.flatnonzero(above[:-1] != above[1:])
        if idx.size:
            times.append(p["t"].to_numpy()[idx[0] + 1])
    return np.sort(np.array(times))


def free_speed(traj: pd.DataFrame, region) -> float:
    """Median over people of each person's median speed inside `region` (a shapely geometry).

    Measured past the bottleneck, where people are no longer held up, this approximates
    how fast the crowd wants to walk (the simulator's desired speed).
    """
    import shapely

    t = traj.sort_values(["id", "t"])
    g = t.groupby("id")
    v = np.hypot(g["x"].diff(), g["y"].diff()) / g["t"].diff()
    inside = shapely.contains_xy(region, t["x"].to_numpy(), t["y"].to_numpy())
    per_person = v[inside].groupby(t["id"][inside]).median()
    return float(per_person.median())


def steady_flow(times: np.ndarray) -> float:
    """People per second, ignoring the first and last 10% who cross (start-up and tail)."""
    lo, hi = np.percentile(times, [10, 90])
    n = ((times >= lo) & (times <= hi)).sum()
    return (n - 1) / (hi - lo)
