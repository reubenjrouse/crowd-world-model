"""Turn real recordings into the same cache format as synthetic scenes (see sim_dataset.py),
so the world model can be evaluated and fine-tuned on real people with the same code.

Each real run becomes data/processed/real/<dataset>/<run>/ with:
  cache.npz   same fields as a synthetic scene (no gates; hurry = measured free speed)
  spec.json   {"family", "source", "series", "run", "width", "walkable_wkt", "exits_wkt"}
"""

import json
from pathlib import Path

import numpy as np
import shapely

from crowdwm.raster import MAX_EXITS, Grid, resample, static_layers

ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = ROOT / "data" / "processed" / "real"
FPS = 5.0


def open_boundary(r, grid: Grid, static, min_events: int = 3, reach_m: float = 1.5) -> np.ndarray:
    """(H, W) uint8: cells where people walk into or out of the camera view away from marked
    exits. Cells with at least `min_events` track starts or ends, within `reach_m` of a
    wall or the map edge (tracking glitches in the open middle of a scene don't count),
    excluding walls and exits, widened by one cell. A fixed property of the recording
    setup, like the walls."""
    from scipy.ndimage import binary_dilation

    g = r.sort_values("k").groupby("id")[["x", "y"]]
    pts = np.concatenate([g.first().to_numpy(), g.last().to_numpy()])
    xs, ys = grid.edges
    h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ys, xs])
    blocked = (static[0] > 0) | (static[1:].max(0) > 0)
    rim = static[0] > 0
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    near_rim = binary_dilation(rim, iterations=int(round(reach_m / grid.cell)))
    edge = binary_dilation((h >= min_events) & near_rim & ~blocked, iterations=1) & ~blocked
    return edge.astype(np.uint8)


def _write(out_dir: Path, traj, walkable, exits, hurry: float, meta: dict, grid: Grid,
           arrivals=None) -> Path:
    """traj: DataFrame with id, t [s], x, y and optional target (exit slot index).
    arrivals: optional function (frame k) -> list of polygons where people are scheduled
    to arrive at that frame (e.g. train doors while unloading)."""
    r = resample(traj.assign(t=traj["t"] - traj["t"].min()), FPS)
    k = r["k"].to_numpy()
    order = np.argsort(k, kind="stable")
    n_frames = int(k.max()) + 1
    ptr = np.searchsorted(k[order], np.arange(n_frames + 1))
    target = r["target"].to_numpy(np.int8) if "target" in r else np.zeros(len(r), np.int8)
    static = static_layers(grid, walkable, exits)
    out_dir.mkdir(parents=True, exist_ok=True)
    from crowdwm.data.sim_dataset import first_appearances, pack_arrivals
    extra_static = {"boundary": open_boundary(r, grid, static)}
    # Where people walked into view (shown to the model for watched frames only)
    extra = {"inflow": pack_arrivals(first_appearances(k, r["id"].to_numpy(), r[["x", "y"]].to_numpy(),
                                                       n_frames, grid))}
    if arrivals is not None:
        boxes = grid.cell_boxes()
        arr = np.zeros((n_frames, grid.size, grid.size), bool)
        for kk in range(n_frames):
            for poly in arrivals(kk):
                arr[kk] |= shapely.intersects(boxes, poly)
        extra["arrivals"] = pack_arrivals(arr)
    np.savez_compressed(
        out_dir / "cache.npz", x0=grid.x0, y0=grid.y0, static=static.astype(np.uint8),
        gate_masks=np.zeros((MAX_EXITS, grid.size, grid.size), np.uint8),
        gate_state=np.zeros((n_frames, MAX_EXITS), np.uint8), ptr=ptr,
        xy=r[["x", "y"]].to_numpy(np.float32)[order], target=target[order],
        pid=r["id"].to_numpy(np.int64)[order],   # person ids (real data), for per-person scoring
        hurry=np.float32(hurry), n_exits=len(exits), **extra, **extra_static)
    meta = dict(meta, walkable_wkt=walkable.wkt, exits_wkt=[e.wkt for e in exits])
    (out_dir / "spec.json").write_text(json.dumps(meta))
    return out_dir


def prepare_juelich(series: str, name: str) -> Path:
    from crowdwm.data.juelich import SERIES, load_run
    from crowdwm.metrics import free_speed

    run = load_run(series, name)
    walkable = run.walkable_area.polygon
    exit_poly = SERIES[series]["view_exit"]
    # Centre the 16 m grid on where people actually are (the Hermes hall is 18 m long)
    seen = shapely.box(run.traj.x.min(), run.traj.y.min(), run.traj.x.max(), run.traj.y.max())
    grid = Grid.centred_on(shapely.union_all([seen, exit_poly]))
    hurry = free_speed(run.traj, SERIES[series]["speed_region"])
    meta = dict(family="bottleneck", source="juelich", series=series, run=name, width=run.width,
                people=run.n_people)
    return _write(REAL_DIR / "juelich" / f"{series}__{name}", run.traj, walkable.intersection(
        grid_box(grid)), [exit_poly], hurry, meta, grid)


CROSSING_DIR = ROOT / "data" / "raw" / "juelich" / "crossing_90_2013"


def _arm(x, y, cx, cy):
    dx, dy = x - cx, y - cy
    return np.where(np.abs(dx) > np.abs(dy), np.where(dx > 0, "E", "W"), np.where(dy > 0, "N", "S"))


def prepare_crossing(name: str) -> Path | None:
    """90-degree crossing run with two perpendicular streams (parts D and E).

    Each person's true destination is the arm where their path ends. Exit polygons are
    placed where tracks end on each destination arm. Returns None for runs that use more
    exits than the model has slots.
    """
    import pandas as pd

    meta = json.loads((CROSSING_DIR / "2013crossing_90_metadata.json").read_text(encoding="utf-8"))
    run = next(r for r in meta["experiment"]["run"] if r["run_name"] == name)
    df = pd.read_csv(CROSSING_DIR / f"{name}.txt", sep=r"\s+", comment="#", header=None).iloc[:, :4]
    df.columns = ["id", "frame", "x", "y"]
    if df[["x", "y"]].abs().max().max() > 50:      # some files are in cm despite the header
        df[["x", "y"]] /= 100.0
    df["t"] = df["frame"] / 16.0
    walkable = shapely.from_wkt(run["wkt_geometry"])

    ends = df.sort_values("frame").groupby("id")[["x", "y"]].last()
    cx, cy = (df.x.min() + df.x.max()) / 2, (df.y.min() + df.y.max()) / 2
    end_arm = pd.Series(_arm(ends.x.to_numpy(), ends.y.to_numpy(), cx, cy), index=ends.index)
    counts = end_arm.value_counts()
    arms = sorted(counts[counts >= 0.05 * len(end_arm)].index)
    if len(arms) > MAX_EXITS:
        return None
    exits = []
    for a in arms:                                 # where tracks end on this arm, padded
        e = ends[end_arm == a]
        x0, x1 = e.x.quantile([0.02, 0.98])
        y0, y1 = e.y.quantile([0.02, 0.98])
        if a in "NS":
            y = e.y.median()
            box = shapely.box(x0 - 0.3, y - 0.4, x1 + 0.3, y + 0.4)
        else:
            x = e.x.median()
            box = shapely.box(x - 0.4, y0 - 0.3, x + 0.4, y1 + 0.3)
        exits.append(box.intersection(walkable))
    slot = {a: i for i, a in enumerate(arms)}
    df["target"] = df["id"].map(end_arm).map(slot).fillna(-1).astype(int)
    df = df[df.target >= 0]                        # drop the few who ended on a stray arm

    seen = shapely.box(df.x.min(), df.y.min(), df.x.max(), df.y.max())
    grid = Grid.centred_on(shapely.union_all([seen, *exits]))
    from crowdwm.metrics import free_speed
    hurry = free_speed(df, seen)
    meta_out = dict(family="crossing", source="juelich_crossing", series="crossing_90", run=name,
                    exit_arms=arms, people=int(df.id.nunique()),
                    turning=float(1 - np.mean([end_arm[i] in arms for i in end_arm.index])))
    return _write(REAL_DIR / "crossing" / name, df[["id", "t", "x", "y", "target"]],
                  walkable.intersection(grid_box(grid)), exits, hurry, meta_out, grid)


CROWDQUEUE_DIR = ROOT / "data" / "raw" / "juelich" / "crowdqueue_2018"


def prepare_crowdqueue(name: str) -> Path:
    """CrowdQueue run (high 'h0' vs low 'h-' motivation, 0.5 m entrance at the end of a
    corridor). Single exit, placed where tracks end; hurry label = measured free speed."""
    import pandas as pd

    meta = json.loads((CROWDQUEUE_DIR / "2018crowdqueue.json").read_text(encoding="utf-8"))
    run = next(r for r in meta["experiment"]["run"] if r["run_name"] == name)
    df = pd.read_csv(CROWDQUEUE_DIR / f"{name}.txt", sep=r"\s+", comment="#", header=None).iloc[:, :4]
    df.columns = ["id", "frame", "x", "y"]
    df["t"] = df["frame"] / 25.0
    walkable = shapely.from_wkt(run["wkt_geometry"])
    ends = df.sort_values("frame").groupby("id")[["x", "y"]].last()
    ex, ey = ends.x.median(), ends.y.median()
    exit_poly = shapely.box(ex - 0.6, ey - 0.4, ex + 0.6, ey + 0.4).intersection(walkable)
    seen = shapely.box(df.x.min(), df.y.min(), df.x.max(), df.y.max())
    grid = Grid.centred_on(shapely.union_all([seen, exit_poly]))
    from crowdwm.metrics import free_speed
    params = run["parameter"]
    meta_out = dict(family="bottleneck", source="juelich_crowdqueue", series="crowdqueue", run=name,
                    width=float(params["corridor_width"]), motivation=params["motivation"],
                    priming=params["priming"], people=int(df.id.nunique()))
    return _write(REAL_DIR / "crowdqueue" / name, df[["id", "t", "x", "y"]],
                  walkable.intersection(grid_box(grid)), [exit_poly], free_speed(df, seen), meta_out, grid)


EINDHOVEN_FILE = ROOT / "data" / "raw" / "eindhoven" / "Eindhoven_centraal_trajectories_days_11_20.parquet"
# A 16 m window at the stairs end of the platform (the full platform is ~82 m long)
EIN_CROP = (-5.5, -9.5, 10.5, 4.5)
# Ways down to the station hall: central stairs, side-lane stairs, escalator heads
EIN_STAIRS = [(-5.2, 1.4, -3.6, 3.9), (-5.2, -8.2, -3.6, -5.7), (-0.6, -4.1, 0.7, 1.3),
              (3.5, 1.5, 5.0, 3.4), (3.3, -7.9, 4.8, -6.1)]
EIN_NAMES = ["stairs", "board train", "walk on"]


def prepare_eindhoven(n_per_day: int = 10, before_s: float = 30, after_s: float = 120) -> list[Path]:
    """Real train arrivals at Eindhoven Centraal, cut to a window around the stairs.

    The walkable area is where people are actually seen (10 days of positions). Each
    person's destination is where their path leaves the window: stairs (0), boarding a
    train at either track edge (1), or walking on along the platform (2). Events are the
    `n_per_day` busiest moments per day (train arrivals), `before_s` .. `after_s` around
    each peak. Day index (0-9) is stored, for a train/test split by day.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from shapely.ops import unary_union

    x0, y0, x1, y1 = EIN_CROP
    tb = pq.read_table(EINDHOVEN_FILE, filters=[("x_position_mm", ">", x0 * 1000 - 500),
                                               ("x_position_mm", "<", x1 * 1000 + 500)])
    df = tb.to_pandas()
    df = pd.DataFrame({"id": df["object_identifier"].to_numpy(), "t": (df["time_ms"] - df["time_ms"].min()) / 1000.0,
                       "x": df["x_position_mm"] / 1000.0, "y": df["y_position_mm"] / 1000.0})
    df = df[(df.x >= x0) & (df.x <= x1) & (df.y >= y0) & (df.y <= y1)]

    # Walkable area: 25 cm cells where people are regularly seen
    cell = 0.25
    h, xe, ye = np.histogram2d(df.x.iloc[::5], df.y.iloc[::5], bins=[int((x1 - x0) / cell), int((y1 - y0) / cell)],
                               range=[[x0, x1], [y0, y1]])
    keep = h > np.percentile(h[h > 0], 15)
    boxes = [shapely.box(xe[i], ye[j], xe[i + 1], ye[j + 1]) for i, j in zip(*np.nonzero(keep))]
    walkable = unary_union(boxes).buffer(0.2).buffer(-0.2)
    if walkable.geom_type == "MultiPolygon":
        walkable = max(walkable.geoms, key=lambda g: g.area)
    walkable = shapely.Polygon(walkable.exterior)        # ignore small interior specks

    stairs = unary_union([shapely.box(*b) for b in EIN_STAIRS]).intersection(walkable)
    wy0, wy1 = walkable.bounds[1], walkable.bounds[3]
    board = unary_union([shapely.box(-3.4, wy1 - 0.6, x1 - 0.6, wy1),
                         shapely.box(-3.4, wy0, x1 - 0.6, wy0 + 0.6)]).intersection(walkable)
    walk_on = shapely.box(x1 - 0.6, wy0, x1, wy1).intersection(walkable)
    exits = [stairs, board, walk_on]

    # Destination of every track = exit nearest its last position in the window
    last = df.sort_values("t").groupby("id")[["x", "y"]].last()
    dist = np.stack([shapely.distance(e, shapely.points(last.x.to_numpy(), last.y.to_numpy())) for e in exits])
    target = pd.Series(dist.argmin(0), index=last.index)
    clean = pd.Series(dist.min(0) < 0.75, index=last.index)   # ended at an exit (not a tracking loss)
    df["target"] = df["id"].map(target)

    # Train arrivals: busiest 10 s bins, at least 5 min apart, per day
    df["day"] = (df.t // 86400).astype(int)
    bins = df.assign(b=(df.t // 10).astype(int)).groupby("b")["id"].nunique()
    out = []
    grid = Grid.centred_on(shapely.box(*EIN_CROP))
    for day in sorted(df.day.unique()):
        cand = bins[(bins.index * 10 // 86400) == day].sort_values(ascending=False)
        picked = []
        for b, n in cand.items():
            if len(picked) >= n_per_day or n < 40:
                break
            if all(abs(b - p) * 10 >= 300 for p in picked):
                picked.append(b)
        for p in picked:
            t_peak = p * 10
            seg = df[(df.t >= t_peak - before_s) & (df.t < t_peak + after_s)]
            name = f"day{day}_t{int(t_peak)}"
            # Train unloading = a burst of people appearing along a track edge (>= 3 within
            # +-2 s). Stands in for the timetable a station would know in advance.
            starts = seg.sort_values("t").groupby("id")[["t", "x", "y"]].first()
            t_rel = starts.t - seg.t.min()
            edge_strips = {"N": shapely.box(-3.4, wy1 - 1.2, x1 - 0.6, wy1).intersection(walkable),
                           "S": shapely.box(-3.4, wy0, x1 - 0.6, wy0 + 1.2).intersection(walkable)}
            on_edge = {"N": (starts.y > wy1 - 1.2) & (starts.x > -3.4),
                       "S": (starts.y < wy0 + 1.2) & (starts.x > -3.4)}
            frames_of = {e: np.round(t_rel[on_edge[e]].to_numpy() * 5).astype(int) for e in edge_strips}

            def arriving(k, frames_of=frames_of, edge_strips=edge_strips):
                return [edge_strips[e] for e, f in frames_of.items() if np.sum(np.abs(f - k) <= 10) >= 3]
            meta = dict(family="platform", source="eindhoven", series="eindhoven", run=name, day=int(day),
                        exit_names=EIN_NAMES, people=int(seg.id.nunique()),
                        clean_destinations=float(clean[seg.id.unique()].mean()))
            out.append(_write(REAL_DIR / "eindhoven" / name, seg[["id", "t", "x", "y", "target"]],
                              walkable, exits, float("nan"), meta, grid,   # no hurry label
                              arrivals=arriving))
    return out


def grid_box(grid: Grid):
    return shapely.box(grid.x0, grid.y0, grid.x0 + grid.cell * grid.size, grid.y0 + grid.cell * grid.size)
