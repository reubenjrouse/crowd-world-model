"""Load runs from the Jülich pedestrian dynamics data archive (ped.fz-juelich.de/da).

Each archive series ships trajectory .txt files (id frame x y [z], usually in cm)
plus a metadata JSON that holds, per run, the walkable area as WKT and the
experiment parameters (e.g. bottleneck width).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pedpy
import shapely

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "juelich"

# Per-series settings the text files don't carry themselves.
# The bottleneck entrance/exit line is y = 0 in both series.
# exit: the simulator's exit. JuPedSim agents walk toward the exit's centre, so an exit
#   far downstream makes everyone converge and jam there; instead use a wide exit just
#   past the bottleneck, so people leave along a broad front as they do in reality.
# free_region: just past the bottleneck, before the exit; compare real vs sim speed here.
# speed_region: further downstream (real data only), where people walk freely; used to
#   measure the crowd's desired walking speed.
# view_exit: where people leave the cameras' view in the real recordings; this is the
#   exit the world model is shown for real runs.
SERIES = {
    "caserne_bottleneck_2006": {                          # crowd walks upward
        "metadata": "2006bottleneck2_metadata.json", "fps": 25.0,
        "width_key": "bottleneck_width",
        "exit": shapely.box(-3.5, 4.1, 3.5, 4.5),
        "view_exit": shapely.box(-3.5, 4.1, 3.5, 4.6),
        "free_region": shapely.box(-2.0, 2.0, 2.0, 4.0),
        "speed_region": shapely.box(-2.0, 2.0, 2.0, 4.2),
    },
    "hermes_bottleneck_2009": {                           # crowd walks downward
        "metadata": "2009bottleneck_metadata.json", "fps": 16.0,
        "width_key": "b_Exit [m]",
        "exit": shapely.box(-5.9, -3.0, 5.9, -2.6),
        "view_exit": shapely.box(-5.9, -6.6, 5.9, -6.0),
        "free_region": shapely.box(-5.0, -2.5, 5.0, -0.8),
        "speed_region": shapely.box(-3.0, -6.0, 5.0, -2.0),
    },
}


def width_runs() -> list[tuple[str, str]]:
    """All (series, run) pairs that vary only the bottleneck width."""
    return [(s, n) for s in SERIES for n in list_runs(s)
            if not (s == "caserne_bottleneck_2006" and not n.startswith("b"))]


@dataclass
class Run:
    series: str
    name: str
    fps: float
    traj: pd.DataFrame          # columns: id, frame, t [s], x [m], y [m]
    walkable_area: pedpy.WalkableArea
    params: dict = field(default_factory=dict)

    @property
    def n_people(self) -> int:
        return self.traj["id"].nunique()

    @property
    def duration(self) -> float:
        return self.traj["t"].max() - self.traj["t"].min()

    @property
    def width(self) -> float:
        return float(self.params[SERIES[self.series]["width_key"]])

    @property
    def exit_polygon(self):
        return SERIES[self.series]["exit"]


def _run_metadata(series: str) -> dict[str, dict]:
    meta_path = RAW_DIR / series / SERIES[series]["metadata"]
    exp = json.loads(meta_path.read_text(encoding="utf-8"))["experiment"]
    # Older metadata files use "parameters", newer ones "run"
    runs = exp.get("parameters") or exp.get("run") or []
    return {r["run_name"]: r for r in runs}


def list_runs(series: str) -> list[str]:
    """Runs that are both described in the metadata and present on disk."""
    return sorted(n for n in _run_metadata(series) if (RAW_DIR / series / f"{n}.txt").exists())


def load_run(series: str, name: str) -> Run:
    fps = SERIES[series]["fps"]
    meta = _run_metadata(series)[name]

    # Files have 4 columns (id frame x y) or 5 (plus z, the head height)
    df = pd.read_csv(RAW_DIR / series / f"{name}.txt", sep=r"\s+", comment="#", header=None)
    df.columns = ["id", "frame", "x", "y", "z"][: df.shape[1]]
    df["x"] = df["x"] / 100.0   # cm -> m
    df["y"] = df["y"] / 100.0
    df["t"] = df["frame"] / fps
    traj = df[["id", "frame", "t", "x", "y"]].sort_values(["frame", "id"]).reset_index(drop=True)

    area = pedpy.WalkableArea(shapely.from_wkt(meta["wkt_geometry"]))
    return Run(series=series, name=name, fps=fps, traj=traj,
               walkable_area=area,
               params=meta.get("parameters") or meta.get("parameter") or {})
