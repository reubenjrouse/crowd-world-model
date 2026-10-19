"""Load the Eindhoven Centraal platform 3-4 trajectories (Zenodo record 13784588).

Parquet columns: time_ms, object_identifier, x_position_mm, y_position_mm (10 Hz).
Each file holds 10 days; files are large, so read a limited number of row groups.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "eindhoven"
FPS = 10.0


def parquet_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.parquet"))


def describe(path: Path) -> dict:
    meta = pq.ParquetFile(path).metadata
    return {"rows": meta.num_rows, "row_groups": meta.num_row_groups,
            "columns": [meta.schema.column(i).name for i in range(meta.num_columns)]}


def load(path: Path, max_row_groups: int | None = None) -> pd.DataFrame:
    """Return columns id, t [s], x [m], y [m], sorted by time.

    t counts from the start of the file (real dates were removed for privacy).
    """
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_row_groups if max_row_groups is None else min(max_row_groups, pf.metadata.num_row_groups)
    df = pf.read_row_groups(list(range(n))).to_pandas()
    out = pd.DataFrame({
        "id": df["object_identifier"],
        "t": (df["time_ms"] - df["time_ms"].min()) / 1000.0,
        "x": df["x_position_mm"] / 1000.0,
        "y": df["y_position_mm"] / 1000.0,
    })
    return out.sort_values(["t", "id"]).reset_index(drop=True)
