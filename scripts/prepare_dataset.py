"""Build cache.npz for every generated scene (run after generate_scenes.py).

Run:  python scripts/prepare_dataset.py --version v1
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def work(args):
    d, force = args
    from crowdwm.data.sim_dataset import prepare_scene
    return prepare_scene(Path(d), force=force) is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    a = p.parse_args()
    root = ROOT / "data" / "processed" / "sim" / a.version
    dirs = [str(d) for d in sorted(root.glob("scene_*")) if (d / "info.json").exists()]
    t0 = time.perf_counter()
    empty = 0
    with Pool(a.workers) as pool:
        for i, ok in enumerate(pool.imap_unordered(work, [(d, a.force) for d in dirs], chunksize=8)):
            empty += not ok
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(dirs)}, {time.perf_counter() - t0:.0f} s")
    caches = [Path(d) / "cache.npz" for d in dirs if (Path(d) / "cache.npz").exists()]
    size = sum(c.stat().st_size for c in caches) / 1e6
    print(f"{len(caches)} scenes cached in {time.perf_counter() - t0:.0f} s, {size:.0f} MB; "
          f"{empty} empty scenes skipped")


if __name__ == "__main__":
    main()
