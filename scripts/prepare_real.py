"""Convert every real dataset into the model's cache format (data/processed/real/...).

  juelich      15 bottleneck width runs (Caserne 2006, Hermes 2009)
  crossing     90-degree crossing, parts D and E (two perpendicular streams)
  crowdqueue   high vs low motivation runs
  eindhoven    10 busiest train arrivals per day, days 11-20 file

Run:  python scripts/prepare_real.py [juelich crossing crowdqueue eindhoven]
"""

import json
import sys
import time

from crowdwm.data import real_dataset as rd
from crowdwm.data.juelich import width_runs

which = sys.argv[1:] or ["juelich", "crossing", "crowdqueue", "eindhoven"]
t0 = time.perf_counter()
if "juelich" in which:
    for s, n in width_runs():
        rd.prepare_juelich(s, n)
    print(f"juelich: {len(width_runs())} runs")
if "crossing" in which:
    names = sorted(p.stem for p in rd.CROSSING_DIR.glob("crossing_90_[de]_*.txt"))
    done = [n for n in names if rd.prepare_crossing(n) is not None]
    print(f"crossing: {len(done)} runs")
if "crowdqueue" in which:
    meta = json.loads((rd.CROWDQUEUE_DIR / "2018crowdqueue.json").read_text(encoding="utf-8"))
    names = [r["run_name"] for r in meta["experiment"]["run"] if (rd.CROWDQUEUE_DIR / f"{r['run_name']}.txt").exists()]
    for n in names:
        rd.prepare_crowdqueue(n)
    print(f"crowdqueue: {len(names)} runs")
if "eindhoven" in which:
    print(f"eindhoven: {len(rd.prepare_eindhoven(10))} events")
print(f"done in {time.perf_counter() - t0:.0f} s")
