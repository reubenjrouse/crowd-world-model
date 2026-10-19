"""Restart a synthetic scene's simulator from a given moment, with chosen hidden state.

Used by the what-if demo (scripts/show_whatif.py) to get the true outcome of each gate
plan: the simulator starts from the positions observed at time t0, with chosen
destinations and speeds.
The scene's geometry, gate plan, deadlock fix and exit rules are the same as in
scenes.run_scene, so with the true hidden state this reproduces the original closely.
"""

import time

import jupedsim as jps
import numpy as np
import pandas as pd
import shapely

from crowdwm.sim.scenes import DT, FPS_OUT, SceneSpec, exit_parts, walkable_area
from crowdwm.sim.unstick import Unstick


def simulate_from(spec: SceneSpec, t0: float, people: list, duration: float,
                  seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Simulate `duration` seconds starting at scene time t0.

    people: list of (id, x, y, exit, desired_speed); exit is the person's intended exit.
    Returns (trajectories with id, n, x, y where n = frames after t0 at 5 fps, wall time).
    """
    rng = np.random.default_rng(seed)
    closed = set()
    for t, i, what in spec.gate_events:          # gate states in force at t0
        if t <= t0:
            closed = closed | {i} if what == "close" else closed - {i}
    pending_gates = [(t - t0, i, w) for t, i, w in spec.gate_events if t > t0]

    model = getattr(jps, spec.model)
    params = getattr(jps, spec.model + "AgentParameters")
    sim = jps.Simulation(model=model(), geometry=walkable_area(spec, closed), dt=DT)
    parts = [exit_parts(spec, e) for e in spec.exits]
    stages = [sim.add_exit_stage(list(p["stage"].exterior.coords)[:-1]) for p in parts]
    journeys = [sim.add_journey(jps.JourneyDescription([s])) for s in stages]
    centres = np.array([[p["stage"].centroid.x, p["stage"].centroid.y] for p in parts])

    def nearest_open(pos, wanted):
        if wanted not in closed:
            return wanted
        d = np.linalg.norm(centres - np.asarray(pos), axis=1)
        d[list(closed)] = np.inf
        return int(np.argmin(d))

    intended, target_of, orig_id = {}, {}, {}
    t_start = time.perf_counter()
    for pid, x, y, ex, v0 in people:
        for attempt in range(6):   # recorded positions can be a hair too close: jitter slightly
            jx, jy = (x, y) if attempt == 0 else (x + rng.normal(0, 0.05), y + rng.normal(0, 0.05))
            tgt = nearest_open((jx, jy), int(ex))
            try:
                aid = sim.add_agent(params(journey_id=journeys[tgt], stage_id=stages[tgt],
                                           position=(jx, jy), desired_speed=float(v0),
                                           radius=spec.radius, time_gap=spec.time_gap))
            except Exception:
                continue
            intended[aid], target_of[aid], orig_id[aid] = int(ex), tgt, pid
            break

    unstick = Unstick(sim, spec.radius, spec.time_gap) if spec.unstick else None
    record_every = round(1 / (FPS_OUT * DT))
    rows = []
    while sim.elapsed_time() <= duration + 1e-9:
        now = sim.elapsed_time()
        while pending_gates and pending_gates[0][0] <= now:
            _, i, what = pending_gates[0]
            new_closed = closed | {i} if what == "close" else closed - {i}
            try:
                sim.switch_geometry(walkable_area(spec, new_closed))
            except Exception:
                pending_gates[0] = (now + 0.2, i, what)
                break
            pending_gates.pop(0)
            closed = new_closed
            for a in sim.agents():
                t = target_of[a.id]
                if t in closed and shapely.contains_xy(parts[t]["stub"], *a.position):
                    continue
                new = nearest_open(a.position, intended[a.id])
                if new != t:
                    sim.switch_agent_journey(agent_id=a.id, journey_id=journeys[new], stage_id=stages[new])
                    target_of[a.id] = new
        if sim.iteration_count() % record_every == 0:
            n = round(now * FPS_OUT)
            rows += [(orig_id[a.id], n, a.position[0], a.position[1]) for a in sim.agents()]
            for a in sim.agents():
                for i, p in enumerate(parts):
                    if i != target_of[a.id] and i not in closed and shapely.contains_xy(p["stage"], *a.position):
                        sim.mark_agent_for_removal(a.id)
        if unstick:
            unstick.step()
        sim.iterate()
    return pd.DataFrame(rows, columns=["id", "n", "x", "y"]), time.perf_counter() - t_start
