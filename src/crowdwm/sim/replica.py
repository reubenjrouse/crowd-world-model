"""Re-create a real run in JuPedSim: same room, same people appearing at the same
times and places. The simulator then decides how everyone moves.

Comparing the replica with the real recording shows how realistic the simulator is
before any tuning (calibration comes in Stage 1).
"""

import time

import jupedsim as jps
import numpy as np
import pandas as pd

from crowdwm.sim.unstick import Unstick

DT = 0.01  # simulator time step [s]


def replay_arrivals(real: pd.DataFrame, walkable, exit_polygon, fps_out: float = 5.0,
                    desired_speed: float = 1.2, radius: float = 0.2, time_gap: float = 1.0,
                    speed_sd: float = 0.0, seed: int = 0, unstick: bool = False,
                    max_time: float = 400.0) -> tuple[pd.DataFrame, dict]:
    """Simulate a crowd whose arrivals copy the real one.

    Each simulated person appears where and when the real person first appeared,
    then walks to the exit. Each person's desired speed is drawn from
    N(desired_speed, speed_sd), clipped to [0.4, 2.5] m/s.
    Returns (trajectories with id/t/x/y, info dict).
    """
    rng = np.random.default_rng(seed)
    first = real.sort_values("t").groupby("id").first().sort_values("t")
    t_offset = real["t"].min()
    speeds = np.clip(rng.normal(desired_speed, speed_sd, len(first)), 0.4, 2.5)
    arrivals = list(zip(first["t"] - t_offset, first["x"], first["y"], speeds))

    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(), geometry=walkable, dt=DT)
    exit_id = sim.add_exit_stage(list(exit_polygon.exterior.coords)[:-1])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))

    record_every = round(1 / (fps_out * DT))
    rows, pending, failed_spawns = [], list(arrivals), 0
    breaker = Unstick(sim, radius, time_gap) if unstick else None
    t0 = time.perf_counter()
    while (pending or sim.agent_count() > 0) and sim.elapsed_time() < max_time:
        now = sim.elapsed_time()
        retry = []
        while pending and pending[0][0] <= now:
            t_arr, x, y, v0 = pending.pop(0)
            try:
                sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
                    journey_id=journey_id, stage_id=exit_id, position=(x, y),
                    desired_speed=v0, radius=radius, time_gap=time_gap))
            except Exception:  # spot occupied or too close to a wall: try again shortly
                failed_spawns += 1
                retry.append((now + 0.2, x, y, v0))
        pending = sorted(retry + pending)

        if sim.iteration_count() % record_every == 0:
            t = sim.elapsed_time()
            rows += [(a.id, t, a.position[0], a.position[1]) for a in sim.agents()]
        if breaker:
            breaker.step()
        sim.iterate()

    traj = pd.DataFrame(rows, columns=["id", "t", "x", "y"])
    info = {"people": len(arrivals), "sim_seconds": sim.elapsed_time(),
            "wall_seconds": time.perf_counter() - t0, "failed_spawns": failed_spawns,
            "left_in_scene": sim.agent_count(), "squeezes": breaker.count if breaker else 0}
    return traj, info
