"""Random synthetic scenes for training: rooms, exits, gates, arrivals, hidden goals.

A scene is described by a SceneSpec (plain data, JSON-serialisable) and run with
run_scene(). Because the spec is data, the same crowd can be re-run under a
different gate schedule to get counterfactual pairs ("what if gate B had closed?").

Coordinates: the room's lower-left corner is (0, 0); doorway stubs stick out 1 m.
Everything fits in 16 x 16 m so one 64 x 64 grid at 0.25 m covers the whole scene.
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jupedsim as jps
import numpy as np
import pandas as pd
import shapely
import yaml

from crowdwm.sim.unstick import Unstick

DT = 0.01
FPS_OUT = 5.0
STUB = 1.0          # doorway corridor depth [m]
CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sim_calibrated.yaml"


# ---------------------------------------------------------------- specification

@dataclass
class ExitSpec:
    side: str           # "N", "S", "E" or "W"
    centre: float       # position along that wall [m]
    width: float        # doorway width [m]
    gated: bool = False


@dataclass
class GroupSpec:
    kind: str           # "waiting" (all at once), "burst" (train doors), "stream" (steady)
    zone: tuple         # spawn rectangle (x0, y0, x1, y1)
    t_start: float
    duration: float     # arrivals spread over this many seconds (0 for "waiting")
    n: int
    dest_mix: list      # probability of heading to each exit


@dataclass
class SceneSpec:
    room_w: float
    room_h: float
    exits: list
    obstacles: list = field(default_factory=list)       # rectangles (x0, y0, x1, y1)
    groups: list = field(default_factory=list)
    gate_events: list = field(default_factory=list)     # (t, exit_index, "close" | "open")
    hurry: float = 1.2          # mean desired speed [m/s], hidden
    speed_sd: float = 0.15
    radius: float = 0.18
    time_gap: float = 0.6
    max_time: float = 180.0
    seed: int = 0
    model: str = "CollisionFreeSpeedModel"
    unstick: bool = True        # let stuck agents squeeze past (see sim/unstick.py)
    family: str = "hall"        # "hall", "platform" or "bottleneck"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        d = dict(d)
        d["exits"] = [ExitSpec(**e) for e in d["exits"]]
        d["groups"] = [GroupSpec(**g) for g in d["groups"]]
        return cls(**d)


# ---------------------------------------------------------------- geometry

def _side_box(spec: SceneSpec, side: str, s0: float, s1: float, o0: float, o1: float):
    """Box in wall-local coordinates: s along the wall, o outward from it."""
    w, h = spec.room_w, spec.room_h
    if side == "E":
        return shapely.box(w + o0, s0, w + o1, s1)
    if side == "W":
        return shapely.box(-o1, s0, -o0, s1)
    if side == "N":
        return shapely.box(s0, h + o0, s1, h + o1)
    return shapely.box(s0, -o1, s1, -o0)  # "S"


def exit_parts(spec: SceneSpec, e: ExitSpec) -> dict:
    lo, hi = e.centre - e.width / 2, e.centre + e.width / 2
    return {
        "stub": _side_box(spec, e.side, lo, hi, 0.0, STUB),
        "stage": _side_box(spec, e.side, lo + 0.05, hi - 0.05, STUB - 0.3, STUB - 0.05),
        # A closed gate is a barrier 2 cm short of each side: the simulator needs the
        # walkable area to stay connected, but nobody fits through 2 cm
        "barrier": _side_box(spec, e.side, lo + 0.02, hi - 0.02, 0.0, 0.1),
        "gate": _side_box(spec, e.side, lo, hi, 0.0, 0.1),
    }


def walkable_area(spec: SceneSpec, closed: set = frozenset()):
    parts = [shapely.box(0, 0, spec.room_w, spec.room_h)]
    parts += [exit_parts(spec, e)["stub"] for e in spec.exits]
    area = shapely.union_all(parts)
    for ob in spec.obstacles:
        area = area.difference(shapely.box(*ob))
    for i in closed:
        area = area.difference(exit_parts(spec, spec.exits[i])["barrier"])
    return area


# ---------------------------------------------------------------- random scenes

def _free_rect(rng, spec, size_x, size_y, margin, avoid):
    for _ in range(200):
        x0 = rng.uniform(margin, spec.room_w - margin - size_x)
        y0 = rng.uniform(margin, spec.room_h - margin - size_y)
        r = shapely.box(x0, y0, x0 + size_x, y0 + size_y)
        if not any(r.intersects(a) for a in avoid):
            return r
    return None


FAMILIES = {"hall": 0.4, "platform": 0.35, "bottleneck": 0.25}
URGENCY_REF = 1.05      # hurry level [m/s] at which the calibrated time gap / radius apply


def _r(v):
    return round(float(v), 2)


def _mix(rng, n_exits):
    return [round(float(p), 3) for p in rng.dirichlet(np.ones(n_exits))]


def sample_scene(rng: np.random.Generator, seed: int = 0, family: str | None = None) -> SceneSpec:
    """Random scene of the given family (or one drawn with FAMILIES probabilities)."""
    family = family or str(rng.choice(list(FAMILIES), p=list(FAMILIES.values())))
    body = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
    spec = SceneSpec(room_w=0, room_h=0, exits=[], hurry=_r(rng.uniform(0.8, 1.6)), seed=seed,
                     radius=body.get("radius", 0.18), time_gap=body.get("time_gap", 0.6),
                     speed_sd=body.get("speed_sd", 0.15),
                     model=body.get("model", "CollisionFreeSpeedModel"), family=family)
    # Urgency is more than walking speed: motivated crowds also keep less personal space
    # (shorter time gap, slightly squeezed bodies). Calibrated values hold at ~1.05 m/s.
    u = (spec.hurry - URGENCY_REF) / 0.55            # about -0.45 (calm) .. +1 (rushed)
    spec.time_gap = round(float(spec.time_gap * np.clip(1 - 0.6 * u, 0.5, 1.4)), 3)
    spec.radius = round(float(spec.radius * np.clip(1 - 0.15 * u, 0.8, 1.1)), 3)
    {"hall": _sample_hall, "platform": _sample_platform, "bottleneck": _sample_bottleneck}[family](rng, spec)

    # Gate actions: each gated exit toggles 0-2 times; some start closed
    for i, e in enumerate(spec.exits):
        if not e.gated:
            continue
        state = "open"
        if rng.random() < 0.2:
            spec.gate_events.append((0.0, i, "close"))
            state = "closed"
        for t in np.sort(rng.uniform(5, 70, int(rng.integers(0, 3)))):
            state = "closed" if state == "open" else "open"
            spec.gate_events.append((round(float(t), 1), i, "close" if state == "closed" else "open"))
    spec.gate_events.sort()
    return spec


def _sample_platform(rng, spec: SceneSpec):
    """Eindhoven-style platform: long strip, stairs at the ends, trains along the long sides
    unloading bursts of people through their doors."""
    spec.room_w, spec.room_h = _r(rng.uniform(11, 14)), _r(rng.uniform(4, 7))
    # Stairs A at the west end (never gated); usually stairs B at the east end
    w_a = _r(rng.uniform(1.5, min(3.0, spec.room_h - 1)))
    spec.exits.append(ExitSpec("W", _r(rng.uniform(w_a / 2 + 0.3, spec.room_h - w_a / 2 - 0.3)), w_a))
    tracks = ["N", "S"] if rng.random() < 0.4 else [str(rng.choice(["N", "S"]))]
    if rng.random() < 0.8:
        w_b = _r(rng.uniform(1.2, min(2.5, spec.room_h - 1)))
        spec.exits.append(ExitSpec("E", _r(rng.uniform(w_b / 2 + 0.3, spec.room_h - w_b / 2 - 0.3)),
                                   w_b, gated=bool(rng.random() < 0.6)))
    if len(tracks) == 1 and rng.random() < 0.3:   # side stairs on the non-track side
        side = "S" if tracks[0] == "N" else "N"
        w_c = _r(rng.uniform(1.2, 2.5))
        spec.exits.append(ExitSpec(side, _r(rng.uniform(3, spec.room_w - 3)), w_c,
                                   gated=bool(rng.random() < 0.6)))
    n_exits = len(spec.exits)

    # Central block (kiosk / stairwell), clear of the ends and the track edges
    if rng.random() < 0.5 and spec.room_h >= 5:
        bw, bh = rng.uniform(2, 4), rng.uniform(0.8, min(2.0, spec.room_h - 3.4))
        x0 = rng.uniform(3, spec.room_w - 3 - bw)
        y0 = rng.uniform(1.7, spec.room_h - 1.7 - bh)
        spec.obstacles.append((_r(x0), _r(y0), _r(x0 + bw), _r(y0 + bh)))

    # Trains: each track gets a train with 2-5 doors that open at the same moment
    for k, track in enumerate(tracks):
        if k > 0 and rng.random() < 0.5:
            continue
        t_arr, n_doors = float(rng.uniform(0, 25)), int(rng.integers(2, 6))
        n_total = int(rng.integers(40, 200))
        mix = _mix(rng, n_exits)
        door_x = np.linspace(1.5, spec.room_w - 2.7, n_doors) + rng.uniform(-0.5, 0.5, n_doors)
        # Door zone just inside the platform edge (more than a body radius from the wall)
        y0, y1 = (spec.room_h - 1.1, spec.room_h - 0.3) if track == "N" else (0.3, 1.1)
        for x in door_x:
            x = float(np.clip(x, 0.3, spec.room_w - 1.5))
            spec.groups.append(GroupSpec("burst", (_r(x), _r(y0), _r(x + 1.2), _r(y1)),
                                         round(t_arr + rng.uniform(0, 1.5), 1), _r(rng.uniform(5, 12)),
                                         max(5, n_total // n_doors), mix))

    # People already waiting on the platform, spread along it
    if rng.random() < 0.5:
        x0 = rng.uniform(1, spec.room_w - 5)
        zone = (_r(x0), _r(1.2), _r(x0 + rng.uniform(2, 4)), _r(spec.room_h - 1.2))
        if not any(shapely.box(*zone).intersects(shapely.box(*o)) for o in spec.obstacles):
            area = shapely.box(*zone).area
            spec.groups.append(GroupSpec("waiting", zone, 0.0, 0.0,
                                         int(rng.integers(5, max(6, int(area * 1.2)))), _mix(rng, n_exits)))


def _sample_bottleneck(rng, spec: SceneSpec):
    """Jülich-style bottleneck: a waiting area, a dividing wall with a gap (doorway or short
    corridor), the main exit beyond it, sometimes a second gated exit in the waiting area."""
    spec.room_w, spec.room_h = _r(rng.uniform(8, 14)), _r(rng.uniform(10, 14))
    gap_w, gap_len = rng.uniform(0.8, 3.0), rng.uniform(0.2, 3.0)
    y_d = rng.uniform(0.5, 0.62) * spec.room_h
    y_top = min(y_d + gap_len, spec.room_h - 2.0)
    gx = rng.uniform(gap_w / 2 + 1.0, spec.room_w - gap_w / 2 - 1.0)
    spec.obstacles += [(0.0, _r(y_d), _r(gx - gap_w / 2), _r(y_top)),
                       (_r(gx + gap_w / 2), _r(y_d), spec.room_w, _r(y_top))]
    w_a = _r(rng.uniform(1.5, 3.0))
    spec.exits.append(ExitSpec("N", _r(rng.uniform(w_a / 2 + 0.5, spec.room_w - w_a / 2 - 0.5)), w_a))
    if rng.random() < 0.5:   # alternative exit in the waiting area, usually gated
        w_b = _r(rng.uniform(1.0, 2.0))
        spec.exits.append(ExitSpec(str(rng.choice(["E", "W"])),
                                   _r(rng.uniform(w_b / 2 + 0.5, y_d - w_b / 2 - 0.5)), w_b,
                                   gated=bool(rng.random() < 0.7)))
    n_exits = len(spec.exits)

    # Dense waiting crowd (up to ~3 people per m^2), sometimes plus a stream from the back
    zw, zh = rng.uniform(4, min(9, spec.room_w - 1)), rng.uniform(3, max(3.1, y_d - 1.2))
    x0 = rng.uniform(0.5, spec.room_w - 0.5 - zw)
    zone = (_r(x0), _r(0.4), _r(x0 + zw), _r(0.4 + zh))
    spec.groups.append(GroupSpec("waiting", zone, 0.0, 0.0,
                                 int(rng.integers(80, max(81, int(zw * zh * 3.5)))), _mix(rng, n_exits)))
    if rng.random() < 0.4:
        sx = rng.uniform(0.3, spec.room_w - 1.8)
        spec.groups.append(GroupSpec("stream", (_r(sx), 0.3, _r(sx + 1.5), 1.8),
                                     _r(rng.uniform(5, 25)), _r(rng.uniform(30, 80)),
                                     int(rng.integers(20, 80)), _mix(rng, n_exits)))


def _sample_hall(rng, spec: SceneSpec):
    """Rectangular room, 2-3 exits on different walls, optional obstacles, mixed arrivals."""
    spec.room_w, spec.room_h = _r(rng.uniform(8, 14)), _r(rng.uniform(8, 14))

    # Exits on distinct walls; the first one is never gated so there's always a way out
    n_exits = int(rng.choice([2, 3], p=[0.5, 0.5]))
    for k, side in enumerate(rng.choice(["N", "S", "E", "W"], n_exits, replace=False)):
        length = spec.room_w if side in "NS" else spec.room_h
        width = round(rng.uniform(0.8, 2.5), 2)
        centre = round(rng.uniform(width / 2 + 0.5, length - width / 2 - 0.5), 2)
        spec.exits.append(ExitSpec(str(side), centre, width, gated=bool(k > 0 and rng.random() < 0.7)))

    # Doorway approaches stay clear of obstacles
    keep_clear = [exit_parts(spec, e)["stub"].buffer(2.0) for e in spec.exits]
    for _ in range(int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))):
        r = _free_rect(rng, spec, rng.uniform(1, 3), rng.uniform(0.5, 2), 1.5,
                       keep_clear + [shapely.box(*o).buffer(1.5) for o in spec.obstacles])
        if r is not None:
            spec.obstacles.append(tuple(round(v, 2) for v in r.bounds))

    occupied = [shapely.box(*o).buffer(0.3) for o in spec.obstacles] + \
               [exit_parts(spec, e)["stub"].buffer(1.0) for e in spec.exits]
    for _ in range(int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))):
        kind = str(rng.choice(["waiting", "burst", "stream"]))
        if kind == "waiting":
            zone = _free_rect(rng, spec, rng.uniform(2.5, 5), rng.uniform(2, 4), 0.4, occupied)
            n, duration = int(rng.integers(20, 90)), 0.0
        elif kind == "burst":   # people stepping off a train along a wall
            along, depth = rng.uniform(3, min(8, spec.room_w - 1)), 1.2
            zone = _free_rect(rng, spec, along, depth, 0.3, occupied)
            n, duration = int(rng.integers(40, 150)), float(rng.uniform(5, 15))
        else:
            zone = _free_rect(rng, spec, 1.5, 1.5, 0.3, occupied)
            n, duration = int(rng.integers(20, 80)), float(rng.uniform(30, 90))
        if zone is None:
            continue
        occupied.append(zone)
        spec.groups.append(GroupSpec(kind=kind, zone=tuple(round(v, 2) for v in zone.bounds),
                                     t_start=round(float(rng.uniform(0, 30)), 1),
                                     duration=round(duration, 1), n=n,
                                     dest_mix=[round(float(p), 3) for p in rng.dirichlet(np.ones(n_exits))]))


# ---------------------------------------------------------------- running

def _spawn_schedule(spec: SceneSpec, rng) -> list:
    """(time, x, y, group, intended_exit, desired_speed, failed_tries) for every person, by time."""
    out = []
    for gi, g in enumerate(spec.groups):
        x0, y0, x1, y1 = g.zone
        if g.kind == "stream":
            times = g.t_start + np.sort(rng.uniform(0, g.duration, g.n))
        elif g.kind == "burst":
            times = g.t_start + np.sort(rng.exponential(g.duration / 3, g.n).clip(0, g.duration))
        else:
            times = np.full(g.n, g.t_start)
        dests = rng.choice(len(spec.exits), g.n, p=np.array(g.dest_mix) / sum(g.dest_mix))
        speeds = np.clip(rng.normal(spec.hurry, spec.speed_sd, g.n), 0.4, 2.5)
        xs, ys = rng.uniform(x0, x1, g.n), rng.uniform(y0, y1, g.n)
        out += list(zip(times, xs, ys, [gi] * g.n, dests, speeds, [0] * g.n))
    return sorted(out, key=lambda r: r[0])


def run_scene(spec: SceneSpec) -> dict:
    """Simulate a scene. Returns trajectories, per-person hidden info and gate states."""
    rng = np.random.default_rng(spec.seed)
    model = getattr(jps, spec.model)
    agent_params = getattr(jps, spec.model + "AgentParameters")
    sim = jps.Simulation(model=model(), geometry=walkable_area(spec), dt=DT)
    parts = [exit_parts(spec, e) for e in spec.exits]
    stages = [sim.add_exit_stage(list(p["stage"].exterior.coords)[:-1]) for p in parts]
    journeys = [sim.add_journey(jps.JourneyDescription([s])) for s in stages]
    centres = np.array([[p["stage"].centroid.x, p["stage"].centroid.y] for p in parts])

    closed: set = set()
    pending_gates = list(spec.gate_events)
    pending = _spawn_schedule(spec, rng)
    agents = {}       # jupedsim id -> dict(intended, target, group, speed, spawn_t)
    target_of = {}    # jupedsim id -> current target exit
    record_every = round(1 / (FPS_OUT * DT))
    rows, gate_rows, delayed = [], [], 0
    unstick = Unstick(sim, spec.radius, spec.time_gap) if spec.unstick else None
    exited_via = {}   # jupedsim id -> exit actually used

    def nearest_open(pos, wanted):
        if wanted not in closed:
            return wanted
        d = np.linalg.norm(centres - np.asarray(pos), axis=1)
        d[list(closed)] = np.inf
        return int(np.argmin(d))

    def retarget_all():
        for a in sim.agents():
            info = agents[a.id]
            t = target_of[a.id]
            if t in closed and shapely.contains_xy(parts[t]["stub"], *a.position):
                continue   # already past the gate when it closed: keep walking out
            new = nearest_open(a.position, info["intended"])
            if new != target_of[a.id]:
                sim.switch_agent_journey(agent_id=a.id, journey_id=journeys[new], stage_id=stages[new])
                target_of[a.id] = new

    t0 = time.perf_counter()
    while sim.elapsed_time() < spec.max_time and (pending or pending_gates or sim.agent_count()):
        now = sim.elapsed_time()

        # Gate actions. A gate only closes once nobody stands in the doorway
        while pending_gates and pending_gates[0][0] <= now:
            t_ev, i, what = pending_gates[0]
            new_closed = closed | {i} if what == "close" else closed - {i}
            try:
                sim.switch_geometry(walkable_area(spec, new_closed))
            except Exception:
                pending_gates[0] = (now + 0.2, i, what)   # doorway occupied: try again shortly
                break
            pending_gates.pop(0)
            closed = new_closed
            retarget_all()

        # Arrivals
        retry = []
        while pending and pending[0][0] <= now:
            t_sp, x, y, gi, dest, v0, tries = pending.pop(0)
            target = nearest_open((x, y), int(dest))
            try:
                aid = sim.add_agent(agent_params(
                    journey_id=journeys[target], stage_id=stages[target], position=(x, y),
                    desired_speed=float(v0), radius=spec.radius, time_gap=spec.time_gap))
            except Exception:   # spot taken: try a moment later
                delayed += 1
                tries += 1
                if tries % 10 == 0:   # keeps failing (e.g. too close to a wall): new spot in the zone
                    zx0, zy0, zx1, zy1 = spec.groups[gi].zone
                    x, y = float(rng.uniform(zx0, zx1)), float(rng.uniform(zy0, zy1))
                retry.append((now + 0.2, x, y, gi, dest, v0, tries))
                continue
            agents[aid] = dict(intended=int(dest), group=gi, speed=float(v0), spawn_t=now)
            target_of[aid] = target
        pending = sorted(retry + pending, key=lambda r: r[0])

        if sim.iteration_count() % record_every == 0:
            t = round(sim.elapsed_time(), 2)
            rows += [(a.id, t, a.position[0], a.position[1], target_of[a.id]) for a in sim.agents()]
            gate_rows.append([t] + [int(i in closed) for i in range(len(spec.exits))])
            # Anyone standing on an open exit leaves through it, whatever they were aiming
            # for (the simulator would otherwise let them block that doorway forever)
            for a in sim.agents():
                for i, p in enumerate(parts):
                    if i != target_of[a.id] and i not in closed and \
                            shapely.contains_xy(p["stage"], *a.position):
                        sim.mark_agent_for_removal(a.id)
                        exited_via[a.id] = i
        if unstick:
            unstick.step()
        sim.iterate()
        for rid in sim.removed_agents():
            exited_via.setdefault(rid, target_of[rid])
            agents[rid]["exit_t"] = round(sim.elapsed_time(), 2)

    traj = pd.DataFrame(rows, columns=["id", "t", "x", "y", "target"])
    people = pd.DataFrame.from_dict(agents, orient="index").rename_axis("id").reset_index()
    people["exited_via"] = people["id"].map(exited_via).astype("Int64")   # <NA>: still inside
    gates = pd.DataFrame(gate_rows, columns=["t"] + [f"closed_{i}" for i in range(len(spec.exits))])
    info = dict(people=len(agents), never_spawned=len(pending), still_inside=sim.agent_count(),
                sim_seconds=round(sim.elapsed_time(), 1),
                compute_seconds=round(time.perf_counter() - t0, 2), delayed_spawns=delayed,
                squeezes=unstick.count if unstick else 0)
    return dict(traj=traj, people=people, gates=gates, info=info)
