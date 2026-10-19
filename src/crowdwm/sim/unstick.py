"""Break collision-free-speed-model deadlocks by letting stuck agents "squeeze".

The CollisionFreeSpeedModel makes an agent stop dead when someone is directly ahead
and never sidestep, so at narrow doorways crowds can freeze into a permanent arch.
Real people turn their shoulders and edge through. We mimic that: an agent that has
moved less than MIN_MOVE in WINDOW seconds temporarily gets a smaller radius and
time gap, then returns to normal.
"""

from collections import deque

import numpy as np

WINDOW = 5.0            # seconds without real movement before an agent counts as stuck
MIN_MOVE = 0.10         # [m]
SQUEEZE_S = 2.0         # how long the squeeze lasts
SQUEEZE_RADIUS = 0.12
SQUEEZE_TIME_GAP = 0.1
CHECK_EVERY = 0.5       # seconds between checks


class Unstick:
    def __init__(self, sim, radius: float, time_gap: float):
        self.sim, self.radius, self.time_gap = sim, radius, time_gap
        self.history = {}           # id -> deque of (t, x, y)
        self.squeezed = {}          # id -> time the squeeze ends
        self.every = max(1, round(CHECK_EVERY / sim.delta_time()))
        self.count = 0

    def step(self):
        if self.sim.iteration_count() % self.every:
            return
        now = self.sim.elapsed_time()
        alive = set()
        for a in self.sim.agents():
            alive.add(a.id)
            h = self.history.setdefault(a.id, deque(maxlen=int(WINDOW / CHECK_EVERY) + 1))
            h.append((now, *a.position))
            if a.id in self.squeezed:
                if now >= self.squeezed[a.id]:
                    a.model.radius, a.model.time_gap = self.radius, self.time_gap
                    del self.squeezed[a.id]
                    h.clear()
                continue
            t_old, x_old, y_old = h[0]
            if now - t_old >= WINDOW and np.hypot(a.position[0] - x_old, a.position[1] - y_old) < MIN_MOVE:
                a.model.radius, a.model.time_gap = SQUEEZE_RADIUS, SQUEEZE_TIME_GAP
                self.squeezed[a.id] = now + SQUEEZE_S
                self.count += 1
        for gone in set(self.history) - alive:
            self.history.pop(gone, None)
            self.squeezed.pop(gone, None)
