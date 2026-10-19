"""Moving-people readout of the world model's forecast.

The model predicts, for every 25 cm cell and step, the share of its people that move to
each nearby cell, the chance of leaving at exits, and how many new people appear. This
turns that into individual dots: start from the real positions at the last watched frame
and move each dot with its cell's move probabilities (a little momentum for smooth paths);
dots never enter walls or closed gates, leave at exits with the model's leaving chance,
and new dots appear where the model expects arrivals. It is one sample of the model's
prediction drawn as people, not a per-person forecast.
"""

import numpy as np


def move_people(xy, flux, leave, blocked, x0, y0, source=None, cell=0.25, seed=0):
    """xy: (N, 2) start positions. flux[t]: (k*k, H, W) move probabilities; leave[t]: (H, W);
    blocked[t]: (H, W) bool for the frame being moved into; source[t]: (H, W) expected new
    people (optional). Returns a list over imagined frames of (M, 2) positions (alive only)."""
    rng = np.random.default_rng(seed)
    k = int(round(np.sqrt(flux[0].shape[0])))
    offs = np.array([(j % k - k // 2, j // k - k // 2) for j in range(k * k)], float) * cell   # (dx, dy)
    pos, vel = xy.astype(float).copy(), np.zeros((len(xy), 2))
    out = []
    for t, (p, lv, blk) in enumerate(zip(flux, leave, blocked)):
        if len(pos):
            col = np.clip(((pos[:, 0] - x0) / cell).astype(int), 0, 63)
            row = np.clip(((pos[:, 1] - y0) / cell).astype(int), 0, 63)
            probs = p[:, row, col].T
            probs = probs / probs.sum(1, keepdims=True)
            choice = (probs.cumsum(1) > rng.random((len(pos), 1))).argmax(1)
            step = offs[choice]
            vel = 0.5 * vel + 0.5 * step
            new = pos + vel + rng.normal(0, 0.02, pos.shape)
            ncol = np.clip(((new[:, 0] - x0) / cell).astype(int), 0, 63)
            nrow = np.clip(((new[:, 1] - y0) / cell).astype(int), 0, 63)
            bad = blk[nrow, ncol]
            new[bad] = pos[bad] + step[bad]          # the sampled move itself always has a clear path
            vel[bad] = step[bad]
            stay = rng.random(len(pos)) >= lv[row, col]
            pos, vel = new[stay], vel[stay]
        if source is not None:                       # new arrivals, where the model expects them
            n_new = rng.poisson(source[t])
            r, c = np.nonzero(n_new)
            if len(r):
                reps = n_new[r, c]
                rr, cc = np.repeat(r, reps), np.repeat(c, reps)
                born = np.stack([x0 + (cc + rng.random(len(cc))) * cell, y0 + (rr + rng.random(len(rr))) * cell], 1)
                pos = np.concatenate([pos, born])
                vel = np.concatenate([vel, np.zeros_like(born)])
        out.append(pos.copy())
    return out
