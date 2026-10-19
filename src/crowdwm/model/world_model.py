"""Action-conditioned crowd world model.

    frame (density, walls, exits, gates, arrivals) --encoder--> 16x16 features
    features + memory --ConvGRU--> new memory            (recurrent latent state)
    [memory, z] + frame --decoder--> per-cell outputs:
        flux logits   where each cell's people move (5x5 neighbourhood, incl. staying)
        sink          fraction leaving the scene at exit cells
        source        people appearing (arrivals)
        destination   which exit the people in each cell are heading to
    memory --pooled--> hurry (crowd's mean desired speed)

The next density is computed by moving people between cells according to the flux,
never into walls or closed gates, so headcount changes only through the explicit
sink and source terms (mass conservation by construction).

During the context frames the model is fed observed frames; afterwards it is fed its
own predictions ("imagination"), while gate states and scheduled arrivals come from the
plan.

Stochastic version (z_dim > 0, Stage 3): z is a 16x16xZ Gaussian "dice roll" per step for
what can't be predicted from the past (e.g. which way a group splits). In training, a
posterior sees the true next frame and the prior learns to match it (KL); in imagination
z is sampled from the prior, so repeated rollouts give different, individually sharp
futures. z_dim = 0 gives the deterministic Stage 1-2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowdwm.raster import MAX_EXITS

DENSITY_SCALE = 4.0     # input scaling for density (people per cell is ~0-2)


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        self.gates = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=k // 2)
        self.cand = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=k // 2)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


def block(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, padding=(k - 1) // 2 if s == 1 else 1),
                         nn.GroupNorm(8, o), nn.GELU())


class CrowdWorldModel(nn.Module):
    def __init__(self, hid: int = 128, enc: int = 96, flux_k: int = 5, z_dim: int = 0):
        super().__init__()
        # density, walls, exits, gates, arrivals, open camera edge (appended last so older
        # checkpoints upgrade with a zero-initialised extra input)
        in_ch = 1 + 1 + MAX_EXITS + 1 + 1 + 1
        self.in_ch, self.hid, self.k, self.z_dim = in_ch, hid, flux_k, z_dim
        self.encoder = nn.Sequential(
            block(in_ch, 48), nn.Conv2d(48, 64, 4, 2, 1), nn.GELU(),       # 64 -> 32
            block(64, 64), nn.Conv2d(64, enc, 4, 2, 1), nn.GELU(),         # 32 -> 16
            block(enc, enc))
        self.gru = ConvGRUCell(enc, hid)
        if z_dim:
            self.prior = nn.Sequential(block(hid, hid), nn.Conv2d(hid, 2 * z_dim, 1))
            self.posterior = nn.Sequential(block(hid + enc, hid), nn.Conv2d(hid, 2 * z_dim, 1))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hid + z_dim, 96, 4, 2, 1), nn.GELU(), block(96, 96),  # 16 -> 32
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GELU())                          # 32 -> 64
        self.fuse = nn.Sequential(block(64 + in_ch, 64), block(64, 64))
        self.head = nn.Conv2d(64, flux_k * flux_k + 2 + MAX_EXITS, 1)
        self.hurry_head = nn.Sequential(nn.Linear(hid, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("path", self._path_matrix(flux_k), persistent=False)
        self.arrivals = True    # False: forecast only the people already present (no source term)
        self.track_outflow = False   # True: record people leaving per exit slot in self.outflow
        self.outflow = []
        self.temperature = 1.0   # spread of sampled futures (stochastic model)
        self.record_flux = False  # True: keep per-step move probabilities in self.flux / self.leave
        self.flux, self.leave, self.source = [], [], []

    @staticmethod
    def _path_matrix(k):
        """path[j, i] = 1 if moving to neighbour j passes through neighbour i.

        A move must have a clear path, not just a clear landing cell; otherwise a 2-cell
        move jumps over 1-cell-thin walls and diagonal moves slip between wall cells.
        """
        pad = k // 2
        offsets = [(j // k - pad, j % k - pad) for j in range(k * k)]

        def steps(c):   # intermediate values along one axis
            return {0: [0], 1: [0, 1], -1: [0, -1], 2: [1], -2: [-1]}[c]

        m = torch.zeros(k * k, k * k)
        for j, (dy, dx) in enumerate(offsets):
            for iy in steps(dy):
                for ix in steps(dx):
                    if (iy, ix) not in [(0, 0), (dy, dx)]:
                        m[j, offsets.index((iy, ix))] = 1
        return m

    # ------------------------------------------------------------------ physics
    def transport(self, density, flux_logits, blocked):
        """Move people between cells. density (B,1,H,W); blocked (B,1,H,W) in {0,1}."""
        b, _, h, w = density.shape
        k, pad = self.k, self.k // 2
        # For every cell, which of its k*k neighbours are blocked (outside the grid counts)
        nb = F.unfold(F.pad(blocked, (pad,) * 4, value=1.0), k).view(b, k * k, h, w)
        # Blocked if the landing cell or any cell on the way is blocked
        nb = nb + torch.einsum("ji,bihw->bjhw", self.path.to(nb.dtype), nb)
        nb[:, (k * k) // 2] = 0                          # staying put is always allowed
        p = torch.softmax(flux_logits.float().masked_fill(nb > 0.5, float("-inf")), dim=1)
        if self.record_flux:     # per-cell move probabilities, for the moving-people view
            self.flux.append(p)
        moved = (density.float() * p).view(b, k * k, h * w)
        return F.fold(moved, (h, w), k, padding=pad)     # sum what arrives in each cell

    # ------------------------------------------------------------------ one step
    def frame(self, density, static, gates, arrivals):
        """static: walls, MAX_EXITS exit slots and (optionally) the open camera edge."""
        if static.shape[1] == 1 + MAX_EXITS:          # no open-edge layer given: none
            static = torch.cat([static, torch.zeros_like(static[:, :1])], 1)
        return torch.cat([density * DENSITY_SCALE, static[:, :1 + MAX_EXITS], gates, arrivals,
                          static[:, 1 + MAX_EXITS:2 + MAX_EXITS]], 1)

    def step(self, density, static, gates_now, gates_next, arrivals_now, h, next_frame=None,
             sample=True):
        """One step. next_frame (training only): the true next frame, seen by the posterior.
        Returns next density, destination logits, new memory, KL for this step (or None)."""
        frame = self.frame(density, static, gates_now, arrivals_now)
        h = self.gru(self.encoder(frame), h)
        kl, dec_in = None, h
        if self.z_dim:
            p_mu, p_logstd = self.prior(h).chunk(2, 1)
            p_std = F.softplus(p_logstd) + 0.05
            if next_frame is not None:
                q_mu, q_logstd = self.posterior(torch.cat([h, self.encoder(next_frame)], 1)).chunk(2, 1)
                q_std = F.softplus(q_logstd) + 0.05
                z = q_mu + q_std * torch.randn_like(q_std)
                # KL balancing (DreamerV3): train the prior toward the posterior more than
                # the reverse, so the posterior doesn't just collapse onto a weak prior
                kl = 0.8 * _kl(q_mu.detach(), q_std.detach(), p_mu, p_std) \
                    + 0.2 * _kl(q_mu, q_std, p_mu.detach(), p_std.detach())
            else:
                # temperature > 1 makes the dice rolls more varied (tuned on validation data)
                z = p_mu + self.temperature * p_std * torch.randn_like(p_std) if sample else p_mu
            dec_in = torch.cat([h, z], 1)
        out = self.head(self.fuse(torch.cat([self.up(dec_in), frame], 1)))
        kk = self.k * self.k
        flux, sink, source, dest = out[:, :kk], out[:, kk:kk + 1], out[:, kk + 1:kk + 2], out[:, kk + 2:]

        walls = static[:, :1]
        exit_slots = static[:, 1:1 + MAX_EXITS]
        # People can leave at exits and at the open edge of the camera view (real data)
        exits = torch.cat([exit_slots, static[:, 1 + MAX_EXITS:2 + MAX_EXITS]], 1).amax(1, keepdim=True)
        blocked = ((walls + gates_next) > 0.5).float()
        nxt = self.transport(density, flux, blocked)
        leaving = nxt * torch.sigmoid(sink.float()) * exits
        if self.record_flux:     # chance of leaving per cell (non-zero only at exits)
            self.leave.append(torch.sigmoid(sink.float()) * exits)
        if self.track_outflow:   # people leaving through each exit slot this step
            self.outflow.append((leaving * exit_slots).sum((2, 3)))
        nxt = nxt - leaving
        if self.arrivals:
            # Arrivals start near zero (~0.2 people per step over the whole grid) and must be learned
            new_people = F.softplus(source.float() - 10.0) * (1 - blocked)
            if self.record_flux:   # expected new people per cell this step
                self.source.append(new_people)
            nxt = nxt + new_people
        return nxt, dest, h, kl

    # ------------------------------------------------------------------ rollout
    def forward(self, density, static, gates, exit_mask, context: int, arrivals=None,
                use_posterior: bool = False, sample: bool = True):
        """density, gates, arrivals: (B, L, 1, H, W); static (B, 1+MAX_EXITS, H, W).

        Observed frames are fed for t < context, predictions afterwards.
        use_posterior (training a stochastic model): the true future frames inform z.
        sample=False uses the prior mean instead of a random draw.
        Returns pred density for frames 1..L-1 (B, L-1, 1, H, W), destination logits
        for frames 0..L-1 (B, L, MAX_EXITS, H, W), hurry estimate after the context (B,).
        The mean KL per latent value is left in self.kl (stochastic training only).
        """
        b, length, _, hh, ww = density.shape
        if arrivals is None:
            arrivals = torch.zeros_like(gates)
        h = density.new_zeros(b, self.hid, hh // 4, ww // 4)
        preds, dests, kls, hurry = [], [], [], None
        cur = density[:, 0]
        for t in range(length):
            if t < context:
                cur = density[:, t]
            t1 = min(t + 1, length - 1)
            nxt_frame = (self.frame(density[:, t1], static, gates[:, t1], arrivals[:, t1])
                         if use_posterior and self.z_dim and t < length - 1 else None)
            nxt, dest, h, kl = self.step(cur, static, gates[:, t], gates[:, t1], arrivals[:, t], h,
                                         next_frame=nxt_frame, sample=sample)
            if kl is not None:
                kls.append(kl)
            dests.append(dest.float().masked_fill(~exit_mask[:, :, None, None], -1e4))
            if t == context - 1:
                hurry = self.hurry_head(h.float().mean((2, 3))).squeeze(1) + 1.2
            if t < length - 1:
                preds.append(nxt)
                cur = nxt
        self.kl = torch.stack(kls).mean() if kls else None
        return torch.stack(preds, 1), torch.stack(dests, 1), hurry


def _kl(mu_q, std_q, mu_p, std_p):
    """KL(q || p) between diagonal Gaussians, averaged over all latent values."""
    return (torch.log(std_p / std_q) + (std_q ** 2 + (mu_q - mu_p) ** 2) / (2 * std_p ** 2) - 0.5).mean()


def load_checkpoint(path, device="cuda") -> "CrowdWorldModel":
    """Build the right model for a checkpoint (deterministic or stochastic) and load it,
    upgrading older checkpoints to the current input layers."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = CrowdWorldModel(z_dim=ckpt.get("args", {}).get("z_dim", 0) or 0).to(device)
    load_compatible(model, ckpt["model"])
    return model.eval()


def load_compatible(model: nn.Module, state: dict) -> list[str]:
    """Load weights from an older checkpoint into a model whose input layers grew
    (6 -> 7 input channels, or the decoder gaining z channels). New input slices start at
    zero, so the upgraded model initially behaves exactly like the old one. Layers that
    don't exist in the checkpoint (prior/posterior) keep their fresh initialisation.
    Returns the names of resized parameters."""
    own, resized = model.state_dict(), []
    for name, w in state.items():
        if name not in own:
            continue
        if own[name].shape == w.shape:
            own[name] = w
        elif own[name].dim() == w.dim():
            new = torch.zeros_like(own[name])
            new[tuple(slice(0, s) for s in w.shape)] = w
            own[name] = new
            resized.append(name)
    model.load_state_dict(own)
    return resized
