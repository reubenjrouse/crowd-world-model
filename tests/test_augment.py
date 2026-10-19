"""Rotating/mirroring a training window must keep all its grids consistent."""

import torch

from crowdwm.data.sim_dataset import augment


def test_augment_keeps_layers_aligned():
    g = torch.Generator().manual_seed(0)
    b, length, e, s = 6, 4, 3, 16
    dest = torch.rand(b, length, e, s, s, generator=g)
    walls = (torch.rand(b, 1, s, s, generator=g) > 0.8).float()
    B = dict(density=dest.sum(2, keepdim=True), dest=dest, static=torch.cat([walls, walls.repeat(1, 3, 1, 1)], 1),
             gates=walls[:, None].repeat(1, length, 1, 1, 1), hurry=torch.ones(b), exit_mask=torch.ones(b, e, dtype=bool))
    A = augment(B, torch.Generator().manual_seed(1))
    # per-exit layers still add up to the crowd layer, and headcounts are unchanged
    torch.testing.assert_close(A["dest"].sum(2, keepdim=True), A["density"])
    torch.testing.assert_close(A["density"].sum((-1, -2)), B["density"].sum((-1, -2)))
    # walls, exits and gates moved together (all built from the same mask here)
    torch.testing.assert_close(A["static"][:, 0], A["static"][:, 1])
    torch.testing.assert_close(A["gates"][:, 0, 0], A["static"][:, 0])
    # something actually changed
    assert not torch.equal(A["density"], B["density"])
    # original batch untouched
    torch.testing.assert_close(B["dest"].sum(2, keepdim=True), B["density"])
