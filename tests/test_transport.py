"""The world model's transport step must conserve people and respect walls."""

import torch

from crowdwm.model.world_model import CrowdWorldModel


def test_no_tunnelling_through_thin_wall():
    m = CrowdWorldModel()
    density = torch.zeros(1, 1, 16, 16)
    density[0, 0, 8, 5] = 1.0                    # one person just left of the wall
    blocked = torch.zeros(1, 1, 16, 16)
    blocked[0, 0, :, 6] = 1.0                    # wall one cell thick
    flux = torch.full((1, 25, 16, 16), -10.0)
    flux[:, 2 * 5 + 4] = 10.0                    # everyone wants to move 2 cells right (over the wall)
    out = m.transport(density, flux, blocked)
    assert out[0, 0, :, 7:].sum() < 1e-6         # nobody ends up on the far side
    torch.testing.assert_close(out.sum(), torch.tensor(1.0))


def test_mass_conserved_with_random_flux():
    m = CrowdWorldModel()
    g = torch.Generator().manual_seed(0)
    density = torch.rand(2, 1, 32, 32, generator=g)
    blocked = (torch.rand(2, 1, 32, 32, generator=g) > 0.8).float()
    density = density * (1 - blocked)
    flux = torch.randn(2, 25, 32, 32, generator=g)
    out = m.transport(density, flux, blocked)
    torch.testing.assert_close(out.sum((1, 2, 3)), density.sum((1, 2, 3)))
    assert (out * blocked).sum() < 1e-5          # nobody moves into a wall
