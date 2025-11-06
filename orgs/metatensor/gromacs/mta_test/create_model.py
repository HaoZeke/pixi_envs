#!/usr/bin/env python3

import metatomic_lj_test
import torch
from torch import nn
from typing import Optional


def displacement_vectors(positions, box: Optional[torch.Tensor] = None):

    disp = positions[None, :, :] - positions[:, None, :]

    if box is not None:
        inv_box = torch.inverse(box)
        disp_frac = torch.einsum("nij,jk->nik", disp, inv_box)
        disp_frac = disp_frac - torch.round(disp_frac)
        disp = torch.einsum("nij,jk->nik", disp_frac, box)

    # Extract only i < j pairs
    N = len(positions)
    indices = torch.triu_indices(N, N, offset=1)
    iu = indices[0]
    ju = indices[1]

    disp_pairs = disp[iu, ju]  # (N_dist, 3)

    return disp_pairs


class GmxNNPotModelWrapper(nn.Module):
    def __init__(self, cutoff, epsilon, sigma):
        super().__init__()

        self._cutoff = cutoff
        self._epsilon = epsilon
        self._sigma = sigma

    def forward(self, positions, atomic_numbers, box: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):
        distances = displacement_vectors(positions, box)

        sigma_r_6 = (self._sigma / torch.linalg.vector_norm(distances, dim=1)) ** 6
        sigma_r_12 = sigma_r_6 * sigma_r_6

        return torch.sum(4.0 * self._epsilon * (sigma_r_12 - sigma_r_6))
    

atomic_type = 18
cutoff = 0.5
sigma = 0.33646
epsilon = 0.94191

mst_model = metatomic_lj_test.lennard_jones_model(
    atomic_type=atomic_type,
    cutoff=cutoff,
    sigma=sigma,
    epsilon=epsilon,
    length_unit="nm",
    energy_unit="kj/mol",
    with_extension=False,
)

gmx_model = GmxNNPotModelWrapper(cutoff=cutoff, sigma=sigma, epsilon=epsilon)

mst_model.save("model.pt",)
torch.jit.script(gmx_model).save("gmx-lj.pt")
