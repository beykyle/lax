import unittest

import numpy as np

from rmatrix import CoupledRMatrixSolver, LagrangeLegendreMesh


def yamaguchi_potential_matrix_blocks(radii, weights, a):
    alpha = 0.2316053
    beta = 1.3918324
    ri, rj = np.meshgrid(radii, radii, indexing="ij")
    wi, wj = np.meshgrid(weights, weights, indexing="ij")
    V = -2.0 * beta * (alpha + beta) ** 2 * np.exp(-beta * (ri + rj))
    V = V * np.sqrt(wi * wj) * a
    return V.reshape((1, 1) + V.shape)


class TestMultiChannelSingleYamaguchi(unittest.TestCase):
    def test_single_channel_reproduces_baye(self):
        HBAR2_OVER_2MU = 41.472
        cases = [
            (8.0, 20, 0.1, -15.078689),
            (8.0, 20, 10.0, 85.634560),
            (15.0, 20, 0.1, -15.078689),
            (15.0, 20, 10.0, 85.634560),
        ]
        for a, N, E, ref in cases:
            mesh = LagrangeLegendreMesh(N, a)
            blocks = yamaguchi_potential_matrix_blocks(
                mesh.radii, mesh.weights, mesh.channel_radius
            )
            solver = CoupledRMatrixSolver(
                mesh, l_values=[0], k_values=[np.sqrt(E / HBAR2_OVER_2MU)]
            )
            S, phases = solver.compute_S_and_phases(blocks)
            delta = phases[0]
            diff = abs(delta - ref)
            print(f"a={a} N={N} E={E} δ={delta:.6f} ref={ref:.6f} Δ={diff:.3e}")
            self.assertLess(diff, 1e-2)
