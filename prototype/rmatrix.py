import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

mp.mp.dps = 40


class LagrangeLegendreMesh:
    def __init__(self, num_points: int, channel_radius: float):
        self.num_points = num_points
        self.channel_radius = channel_radius
        self._initialize_mesh()

    def _initialize_mesh(self):
        N, a = self.num_points, self.channel_radius
        x_raw, w_raw = leggauss(N)
        x = 0.5 * (x_raw + 1.0)
        w = 0.5 * w_raw
        self.nodes = x
        self.weights = w
        self.radii = x * a
        T = np.zeros((N, N))
        np.fill_diagonal(
            T, (4 * N * (N + 1) + 3 + (1 - 6 * x) / (x * (1 - x))) / (3 * x * (1 - x))
        )
        i, j = np.triu_indices(N, k=1)
        val = (
            N * (N + 1)
            + 1
            + (x[i] + x[j] - 2 * x[i] * x[j]) / (x[i] - x[j]) ** 2
            - 1 / (1 - x[i])
            - 1 / (1 - x[j])
        )
        val /= np.sqrt(x[i] * (1 - x[i]) * x[j] * (1 - x[j]))
        val *= np.where((i + j) % 2 == 1, -1.0, 1.0)
        T[i, j] = val
        T[j, i] = val
        T /= a**2
        self.kinetic = T
        q2 = 1.0 / np.sqrt(x * (1 - x))
        q1 = np.zeros_like(q2)
        if N % 2 == 1:
            q2 = -q2
        q1[::2] *= -1
        q2[::2] *= -1
        q1 /= np.sqrt(a)
        q2 /= np.sqrt(a)
        self.q1 = q1
        self.q2 = q2


class CoupledRMatrixSolver:
    def __init__(self, mesh, l_values, k_values, eta_values=None):
        self.mesh = mesh
        self.N = mesh.num_points
        self.a = mesh.channel_radius
        self.l_values = np.atleast_1d(np.array(l_values, dtype=int))
        self.k_values = np.atleast_1d(np.array(k_values, dtype=float))

        if eta_values is None:
            self.eta_values = np.zeros_like(k_values)
        else:
            self.eta_values = np.atleast_1d(np.array(eta_values, dtype=float))

        self.Nc = len(self.l_values)
        Q = np.zeros((self.Nc * self.N, self.Nc))
        Q1 = np.zeros_like(Q)
        for c in range(self.Nc):
            Q[c * self.N : (c + 1) * self.N, c] = self.mesh.q2
            Q1[c * self.N : (c + 1) * self.N, c] = self.mesh.q1
        self.Q = Q
        self.Q1 = Q1
        H_plus = np.zeros(self.Nc, dtype=complex)
        H_plus_p = np.zeros(self.Nc, dtype=complex)
        H_minus = np.zeros(self.Nc, dtype=complex)
        H_minus_p = np.zeros(self.Nc, dtype=complex)
        for c in range(self.Nc):
            k = self.k_values[c]
            l = int(self.l_values[c])
            eta = self.eta_values[c]
            G = mp.coulombg(l, eta, k * self.a)
            F = mp.coulombf(l, eta, k * self.a)
            dG = mp.diff(lambda z: mp.coulombg(l, eta, z), k * self.a)
            dF = mp.diff(lambda z: mp.coulombf(l, eta, z), k * self.a)
            Hp = complex(G) + 1j * complex(F)
            Hp_p = (complex(dG) + 1j * complex(dF)) * (k * self.a)
            Hm = complex(G) - 1j * complex(F)
            Hm_p = (complex(dG) - 1j * complex(dF)) * (k * self.a)
            H_plus[c] = Hp
            H_plus_p[c] = Hp_p
            H_minus[c] = Hm
            H_minus_p[c] = Hm_p
        self.Hp = np.diag(H_plus)
        self.Hp_p = np.diag(H_plus_p)
        self.Hm = np.diag(H_minus)
        self.Hm_p = np.diag(H_minus_p)

    def assemble_C(self, potential_blocks):
        Nc, N = self.Nc, self.N
        T = self.mesh.kinetic
        C = np.zeros((Nc * N, Nc * N), dtype=float)
        for c in range(Nc):
            for c2 in range(Nc):
                block = T.copy() if c == c2 else np.zeros_like(T)
                block += potential_blocks[c, c2]
                if c == c2:
                    block[np.diag_indices(N)] -= self.k_values[c] ** 2
                C[c * N : (c + 1) * N, c2 * N : (c2 + 1) * N] = block
        return C

    def compute_S_and_phases(self, potential_blocks):
        C = self.assemble_C(potential_blocks)
        # Cinv = np.linalg.inv(C)
        # M11 = self.Q.T @ (Cinv @ self.Q)

        M11 = self.Q.T @ np.linalg.solve(C, self.Q)
        # M12 = self.Q1.T @ (Cinv @ self.Q)
        # M22 = self.Q1.T @ (Cinv @ self.Q1)

        # try:
        #    M22_inv = np.linalg.inv(M22)
        # except:
        #    M22_inv = np.linalg.pinv(M22)
        crma0 = M11  # - M12.T @ (M22_inv @ M12)
        Rmat = crma0 / self.a
        A = self.Hm - (Rmat @ self.Hm_p)
        B = self.Hp - (Rmat @ self.Hp_p)
        S = A @ np.linalg.inv(B)
        eigvals = np.linalg.eigvals(S)
        phases = 0.5 * np.angle(eigvals)
        return S, np.degrees(phases)
