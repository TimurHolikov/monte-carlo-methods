# ising_core.py
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# -----------------------------
# Geometry
# -----------------------------

def build_bonds(Lx: int, Ly: int) -> np.ndarray:
    """
    bonds has shape (2N,2). Each site contributes two undirected bonds stored once:
    (n -> right) and (n -> down), with periodic boundary conditions.
    """
    N = Lx * Ly
    bonds = np.empty((2 * N, 2), dtype=np.int32)
    k = 0
    for x in range(Lx):
        for y in range(Ly):
            n = x * Ly + y
            bonds[k] = (n, x * Ly + ((y + 1) % Ly)); k += 1
            bonds[k] = (n, ((x + 1) % Lx) * Ly + y); k += 1
    return bonds


def build_neighbors(Lx: int, Ly: int) -> np.ndarray:
    """nbrs[n,4] = (right,left,down,up) with PBC."""
    N = Lx * Ly
    nbrs = np.empty((N, 4), dtype=np.int32)
    for x in range(Lx):
        for y in range(Ly):
            n = x * Ly + y
            nbrs[n, 0] = x * Ly + ((y + 1) % Ly)          # right
            nbrs[n, 1] = x * Ly + ((y - 1) % Ly)          # left
            nbrs[n, 2] = ((x + 1) % Lx) * Ly + y          # down
            nbrs[n, 3] = ((x - 1) % Lx) * Ly + y          # up
    return nbrs


# -----------------------------
# Observables
# -----------------------------

def energy(spins: np.ndarray, bonds: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    """
    Ising Hamiltonian (classical):
        H = -J * sum_<ij> s_i s_j - h * sum_i s_i
    bonds counts each NN bond once (right+down).
    """
    i = bonds[:, 0]
    j = bonds[:, 1]
    bond_term = np.sum(spins[i] * spins[j])
    field_term = np.sum(spins)
    return float(-J * bond_term - h * field_term)


def energy_density(spins: np.ndarray, bonds: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    return energy(spins, bonds, J=J, h=h) / spins.size


def magnetization(spins: np.ndarray) -> float:
    return float(np.mean(spins))


def correlation_map_fft(spins_1d: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """
    Two-point correlation C(dx,dy) via FFT:
      C = ifft(|fft(s)|^2)/N
    returns shape (Lx, Ly).
    """
    s = spins_1d.reshape(Lx, Ly).astype(float)
    N = s.size
    F = np.fft.fft2(s)
    C = np.fft.ifft2(np.abs(F) ** 2).real / N
    return C


def radial_average_corr(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial binning with periodic metric (min distance).
    """
    Lx, Ly = C.shape
    r_max = int(np.floor(np.sqrt((Lx//2)**2 + (Ly//2)**2)))
    sums = np.zeros(r_max + 1, dtype=float)
    counts = np.zeros(r_max + 1, dtype=int)

    for dx in range(Lx):
        ddx = min(dx, Lx - dx)
        for dy in range(Ly):
            ddy = min(dy, Ly - dy)
            r = int(np.rint(np.sqrt(ddx**2 + ddy**2)))
            if r <= r_max:
                sums[r] += C[dx, dy]
                counts[r] += 1

    Cr = np.zeros_like(sums)
    m = counts > 0
    Cr[m] = sums[m] / counts[m]
    return np.arange(r_max + 1), Cr


# -----------------------------
# Autocorrelation / tau_int
# -----------------------------

def autocorr_norm(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation rho(lag) with rho(0)=1.
    O(N*max_lag) simple estimator (good enough for project sizes).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.mean(x * x)
    rho = np.empty(max_lag + 1, dtype=float)
    rho[0] = 1.0
    if var == 0.0:
        rho[1:] = 0.0
        return rho
    for lag in range(1, max_lag + 1):
        rho[lag] = np.mean(x[:-lag] * x[lag:]) / var
    return rho


def tau_int_from_rho(rho: np.ndarray) -> float:
    """
    Integrated autocorrelation time via a simple 'first non-positive' cutoff:
      tau_int = 1/2 + sum_{lag>=1} rho(lag)   until rho<=0.
    """
    s = 0.5
    for lag in range(1, len(rho)):
        if rho[lag] <= 0:
            break
        s += rho[lag]
    return float(s)


# -----------------------------
# Updates: SW and Metropolis
# -----------------------------

def sw_update(spins: np.ndarray, bonds: np.ndarray, beta: float, rng: np.random.Generator,
              J: float = 1.0, return_details: bool = False):
    """
    Standard Swendsen–Wang update for 2D Ising with h=0.
    (Field h!=0 needs a modification; we'll add later if desired.)
    """
    N = spins.size
    p_bond = 1.0 - np.exp(-2.0 * beta * J)

    i = bonds[:, 0]
    j = bonds[:, 1]

    parallel = (spins[i] == spins[j])
    w = parallel & (rng.random(size=bonds.shape[0]) < p_bond)  # boolean

    active = w
    # IMPORTANT: only store active edges, all values are 1.
    A = csr_matrix(
        (np.ones(active.sum(), dtype=np.int8), (i[active], j[active])),
        shape=(N, N)
    )
    A = A + A.T  # make undirected graph

    n_comp, labels = connected_components(A, directed=False)

    flip = rng.random(n_comp) < 0.5
    spins_new = spins.copy()
    spins_new[flip[labels]] *= -1

    if return_details:
        return spins_new, {"p_bond": p_bond, "w": w, "labels": labels, "flip": flip, "n_comp": n_comp}
    return spins_new


def metropolis_sweep(spins: np.ndarray, nbrs: np.ndarray, beta: float, rng: np.random.Generator,
                     J: float = 1.0, h: float = 0.0) -> float:
    """One sweep = N random single-spin proposals."""
    N = spins.size
    acc = 0
    for _ in range(N):
        n = int(rng.integers(0, N))
        s = spins[n]
        nn_sum = spins[nbrs[n]].sum()
        dE = 2.0 * s * (J * nn_sum + h)
        if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
            spins[n] = -s
            acc += 1
    return acc / N


# -----------------------------
# Simulation wrapper (class)
# -----------------------------

@dataclass
class Ising2D:
    Lx: int
    Ly: int
    J: float = 1.0
    h: float = 0.0
    kB: float = 1.0
    seed: int = 0

    def __post_init__(self):
        self.N = self.Lx * self.Ly
        self.rng = np.random.default_rng(self.seed)
        self.bonds = build_bonds(self.Lx, self.Ly)
        self.nbrs = build_neighbors(self.Lx, self.Ly)
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)

    def beta(self, T: float) -> float:
        return 1.0 / (self.kB * T)

    def reset_random(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(np.int8)

    def run(self, algo: str, T: float, n_therm: int, n_meas: int, stride: int = 1,
            record_configs: bool = False, config_every: int = 50):
        """
        algo: "sw" or "metropolis"
        stride: number of updates between measurements (often 1 is fine)
        record_configs: store some spin snapshots for visualization
        config_every: store config every this many measurements
        """
        beta = self.beta(T)

        # thermalize
        if algo == "sw":
            if self.h != 0.0:
                raise ValueError("Standard SW here assumes h=0. Set h=0 or implement ghost-spin first.")
            for _ in range(n_therm):
                self.spins = sw_update(self.spins, self.bonds, beta, self.rng, J=self.J)
        elif algo == "metropolis":
            for _ in range(n_therm):
                metropolis_sweep(self.spins, self.nbrs, beta, self.rng, J=self.J, h=self.h)
        else:
            raise ValueError("algo must be 'sw' or 'metropolis'")

        E = np.empty(n_meas)
        M = np.empty(n_meas)
        acc = np.empty(n_meas) if algo == "metropolis" else None

        configs = []
        for t in range(n_meas):
            for _ in range(stride):
                if algo == "sw":
                    self.spins = sw_update(self.spins, self.bonds, beta, self.rng, J=self.J)
                else:
                    acc[t] = metropolis_sweep(self.spins, self.nbrs, beta, self.rng, J=self.J, h=self.h)

            E[t] = energy_density(self.spins, self.bonds, J=self.J, h=self.h)
            M[t] = magnetization(self.spins)

            if record_configs and (t % config_every == 0 or t == n_meas - 1):
                configs.append(self.spins.copy())

        return E, M, acc, configs
