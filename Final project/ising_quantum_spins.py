#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# -----------------------------
# Geometry: bonds + neighbors
# -----------------------------

def build_bonds(Lx, Ly):
    """Return array of shape (2N,2) listing each nearest-neighbor bond once (right + down), PBC."""
    N = Lx * Ly
    bonds = np.empty((2 * N, 2), dtype=np.int32)
    k = 0
    for x in range(Lx):
        for y in range(Ly):
            n = x * Ly + y
            bonds[k] = (n, x * Ly + ((y + 1) % Ly)); k += 1          # right
            bonds[k] = (n, ((x + 1) % Lx) * Ly + y); k += 1          # down
    return bonds

def build_neighbors(Lx, Ly):
    """Return nbrs[n,4] = (right,left,down,up), PBC. Useful for fast Metropolis ΔE."""
    N = Lx * Ly
    nbrs = np.empty((N, 4), dtype=np.int32)
    for x in range(Lx):
        for y in range(Ly):
            n = x * Ly + y
            nbrs[n, 0] = x * Ly + ((y + 1) % Ly)             # right
            nbrs[n, 1] = x * Ly + ((y - 1) % Ly)             # left
            nbrs[n, 2] = ((x + 1) % Lx) * Ly + y             # down
            nbrs[n, 3] = ((x - 1) % Lx) * Ly + y             # up
    return nbrs


# -----------------------------
# Physics: observables
# -----------------------------

def energy(spins, bonds, J=1.0, h=0.0):
    i = bonds[:, 0]
    j = bonds[:, 1]
    bond_term = np.sum(spins[i] * spins[j])       # each bond counted once
    field_term = np.sum(spins)
    return float(-J * bond_term - h * field_term)

def energy_density(spins, bonds, J=1.0, h=0.0):
    return energy(spins, bonds, J=J, h=h) / spins.size

def magnetization(spins):
    return float(np.mean(spins))


# -----------------------------
# Updates: Swendsen–Wang and Metropolis
# -----------------------------

def sw_update(spins, bonds, beta, rng, J=1.0, return_details=False):
    """
    One Swendsen-Wang update for 2D Ising at h=0 (standard version).
    For h != 0, this needs modification (ghost spin trick) — later.
    """
    N = spins.size
    p_bond = 1.0 - np.exp(-2.0 * beta * J)

    i = bonds[:, 0]
    j = bonds[:, 1]

    parallel = (spins[i] == spins[j])
    w = parallel & (rng.random(size=bonds.shape[0]) < p_bond)   # boolean

    active = w
    A = csr_matrix(
        (np.ones(active.sum(), dtype=np.int8), (i[active], j[active])),
        shape=(N, N)
    )
    A = A + A.T

    n_comp, labels = connected_components(A, directed=False)

    flip = rng.random(n_comp) < 0.5
    spins_new = spins.copy()
    spins_new[flip[labels]] *= -1

    if return_details:
        return spins_new, {"p_bond": p_bond, "w": w, "labels": labels, "flip": flip, "n_comp": n_comp}
    return spins_new

def metropolis_sweep(spins, nbrs, beta, rng, J=1.0, h=0.0):
    """One sweep = N single-spin proposals."""
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
# Autocorrelation + tau_int
# -----------------------------

def autocorr_norm(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.mean(x * x)
    rho = np.empty(max_lag + 1, dtype=float)
    rho[0] = 1.0
    if var == 0:
        rho[1:] = 0.0
        return rho
    for lag in range(1, max_lag + 1):
        rho[lag] = np.mean(x[:-lag] * x[lag:]) / var
    return rho

def tau_int_from_rho(rho):
    """Simple cutoff at first non-positive rho."""
    s = 0.5
    for lag in range(1, len(rho)):
        if rho[lag] <= 0:
            break
        s += rho[lag]
    return s


# -----------------------------
# Runs: thermalize + measure
# -----------------------------

def run_sw(Lx, Ly, T, n_therm, n_meas, J=1.0, h=0.0, kB=1.0, seed=0):
    assert h == 0.0, "Standard SW update here assumes h=0. We'll add field later (ghost spin)."
    rng = np.random.default_rng(seed)
    N = Lx * Ly
    beta = 1.0 / (kB * T)

    bonds = build_bonds(Lx, Ly)
    spins = rng.choice([-1, 1], size=N).astype(np.int8)

    for _ in range(n_therm):
        spins = sw_update(spins, bonds, beta, rng, J=J)

    E = np.empty(n_meas)
    M = np.empty(n_meas)
    for t in range(n_meas):
        spins = sw_update(spins, bonds, beta, rng, J=J)
        E[t] = energy_density(spins, bonds, J=J, h=h)
        M[t] = magnetization(spins)
    return E, M

def run_metropolis(Lx, Ly, T, n_therm, n_meas, J=1.0, h=0.0, kB=1.0, seed=0):
    rng = np.random.default_rng(seed)
    N = Lx * Ly
    beta = 1.0 / (kB * T)

    bonds = build_bonds(Lx, Ly)
    nbrs = build_neighbors(Lx, Ly)
    spins = rng.choice([-1, 1], size=N).astype(np.int8)

    for _ in range(n_therm):
        metropolis_sweep(spins, nbrs, beta, rng, J=J, h=h)

    E = np.empty(n_meas)
    M = np.empty(n_meas)
    for t in range(n_meas):
        metropolis_sweep(spins, nbrs, beta, rng, J=J, h=h)
        E[t] = energy_density(spins, bonds, J=J, h=h)
        M[t] = magnetization(spins)
    return E, M

def measure_autocorrs(E, M, max_lag):
    rhoE = autocorr_norm(E, max_lag)
    rhoM = autocorr_norm(M, max_lag)
    rhoAbsM = autocorr_norm(np.abs(M), max_lag)

    return {
        "rhoE": rhoE, "tauE": tau_int_from_rho(rhoE),
        "rhoM": rhoM, "tauM": tau_int_from_rho(rhoM),
        "rhoAbsM": rhoAbsM, "tauAbsM": tau_int_from_rho(rhoAbsM),
    }

# --- parameters ---
Lx = Ly = 32
J = 1.0
kB = 1.0
Tc = 2.269  # for J=1, kB=1 (2D Ising)
temps = [1.5, Tc, 3.5]

n_therm = 300
n_meas  = 4000
max_lag = 300
seed = 1

for T in temps:
    E_sw, M_sw = run_sw(Lx, Ly, T, n_therm, n_meas, J=J, kB=kB, seed=seed)
    E_me, M_me = run_metropolis(Lx, Ly, T, n_therm, n_meas, J=J, kB=kB, seed=seed)

    rhoE_sw = autocorr_norm(E_sw, max_lag)
    rhoE_me = autocorr_norm(E_me, max_lag)

    tau_sw = tau_int_from_rho(rhoE_sw)
    tau_me = tau_int_from_rho(rhoE_me)

    print(f"T={T:.3f}: tau_int(E) SW={tau_sw:.2f}, Metropolis={tau_me:.2f}")

    plt.figure()
    plt.plot(rhoE_sw, label="SW")
    plt.plot(rhoE_me, label="Metropolis")
    plt.title(f"Energy autocorrelation rho_E(lag), T={T:.3f}, L={Lx}")
    plt.xlabel("lag")
    plt.ylabel("rho_E")
    plt.grid(True)
    plt.legend()
    plt.show()
    stats_sw = measure_autocorrs(E_sw, M_sw, max_lag)
    stats_me = measure_autocorrs(E_me, M_me, max_lag)

    print(f"T={T:.3f}")
    print(f"  SW:  tauE={stats_sw['tauE']:.2f}, tau|M|={stats_sw['tauAbsM']:.2f}")
    print(f"  Met: tauE={stats_me['tauE']:.2f}, tau|M|={stats_me['tauAbsM']:.2f}")