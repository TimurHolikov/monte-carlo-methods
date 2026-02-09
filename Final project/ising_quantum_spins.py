#%%
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

rng = np.random.default_rng(seed=42)

Lx = 32
Ly = 32
N = Lx * Ly

# kB = 1.380649e-23
kB = 1.0
J  = 1.0
T  = 2.269
beta = 1.0/(kB*T)

spins = rng.choice([-1, 1], size=N)

def build_bonds(Lx, Ly):
    N = Lx * Ly
    bonds = np.empty((2*N, 2), dtype=np.int32)
    k = 0
    for x in range(Lx):
        for y in range (Ly):
            n = x*Ly + y
            
            # right neighbour (x, y+1)
            m = x*Ly + ((y+1) % Ly)
            bonds[k] = (n,m); k += 1
            
            # down neighbour  (x+1, y)
            m = ((x + 1) % Lx)*Ly + y
            bonds[k] = (n, m); k += 1
            
    return bonds

bonds = build_bonds(Lx, Ly)

def sw_update(spins, bonds, beta, rng, J=1.0, return_details=False):
    N = spins.size
    p_bond = 1.0 - np.exp(-2.0 * beta * J)

    i = bonds[:, 0]
    j = bonds[:, 1]

    parallel = (spins[i] == spins[j])
    w = (parallel & (rng.random(size=bonds.shape[0]) < p_bond))

    active = w.astype(bool)
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

def magnetization(spins):
    return float(np.mean(spins))

def energy_density(spins, bonds, J=1.0, h=0.0):
    return energy(spins, bonds, J, h) / spins.size

def energy(spins, bonds, J=1.0, h=0.0):
    i = bonds[:, 0]
    j = bonds[:, 1]
    bond_term = np.sum(spins[i] * spins[j])
    field_term = np.sum(spins)
    H = -J * bond_term - h * field_term
    return float(H)

print("E/N =", energy_density(spins, bonds, J=J, h=0.0))
print("m   =", magnetization(spins))

n_steps = 50
E_series = np.empty(n_steps)
M_series = np.empty(n_steps)

for t in range(n_steps):
    spins = sw_update(spins, bonds, beta, rng, J=J)
    E_series[t] = energy_density(spins, bonds, J=J, h=0.0)
    M_series[t] = magnetization(spins)

print("E/N mean:", E_series.mean())
print("M mean  :", M_series.mean())

import matplotlib.pyplot as plt

plt.figure()
plt.plot(E_series)
plt.title("Energy density vs SW step")
plt.xlabel("step")
plt.ylabel("E/N")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(M_series)
plt.title("Magnetization vs SW step")
plt.xlabel("step")
plt.ylabel("m")
plt.grid(True)
plt.show()