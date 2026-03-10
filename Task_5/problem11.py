#%%
from math import log, exp
from random import random
import matplotlib.pyplot as plt

# direct algorithm from problem 9
def one_cascade_direct(alpha=0.2, mu=0.01):
    q = 1.0
    z = 1.0
    n_emissions = 0
    while True:
        r1 = random()
        Lq = log(1.0 / q)
        q_new = exp(-((Lq**2 - (2.0 / alpha) * log(r1)) ** 0.5))

        if q_new < mu:
            break

        r2 = random()
        z_new = z * (1.0 - q_new ** r2)

        q = q_new
        z = z_new
        n_emissions += 1

    return z, n_emissions

# veto algorithm
def one_cascade_veto(alpha=0.2, mu=0.01):
    q = 1.0
    z = 1.0
    n_emissions = 0
    while True:
        r1 = random()

        # candidate q' from overestimate g(q)=alpha/q^2
        q_new = 1.0 / (1.0 / q - (1.0 / alpha) * log(r1))
        if q_new < mu:
            break

        # z' uniform on allowed interval
        r2 = random()
        z_candidate = r2 * z * (1.0 - q_new)

        # acceptance probability P/P'
        accept_prob = (z * q_new * (1.0 - q_new)) / (z - z_candidate)

        r3 = random()
        if r3 < accept_prob:
            z = z_candidate
            n_emissions += 1

        # in any case the evolution scale moves to q'
        q = q_new

    return z, n_emissions

def sample(fn, M=10000, alpha=0.2, mu=0.01):
    z_final = []
    multiplicities = []

    for _ in range(M):
        z_end, n = fn(alpha=alpha, mu=mu)
        z_final.append(z_end)
        multiplicities.append(n)

    mean_z = sum(z_final) / M
    mean_z2 = sum(v * v for v in z_final) / M
    var_z = mean_z2 - mean_z**2
    mean_n = sum(multiplicities) / M

    return z_final, multiplicities, mean_z, var_z, mean_n

if __name__ == "__main__":
    alpha = 0.2
    mu = 0.01
    M = 20000

    zd, nd, mean_zd, var_zd, mean_nd = sample(one_cascade_direct, M=M, alpha=alpha, mu=mu)
    zv, nv, mean_zv, var_zv, mean_nv = sample(one_cascade_veto,   M=M, alpha=alpha, mu=mu)

    print("Direct:")
    print(f"<z_final> = {mean_zd:.6f}, Var(z) = {var_zd:.6f}, <n> = {mean_nd:.6f}")

    print("Veto:")
    print(f"<z_final> = {mean_zv:.6f}, Var(z) = {var_zv:.6f}, <n> = {mean_nv:.6f}")

    plt.figure(figsize=(7,5))
    plt.hist(zd, bins=50, density=True, histtype='step', linewidth=2, label='direct')
    plt.hist(zv, bins=50, density=True, histtype='step', linewidth=2, label='veto')
    plt.xlabel(r'final momentum fraction $z$')
    plt.ylabel('probability density')
    plt.title('Final momentum fraction: direct vs veto')
    plt.legend()
    plt.show()

    bins = range(max(max(nd), max(nv)) + 2)
    plt.figure(figsize=(7,5))
    plt.hist(nd, bins=bins, density=True, histtype='step', linewidth=2, align='left', label='direct')
    plt.hist(nv, bins=bins, density=True, histtype='step', linewidth=2, align='left', label='veto')
    plt.xlabel('number of emitted particles')
    plt.ylabel('probability')
    plt.title('Multiplicity: direct vs veto')
    plt.legend()
    plt.show()
#%%