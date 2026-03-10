#%%
from math import log, exp
from random import random
import matplotlib.pyplot as plt

def one_cascade(alpha=0.2, mu=0.01):
    q = 1.0
    z = 1.0
    n_emissions = 0
    history = [(q, z)]

    while True:
        r1 = random()

        Lq = log(1.0 / q)
        q_new = exp(-((Lq**2 - (2.0/alpha)*log(r1))**0.5))

        if q_new < mu:
            q = mu
            history.append((q, z))
            break

        r2 = random()
        z_new = z * (1.0 - q_new**r2)

        q = q_new
        z = z_new
        n_emissions += 1
        history.append((q, z))

    return z, n_emissions, history

def sample_cascades(M=10000, alpha=0.2, mu=0.01):
    z_final = []
    multiplicities = []

    for _ in range(M):
        z_end, n, _ = one_cascade(alpha=alpha, mu=mu)
        z_final.append(z_end)
        multiplicities.append(n)

    mean_z = sum(z_final) / M
    mean_z2 = sum(v*v for v in z_final) / M
    var_z = mean_z2 - mean_z**2

    return z_final, multiplicities, mean_z, var_z

if __name__ == "__main__":
    alpha = 0.2
    mu = 0.01
    M = 20000

    z_final, multiplicities, mean_z, var_z = sample_cascades(M=M, alpha=alpha, mu=mu)
    
    p0 = sum(1 for n in multiplicities if n == 0) / len(multiplicities)
    print("P(n=0) =", p0)

    print(f"alpha = {alpha}, mu = {mu}")
    print(f"<z_final> = {mean_z:.6f}")
    print(f"Var(z_final) = {var_z:.6f}")
    print(f"<n_emissions> = {sum(multiplicities)/len(multiplicities):.6f}")

    plt.figure(figsize=(7,5))
    plt.hist(z_final, bins=50, density=True)
    plt.xlabel(r"final momentum fraction $z$")
    plt.ylabel("probability density")
    plt.title("Distribution of final momentum fraction")
    plt.show()

    plt.figure(figsize=(7,5))
    plt.hist(multiplicities, bins=range(max(multiplicities)+2), density=True, align='left')
    plt.xlabel("number of emitted particles")
    plt.ylabel("probability")
    plt.title("Multiplicity distribution")
    plt.show()
    
    for mu_test in [0.1, 0.05, 0.02, 0.01, 0.005]:
        z_final, multiplicities, mean_z, var_z = sample_cascades(M=5000, alpha=0.2, mu=mu_test)
        mean_n = sum(multiplicities) / len(multiplicities)
        print(f"mu={mu_test:>6}   <z_final>={mean_z:.4f}   Var(z)={var_z:.4f}   <n>={mean_n:.4f}")
#%%
