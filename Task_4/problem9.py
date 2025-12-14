#%%
import numpy as np
import matplotlib.pyplot as plt


class Cascade:
    def __init__(self, alpha=0.3, mu=0.01, seed=1):
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.rng = np.random.default_rng(seed)

    def sample_qprime(self, q):
        # r in (0,1); for safety avoid exactly 0
        r = self.rng.random()
        if r == 0.0:
            r = np.nextafter(0.0, 1.0)

        L = np.log(1.0 / q)
        Lp = np.sqrt(L * L - (2.0 / self.alpha) * np.log(r))
        return np.exp(-Lp)

    def sample_zprime(self, z, qprime):
        r2 = self.rng.random()
        if r2 == 0.0:
            r2 = np.nextafter(0.0, 1.0)
        return z * (1.0 - (qprime ** r2))

    def run_one_event(self):
        q, z = 1.0, 1.0
        m = 0

        while True:
            qprime = self.sample_qprime(q)

            if qprime <= self.mu:
                return z, m

            z = self.sample_zprime(z, qprime)
            q = qprime
            m += 1

    def simulate(self, n_events=20000):
        z_final = np.empty(n_events, dtype=float)
        mult = np.empty(n_events, dtype=int)

        for i in range(n_events):
            z_final[i], mult[i] = self.run_one_event()

        return z_final, mult


def analyze_and_plot(z_final, mult, mu, bins_z=60):
    m0 = np.mean(mult == 0)
    print(f"mu = {mu}")
    print(f"P(m=0) = {m0:.4f} (this is the delta-peak at z=1)")


    # continuous part: only events with at least one emission
    mask = mult > 0
    z_cont = z_final[mask]


    # --- z_final histogram (continuous part)
    plt.figure()
    # log-bins make the small-z structure readable
    bins = np.logspace(-4, 0, 70)
    plt.hist(z_cont, bins=bins, density=True)
    plt.xscale('log')
    plt.xlabel(r"$z_{\mathrm{final}}$ (only $m\ge 1$)")
    plt.ylabel("density")
    plt.title(f"Final momentum (continuous part), mu={mu}")
    plt.grid(True)


    # --- multiplicity histogram
    plt.figure()
    mmax = int(mult.max())
    plt.hist(mult, bins=np.arange(mmax + 2) - 0.5, density=True)
    plt.xlabel("multiplicity m")
    plt.ylabel("probability")
    plt.title(f"Multiplicity distribution, mu={mu}")
    plt.grid(True)


    # --- key numbers (full distribution)
    mean_z = z_final.mean()
    var_z = z_final.var(ddof=1)
    mean_m = mult.mean()
    var_m = mult.var(ddof=1)


    print(f"<z_final> = {mean_z:.6f}")
    print(f"Var(z_final) = {var_z:.6e}")
    print(f"<m> = {mean_m:.6f}")
    print(f"Var(m) = {var_m:.6e}")


def scan_mu(alpha=0.3, mu_list=(0.1, 0.03, 0.01, 0.003), n_events=20000, seed=1):
    mus = np.array(mu_list, dtype=float)
    mean_zs = np.empty(len(mus))
    var_zs = np.empty(len(mus))
    mean_ms = np.empty(len(mus))

    for i, mu in enumerate(mus):
        cas = Cascade(alpha=alpha, mu=mu, seed=seed + i)  # разные seed для разных mu
        z_final, mult = cas.simulate(n_events=n_events)

        analyze_and_plot(z_final, mult, mu)

        mean_zs[i] = z_final.mean()
        var_zs[i] = z_final.var(ddof=1)
        mean_ms[i] = mult.mean()

    plt.figure()
    plt.plot(mus, mean_zs, marker="o")
    plt.xscale("log")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\langle z_{\mathrm{final}}\rangle$")
    plt.title("Mean final momentum vs mu")
    plt.grid(True)

    plt.figure()
    plt.plot(mus, var_zs, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\mathrm{Var}(z_{\mathrm{final}})$")
    plt.title("Variance of final momentum vs mu")
    plt.grid(True)

    plt.figure()
    plt.plot(mus, mean_ms, marker="o")
    plt.xscale("log")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\langle m\rangle$")
    plt.title("Mean multiplicity vs mu")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    scan_mu()
