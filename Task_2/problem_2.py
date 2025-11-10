import numpy as np
from numpy.random import random as rnd
import matplotlib.pyplot as plot

def mc_sample_inversion(N):
    # f(x) = x^2 on [0,1) -> F(x) = x^3 -> x = u^(1/3)
    u = rnd(N)
    x = np.cbrt(u)
    return x


def problem_2a(N = 100_000, bins = 40):
    def bin_avg_theory(edges):
        # For p(x)=3x^2 on [0,1]: h = (b^3 - a^3)/(b - a)
        a = edges[:-1]
        b = edges[1:]
        return (b**3 - a**3) / (b - a)

    def moment(k):
        # E[X^k] = ∫_0^1 x^k * p(x) dx = ∫_0^1 x^k * 3x^2 dx = 3 ∫_0^1 x^{k+2} dx
        # ∫_0^1 x^n dx = 1/(n+1)
        return 3.0 / (k + 3.0)

    def mean_and_var():
        ex1 = moment(1) # E[X]
        ex2 = moment(2) # E[X^2]
        var = ex2 - ex1**2
        return ex1, var
    
    x = mc_sample_inversion(N)

    counts, edges, _ = plot.hist(
        x, bins=bins, range=(0.0,1.0), density=True,
        alpha=0.7, edgecolor='black', label='Histogram'
    )

    # 3) theory: bin-average density and continuous pdf
    centers = 0.5*(edges[:-1] + edges[1:])
    avg_theory = bin_avg_theory(edges)
    grid = np.linspace(0, 1, 400)

    pdf = 3.0 * grid**2

    # 4) overlay
    plot.step(centers, avg_theory, where='mid', linewidth=2, label=r'$\frac{1}{\Delta x}\int_{bin} p(x)\,dx$')
    plot.plot(grid, pdf, 'r-', lw=2, label=r'$p(x)=3x^2$')

    plot.xlabel('x')
    plot.ylabel('Density')
    plot.title('Problem 2(a)')
    plot.legend()
    plot.show()

    # 5) numeric agreement: L2 and max abs error per bin
    density = counts
    l2 = float(np.sqrt(np.mean((density - avg_theory)**2)))
    max_abs = float(np.max(np.abs(density - avg_theory)))
    print(f"L2 error (hist vs bin-avg theory): {l2:.4e}")
    print(f"Max |diff| across bins: {max_abs:.4e}")

    mu_true, var_true = mean_and_var()
    print(f"E[X] exact(by integral) = {mu_true:.6f}, MC ≈ {x.mean():.6f}")
    print(f"Var exact(by integral) = {var_true:.6f}, MC ≈ {x.var(ddof=0):.6f} \n")

def problem_2b(N = 100_000, bins = 40):
    x = mc_sample_inversion(N)

    counts, edges = np.histogram(x, bins=bins, range=(0.0, 1.0), density=False)
    w = np.diff(edges) # bin widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 3) theoretical bin probability: p_k = ∫_a^b 3x^2 dx = b^3 - a^3
    a = edges[:-1]
    b = edges[1:]
    p_theory = b**3 - a**3


    # 4) variance estimator
    # sigma_bin = Nbin (N - Nbin) / [N^2 (N - 1)]
    Nbin = counts.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_frac = Nbin * (N - Nbin) / (N**2 * (N - 1))
        rel_err = np.sqrt(sigma_frac) / (Nbin / N)
        rel_err_1_over_sqrt = 1.0/np.sqrt(Nbin)
        rel_err_theory = np.sqrt((1.0 - p_theory) / (N * p_theory))
    
    
    # 5) plot comparisons
    sigma_theory_frac = p_theory*(1.0 - p_theory)/N  # Var( Nbin/N )
    (ax1, ax2) = plot.subplots(1, 2, figsize=(11,4))

    ax1.plot(centers, rel_err, 'o', ms=5, alpha=0.8,
                      label=r'$\frac{\sqrt{\sigma_{bin}}}{N_{bin}/N}$')
    ax1.plot(centers, rel_err_1_over_sqrt, 's', ms=4, alpha=0.7, label=r'$1/\sqrt{N_{bin}}$')
    ax1.plot(centers, rel_err_theory, '-', lw=2, label=r'$\sqrt{\frac{1-p_k}{N\,p_k}}$')
    ax1.set_xlabel('x'); ax1.set_ylabel('Relative error per bin'); ax1.legend()
    ax1.set_title('Problem 2(b)')

    ax2.plot(centers, sigma_frac, '.', label=r'$\sigma_{bin}$')
    ax2.plot(centers, sigma_theory_frac, '-', label=r'$p_k(1-p_k)/N$ (theory)')
    ax2.set_xlabel('x'); ax2.set_ylabel('Var of bin fraction')
    ax2.legend(); plot.tight_layout(); plot.show()

    def summarize(name, ref, est):
        m = np.isfinite(ref) & np.isfinite(est)
        l2 = float(np.sqrt(np.mean((ref[m] - est[m])**2)))
        mx = float(np.max(np.abs(ref[m] - est[m])))
        print(f"{name}:   L2={l2:.4e},  MaxAbs={mx:.4e}")

    print("Comparison to 1/sqrt(N_bin):")
    summarize("empirical vs 1/sqrt(N_bin)", rel_err, rel_err_1_over_sqrt)
    print("Comparison to theory sqrt((1-p)/(N p)):")
    summarize("empirical vs theory", rel_err, rel_err_theory)


if __name__ == "__main__":
    # problem_2a(2_000_000)
    # problem_2b(2_000_000)