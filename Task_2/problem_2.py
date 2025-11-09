import numpy as np
from numpy.random import random as rnd
import matplotlib.pyplot as plot

def bin_avg_theory(edges):
    # For p(x)=3x^2 on [0,1]: h = (b^3 - a^3)/(b - a)
    a = edges[:-1]
    b = edges[1:]
    return (b**3 - a**3) / (b - a)

def problem_2a(N = 100_000):
    # f(x) = x^2 on [0,1) -> F(x) = x^3 -> x = u^(1/3)
    u = rnd(N)
    x = np.cbrt(u)

    counts, edges, patches = plot.hist(
        x, bins=40, range=(0.0,1.0), density=True,
        alpha=0.7, edgecolor='black', label='Histogram'
    )

    # 3) theory: bin-average density and continuous pdf
    centers = 0.5*(edges[:-1] + edges[1:])
    avg_theory = bin_avg_theory(edges)
    grid = np.linspace(0, 1, 400)

    pdf = 3.0 * grid**2

    # 4) overlay
    plot.step(centers, avg_theory, where='mid', linewidth=2, label='(∫_bin p)/Δx')
    plot.plot(grid, pdf, 'r-', lw=2, label=r'$p(x)=3x^2$')

    plot.xlabel('x')
    plot.ylabel('Density')
    plot.title('Problem 2(a)')
    plot.legend()
    plot.show()

    # 5) numeric agreement: L2 and max abs error per bin
    density_hat = counts  # already normalized as density
    l2 = float(np.sqrt(np.mean((density_hat - avg_theory)**2)))
    max_abs = float(np.max(np.abs(density_hat - avg_theory)))
    print(f"L2 error (hist vs bin-avg theory): {l2:.4e}")
    print(f"Max |diff| across bins: {max_abs:.4e}")

    # 6) moment checks
    mean_true = 3/4
    var_true  = 3/5 - (3/4)**2  # 3/80
    print(f"E[X] true = {mean_true:.6f},  MC ≈ {x.mean():.6f}")
    print(f"Var true  = {var_true:.6f},  MC ≈ {x.var(ddof=0):.6f}")


if __name__ == "__main__":
    problem_2a(2_000_000)