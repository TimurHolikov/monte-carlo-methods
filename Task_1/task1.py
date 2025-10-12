import random
from random import random as rnd
import numpy as np
import matplotlib.pyplot as plot

random.seed(1234)

N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
pi_estimates = []
errors = []

for N in N_values:
    hits = 0
    x_points = []
    y_points = []
    colors = []

    for _ in range(N):
        x = rnd()
        y = rnd()
        x_points.append(x)
        y_points.append(y)
        if 1 - x**2 - y**2 > 0:
            hits += 1
            colors.append('red')
        else:
            colors.append('blue')

    pi_estimate = 4/N * hits
    pi_estimates.append(pi_estimate)
    errors.append(abs(np.pi - pi_estimate))

    plot.figure(figsize=(5,5))
    plot.scatter(x_points, y_points, c=colors, s=1)
    theta = np.linspace(0, np.pi/2, 300)
    plot.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
    plot.gca().set_aspect('equal')
    plot.title(f"Pi estimation (N={N})")
    plot.xlabel("x")
    plot.ylabel("y")
    plot.show()

# Convergence plot
plot.figure(figsize=(6,4))
plot.plot(N_values, pi_estimates, 'o-', label=r'Estimate of $\pi$')
plot.axhline(np.pi, color='gray', linestyle='--', label=r'Exact $\pi$')
plot.xscale('log')
plot.xlabel("N")
plot.ylabel(r"$\pi$ estimate")
plot.title("Monte Carlo Convergence")
plot.legend()
plot.show()

# Error analysis
plot.figure(figsize=(6,4))
plot.plot(N_values, errors, 'o-', label=r'|$\pi_{\rm est} - \pi$|')
plot.plot(N_values, 1/np.sqrt(N_values), 'r--', label=r'1/$\sqrt{N}$ scaling')
plot.xscale('log')
plot.yscale('log')
plot.xlabel("N")
plot.ylabel("Error")
plot.title("Error scaling with N")
plot.legend()
plot.show()