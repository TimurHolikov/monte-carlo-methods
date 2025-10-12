import random
from random import random as rnd
import numpy as np
import matplotlib.pyplot as plot

random.seed(1234)

N_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
pi_estimates = []
errors = []
x_points_list = []
y_points_list = []
colors_list = []

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
    x_points_list.append(x_points)
    y_points_list.append(y_points)
    colors_list.append(colors)

# Plots
nrows, ncols = 2, 4
fig1, axes1 = plot.subplots(nrows, ncols, figsize=(3.5*ncols, 3.5*nrows))
axes1 = axes1.flatten()
for idx, N in enumerate(N_values):
    axes1[idx].scatter(x_points_list[idx], y_points_list[idx], c=colors_list[idx], s=1)
    theta = np.linspace(0, np.pi/2, 300)
    axes1[idx].plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
    axes1[idx].set_aspect('equal')
    axes1[idx].set_title(f"N={N}")
    axes1[idx].set_xlabel("x")
    axes1[idx].set_ylabel("y")

# for idx in range(len(N_values), nrows*ncols):
#     fig1.delaxes(axes1[idx])

fig1.suptitle("Pi estimation for different N")
plot.tight_layout()
plot.show(block=False)

# Convergence plot
fig2, (ax2, ax3) = plot.subplots(1, 2, figsize=(12, 5))
ax2.plot(N_values, pi_estimates, 'o-', label=r'Estimate of $\pi$')
ax2.axhline(np.pi, color='gray', linestyle='--', label=r'Exact $\pi$')
ax2.set_xscale('log')
ax2.set_xlabel("N")
ax2.set_ylabel(r"$\pi$ estimate")
ax2.set_title("Monte Carlo Convergence")
ax2.legend()

ax3.plot(N_values, errors, 'o-', label=r'|$\pi_{\rm est} - \pi$|')
ax3.plot(N_values, 1/np.sqrt(np.array(N_values)), 'r--', label=r'1/$\sqrt{N}$ scaling')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel("N")
ax3.set_ylabel("Error")
ax3.set_title("Error scaling with N")
ax3.legend()

fig2.suptitle("Convergence and Error Analysis")
plot.tight_layout()
plot.show()