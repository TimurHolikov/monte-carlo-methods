import random
from random import random as rnd
import numpy as np
import matplotlib.pyplot as plot

def monte_carlo_pi(N_values, seed = None):
    pi_estimates = []
    errors = []
    x_points_list = []
    y_points_list = []
    colors_list = []

    seed = seed if seed is not None else 1234
    random.seed(seed)

    for N in N_values:
        hits = 0
        colors = []

        # r = [rnd() for _ in range(2*N)]
        # x_points = r[0::2]
        # y_points = r[1::2]
        # for x, y in zip(x_points, y_points):
        #     if 1 - x**2 - y**2 > 0:
        #         hits += 1
        #         colors.append('red')
        #     else:
        #         colors.append('blue')

        # x_points = []
        # y_points = []
        # for _ in range(N):
        #     x = rnd()
        #     y = rnd()
        #     x_points.append(x)
        #     y_points.append(y)
        #     if 1 - x**2 - y**2 > 0:
        #         hits += 1
        #         colors.append('red')
        #     else:
        #         colors.append('blue')

        np.random.seed(seed)
        x_points = np.random.rand(N)
        y_points = np.random.rand(N)

        inside = (1 - x_points**2 - y_points**2) > 0
        hits = np.sum(inside)

        colors = np.where(inside, 'red', 'blue')

        pi_estimate = 4/N * hits
        pi_estimates.append(pi_estimate)
        errors.append(abs(np.pi - pi_estimate))

        x_points_list.append(x_points)
        y_points_list.append(y_points)
        colors_list.append(colors)

    return pi_estimates, errors, x_points_list, y_points_list, colors_list

# Run simulation
N_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
pi_estimates, errors, x_points_list, y_points_list, colors_list = monte_carlo_pi(N_values)

# a) Plots
nrows, ncols = 2, 3
fig1, axes1 = plot.subplots(nrows, ncols, figsize=(3.5*ncols, 3.5*nrows))
axes1 = axes1.flatten()
for i, N in enumerate(N_values[0:6]):
    axes1[i].scatter(x_points_list[i], y_points_list[i], c=colors_list[i], s=1)
    theta = np.linspace(0, np.pi/2, 300)
    axes1[i].plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
    axes1[i].set_aspect('equal')
    axes1[i].set_title(r"$N = {}, \pi \approx {:.4f}$".format(N, pi_estimates[i]))
    axes1[i].set_xlabel("x")
    axes1[i].set_ylabel("y")

# b) Poisson estimate
p = np.pi/4
poisson_sigma = 4 * np.sqrt(p*(1-p)/np.array(N_values))

fig1.suptitle("Pi estimation for different N")
plot.tight_layout()
plot.show(block=False)

## Convergence plot
fig2, (ax2, ax3) = plot.subplots(1, 2, figsize=(12, 5))
ax2.plot(N_values, pi_estimates, 'o-', label=r'Estimate of $\pi$')
ax2.axhline(np.pi, color='gray', linestyle='--', label=r'Exact $\pi$')
ax2.set_xscale('log')
ax2.set_xlabel("N")
ax2.set_ylabel(r"$\pi$ estimate")
ax2.set_title("Monte Carlo Convergence")
ax2.legend()

## Error plot
ax3.plot(N_values, errors, 'o-', label=r'|$\pi_{\rm est} - \pi$|')
ax3.plot(N_values, 1/np.sqrt(np.array(N_values)), 'r--', label=r'1/$\sqrt{N}$ scaling')
ax3.plot(N_values, poisson_sigma, 'g--', label='Poisson estimate')
ax3.set_xscale('log')
# ax3.set_yscale('log')
ax3.set_xlabel("N")
ax3.set_ylabel("Error")
ax3.set_title("Error scaling with N")
ax3.legend()

fig2.suptitle("Convergence and Error Analysis")
plot.tight_layout()
plot.show(block=False)

# c) Different seeds
## Fixed N
N_value_b = 150000
# seeds = list(range(300))
seeds = random.sample(range(1, 100001), 300)
pi_values = []

for s in seeds:
    pi_est, _, _, _, _ = monte_carlo_pi([N_value_b], seed=s)
    pi_values.append(pi_est[0])

plot.figure()
plot.hist(pi_values, bins=60, alpha=0.7, edgecolor='black')
plot.axvline(np.pi, color='red', linestyle='--', label='Exact π')
plot.xlabel('π estimate')
plot.ylabel('Counts')
plot.title(f'Histogram of π estimates for N={N_value_b} and different seeds')
plot.show(block=False)

## Different N and seeds
seeds = [4242, 1213, 9994, 5411, 43242, 1, 2]

plot.figure(figsize=(8, 5))
for s in seeds:
    pi_vals = []
    for N in N_values:
        pi_est, _, _, _, _ = monte_carlo_pi([N], seed=s)
        pi_vals.append(pi_est[0])
    plot.plot(N_values, pi_vals, 'o-', label=f'seed={s}')

plot.axhline(np.pi, color='gray', linestyle='--', label=r'Exact $\pi$')
plot.xscale('log')
plot.xlabel('N')
plot.ylabel('π estimate')
plot.title('π estimates for different seeds and N')
plot.legend()
plot.show()