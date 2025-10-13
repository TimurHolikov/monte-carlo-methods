import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Monte Carlo setup ---
N_total = 2000  # total points
np.random.seed(42)
x = np.random.rand(N_total)
y = np.random.rand(N_total)
inside = (x**2 + y**2) <= 1

# Precompute running π estimates and errors
pi_estimates = []
errors = []
for i in range(1, N_total + 1):
    hits = np.sum(inside[:i])
    pi_est = 4 * hits / i
    pi_estimates.append(pi_est)
    errors.append(abs(np.pi - pi_est))

# --- Figure setup ---
fig, (ax_scatter, ax_conv) = plt.subplots(1, 2, figsize=(11, 5))

# Left: points in square
theta = np.linspace(0, np.pi/2, 300)
ax_scatter.plot(np.cos(theta), np.sin(theta), 'k', linewidth=2)
ax_scatter.set_aspect('equal')
ax_scatter.set_xlim(0, 1)
ax_scatter.set_ylim(0, 1)
ax_scatter.set_title("Monte Carlo sampling")
ax_scatter.set_xlabel("x")
ax_scatter.set_ylabel("y")

# points (empty initially)
points_inside, = ax_scatter.plot([], [], 'ro', markersize=3, label='Inside')
points_outside, = ax_scatter.plot([], [], 'bo', markersize=3, label='Outside')
ax_scatter.legend(loc='upper right')

# Right: convergence of π
ax_conv.set_xlim(0, N_total)
ax_conv.set_ylim(3.05, 3.25)
ax_conv.axhline(np.pi, color='gray', linestyle='--', label=r'Exact $\pi$')
line_pi, = ax_conv.plot([], [], 'r-', lw=1.5, label=r'$\pi_{\rm est}$')
ax_conv.set_title("Convergence of π estimate")
ax_conv.set_xscale('log')
ax_conv.set_xlabel("N")
ax_conv.set_ylabel(r"$\pi$")
ax_conv.legend(loc='best')

# --- Animation function ---
def update(frame):
    # update scatter
    current_inside = x[:frame][inside[:frame]]
    current_outside = x[:frame][~inside[:frame]]
    current_inside_y = y[:frame][inside[:frame]]
    current_outside_y = y[:frame][~inside[:frame]]
    points_inside.set_data(current_inside, current_inside_y)
    points_outside.set_data(current_outside, current_outside_y)
    
    # update convergence plot
    line_pi.set_data(np.arange(1, frame+1), pi_estimates[:frame])
    
    return points_inside, points_outside, line_pi

# --- Create animation ---
ani = FuncAnimation(fig, update, frames=np.arange(10, N_total, 10),
                    interval=30, blit=True, repeat=False)

plt.tight_layout()
plt.show()