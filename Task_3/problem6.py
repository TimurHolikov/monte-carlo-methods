#%%
from random import choice, gauss
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

DIRS = [(1,0), (0,1), (-1,0), (0,-1)]

# --- (b) one walk: store the whole path ---
def random_walk_path(N):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(N):
        dx, dy = choice(DIRS)
        x += dx
        y += dy
        path.append((x, y))
    return path

def plot_one_walk(path):
    xs, ys = zip(*path)

    plt.figure(figsize=(7,7))
    plt.plot(xs, ys, lw=1)
    plt.scatter(xs, ys, s=10, alpha=0.6)
    plt.scatter([xs[0]], [ys[0]], c='g', s=50, label='Start')
    plt.scatter([xs[-1]],[ys[-1]], c='r', s=50, label='End')
    plt.axis("equal")
    plt.title(f"One 2D random walk (N={len(path)-1})")
    plt.legend()
    plt.show()

# --- helper: final point only (faster for MSD) ---
def random_walk_final(N):
    x, y = 0, 0
    for _ in range(N):
        dx, dy = choice(DIRS)
        x += dx
        y += dy
    return x, y

# --- (c) mean squared distance for unit steps ---
def mean_r2_unit(N, M=10000):
    s = 0.0
    for _ in range(M):
        x, y = random_walk_final(N)
        s += x*x + y*y
    return s / M

# --- (d) Gaussian step length kappa ~ N(0,1), step = kappa * e_i ---
def random_walk_final_gaussian(N):
    x, y = 0.0, 0.0
    for _ in range(N):
        dx, dy = choice(DIRS)
        kappa = gauss(0.0, 1.0)   # Gaussian N(0,1)
        x += kappa * dx
        y += kappa * dy
    return x, y

def mean_r2_gaussian(N, M=10000):
    s = 0.0
    for _ in range(M):
        x, y = random_walk_final_gaussian(N)
        s += x*x + y*y
    return s / M

# --- run (b) ---
path = random_walk_path(100)
plot_one_walk(path)

# --- run (c) and (d) for some N ---
print("Unit step (c):")
for N in [10, 50, 100, 200, 500, 1000]:
    print(N, mean_r2_unit(N, M=20000))

print("\nGaussian step length (d):")
for N in [10, 50, 100, 200, 500, 1000]:
    print(N, mean_r2_gaussian(N, M=20000))

# --- optional: plot MSD curves ---
Ns = list(range(1, 401))
M = 5000

msd_unit = [mean_r2_unit(N, M) for N in Ns]
msd_gaus = [mean_r2_gaussian(N, M) for N in Ns]

plt.figure(figsize=(7,5))
plt.plot(Ns, msd_unit, label="unit step")
plt.plot(Ns, msd_gaus, label="Gaussian kappa")
plt.xlabel("N")
plt.ylabel(r"$\langle r^2(N)\rangle$")
plt.title("MSD: unit vs Gaussian step length")
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(Ns, [msd_unit[i]/Ns[i] for i in range(len(Ns))], label="unit: MSD/N")
plt.plot(Ns, [msd_gaus[i]/Ns[i] for i in range(len(Ns))], label="Gaussian: MSD/N")
plt.xlabel("N")
plt.ylabel(r"$\langle r^2(N)\rangle / N$")
plt.title("MSD/N (should approach a constant)")
plt.legend()
plt.show()
