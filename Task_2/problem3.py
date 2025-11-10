import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x**2 + y**2) * (x**2 + y**2 <= 1.0) * (x >= 1/3) * (y >= 1/5)

def problem_3a(N = 50_000, alpha = 1.0, seed = 1):
    r = np.random.default_rng(seed)
    acc_total = 0
    xs, ys = [], []
    trials = 0

    while acc_total < N:
        x = r.random(N)
        y = r.random(N)
        w = f(x, y) / alpha
        u = r.random(N)
        keep = u < w
        xs.append(x[keep]); ys.append(y[keep])
        acc_total += keep.sum()
        trials += N

    x = np.concatenate(xs)[:N]
    y = np.concatenate(ys)[:N]
    acc_ratio = N / trials
    print(f"[3a] acceptance ≈ {acc_ratio:.3f}")

    t = np.linspace(0, np.pi/2, 300)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=5, alpha=0.7, label="accepted")
    plt.plot(np.cos(t), np.sin(t), lw=1.5, label=r"x^2+y^2=1")
    plt.axvline(1/3, ls="--", lw=1); plt.axhline(1/5, ls="--", lw=1)
    plt.gca().set_aspect('equal', 'box'); plt.xlim(0,1); plt.ylim(0,1)
    plt.title(r"Problem 3(a): $g_{\alpha}$ ≡ const, $\alpha$=1")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

    return x, y, acc_ratio

# ============== 3(b): g_b(x,y) = β (x+y) на [0,1]^2 =================
def sample_y_from_marginal(n, r):
    u = r.random(n)
    return 0.5 * (-1.0 + np.sqrt(1.0 + 8.0*u))

def sample_x_given_y(y, r):
    u = r.random(len(y))
    return -y + np.sqrt(y*y + 2.0*u*(0.5 + y))

def problem_3b(N = 50_000, beta = 0.848, seed = 2):
    r = np.random.default_rng(seed)
    xs, ys = [], []
    trials = 0

    while sum(len(a) for a in xs) < N:
        y = sample_y_from_marginal(60_000, r)
        x = sample_x_given_y(y, r)
        denom = beta * (x + y)
        w = np.where(denom > 0, f(x, y) / denom, 0.0)
        u = r.random(len(w))
        keep = u < w
        xs.append(x[keep]); ys.append(y[keep])
        trials += len(w)

    x = np.concatenate(xs)[:N]
    y = np.concatenate(ys)[:N]
    acc_ratio = N / trials
    print(f"[3b] acceptance ≈ {acc_ratio:.3f}  (β = {beta})")

    t = np.linspace(0, np.pi/2, 300)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=5, alpha=0.7, label="accepted")
    plt.plot(np.cos(t), np.sin(t), lw=1.5, label="x^2+y^2=1")
    plt.axvline(1/3, ls="--", lw=1); plt.axhline(1/5, ls="--", lw=1)
    plt.gca().set_aspect('equal', 'box'); plt.xlim(0,1); plt.ylim(0,1)
    plt.title(r"Problem 3(b): $g_{\beta}$ ∝ (x+y), $\beta$≈0.848")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

    return x, y, acc_ratio

if __name__ == "__main__":
    xa, ya, acc_a = problem_3a(N=40_000, alpha=1.0, seed=11)
    xb, yb, acc_b = problem_3b(N=40_000, beta=0.848, seed=12)
    
