import numpy as np
from scipy.integrate import quad

def f(x, y):
    return (x**2 + y**2) * (x**2 + y**2 <= 1.0) * (x >= 1/3) * (y >= 1/5)

def sample_flat(n, rng):
    return rng.random(n), rng.random(n)

def sample_gb(n, rng):
    u = rng.random(n)
    y = 0.5 * (-1.0 + np.sqrt(1.0 + 8.0*u)) # Y ~ (1/2)+y
    u = rng.random(n)
    x = -y + np.sqrt(y*y + 2.0*u*(0.5 + y)) # X|Y ~ x+y
    return x, y

def mc_integrate(N, sampler, g_eval, seed=1):
    r = np.random.default_rng(seed)
    x, y = sampler(N, r)
    w = f(x, y) / g_eval(x, y)
    I = w.mean()
    err = np.sqrt((np.mean(w*w) - I**2) / (N - 1))
    return I, err

# g evaluators
g_flat = lambda x,y: np.ones_like(x)
g_b = lambda x,y: (x + y)

def exact_integral(nx=1500, ny=1500):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    F = (X**2 + Y**2) * (X**2 + Y**2 <= 1.0) * (X >= 1/3) * (Y >= 1/5)
    Iy = np.trapezoid(F, y)
    return np.trapezoid(Iy, x)

if __name__ == "__main__":
    N = 2_000_000
    I_flat, err_flat = mc_integrate(N, sample_flat, g_flat, seed=11)
    I_gb,   err_gb   = mc_integrate(N, sample_gb,   g_b,   seed=12)
    I_exact = exact_integral()

    print(f"\nExact I = {I_exact:.8f}")
    print(f"Flat g=1: I≈{I_flat:.8f} ± {err_flat:.2e}")
    print(f"Imp. g_b=x+y: I≈{I_gb:.8f} ± {err_gb:.2e} \n")

    # for seed in range(10):
    #     I_flat, err_flat = mc_integrate(N, sample_flat, g_flat, seed)
    #     I_gb,   err_gb   = mc_integrate(N, sample_gb,   g_b,   seed)
    #     print(f"{seed:2d} | flat {I_flat:.6f} | gb {I_gb:.6f}")