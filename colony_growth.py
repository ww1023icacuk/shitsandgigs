"""
Colony / biofilm spreading over a 2D surface.

Governing equations (see README.md for the derivation):

    du/dt = div( D(u) grad u ) + r * g(n) * u * (1 - u/K)
    dn/dt = Dn * lap(n) - gamma * r * g(n) * u * (1 - u/K)      [optional]

    D(u) = D0 * (u/K)^m        m = 0  -> Fisher-KPP (linear diffusion)
                               m > 0  -> density-dependent spreading (sharp biofilm edge)
    g(n) = n / (n + Kn)        Monod nutrient limitation (g = 1 if nutrient is off)

Boundary conditions: no-flux (zero normal gradient) on all edges.
Numerics: explicit finite differences (forward Euler in time, conservative
central differences in space) with an automatically chosen stable time step.

Units are up to you as long as they are consistent; defaults use mm and hours.
"""

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import ndimage


@dataclass
class Params:
    L: float = 20.0          # domain side length [mm]
    N: int = 256             # grid points per side
    D0: float = 0.01         # cell diffusivity / motility [mm^2/h]
    r: float = 1.0           # growth rate [1/h]
    K: float = 1.0           # carrying capacity (density is reported as u/K)
    m: float = 0.0           # density-dependent diffusion exponent
    n_colonies: int = 40     # number of random seed colonies
    seed_radius: float = 0.2 # initial colony radius [mm]
    t_end: float = 25.0      # simulated time [h]
    n_frames: int = 80       # snapshots saved for the animation
    nutrient: bool = False   # switch on nutrient coupling
    Dn: float = 0.05         # nutrient diffusivity [mm^2/h]
    Kn: float = 0.5          # Monod half-saturation constant
    gamma: float = 1.0       # nutrient consumed per unit biomass grown
    n0: float = 1.0          # initial nutrient concentration
    threshold: float = 0.5   # u/K above which a pixel counts as "colony"
    rng_seed: int = 0


def div_D_grad(u, D, dx):
    """div(D grad u) with no-flux boundaries, conservative face-centred form."""
    up = np.pad(u, 1, mode="edge")
    Dp = np.pad(D, 1, mode="edge")
    c = (slice(1, -1), slice(1, -1))
    # Face diffusivities (arithmetic mean of neighbouring cells)
    De = 0.5 * (Dp[1:-1, 2:] + Dp[c])
    Dw = 0.5 * (Dp[1:-1, :-2] + Dp[c])
    Dn_ = 0.5 * (Dp[2:, 1:-1] + Dp[c])
    Ds = 0.5 * (Dp[:-2, 1:-1] + Dp[c])
    return (De * (up[1:-1, 2:] - u) - Dw * (u - up[1:-1, :-2])
            + Dn_ * (up[2:, 1:-1] - u) - Ds * (u - up[:-2, 1:-1])) / dx**2


def initial_condition(p, x, y, rng):
    u = np.zeros((p.N, p.N))
    margin = 2 * p.seed_radius
    centres = rng.uniform(margin, p.L - margin, size=(p.n_colonies, 2))
    for cx, cy in centres:
        u += p.K * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * p.seed_radius**2))
    return np.minimum(u, p.K)


def simulate(p, u0=None):
    """Integrate the model; returns grid, snapshot times and u/K snapshots.

    u0 overrides the random seeded initial condition if given."""
    rng = np.random.default_rng(p.rng_seed)
    dx = p.L / p.N
    xs = (np.arange(p.N) + 0.5) * dx
    x, y = np.meshgrid(xs, xs)

    u = initial_condition(p, x, y, rng) if u0 is None else u0.copy()
    n = np.full_like(u, p.n0) if p.nutrient else None

    # Stability: diffusion needs dt <= dx^2 / (4 D_max); keep growth step small too.
    D_max = max(p.D0, p.Dn if p.nutrient else 0.0)
    dt = min(0.2 * dx**2 / D_max, 0.05 / p.r)
    n_steps = int(np.ceil(p.t_end / dt))
    dt = p.t_end / n_steps
    save_every = max(1, n_steps // p.n_frames)

    times, frames = [0.0], [u / p.K]
    for step in range(1, n_steps + 1):
        D = p.D0 * (np.clip(u, 0, None) / p.K) ** p.m if p.m else np.full_like(u, p.D0)
        g = n / (n + p.Kn) if p.nutrient else 1.0
        growth = p.r * g * u * (1 - u / p.K)
        u = u + dt * (div_D_grad(u, D, dx) + growth)
        if p.nutrient:
            n = np.clip(n + dt * (div_D_grad(n, np.full_like(n, p.Dn), dx)
                                  - p.gamma * growth), 0, None)
        if step % save_every == 0 or step == n_steps:
            times.append(step * dt)
            frames.append(u / p.K)

    print(f"dx = {dx:.4g}, dt = {dt:.4g}, steps = {n_steps}")
    return xs, np.array(times), np.array(frames)


def colony_areas(frame, threshold, dx):
    """Areas of connected colonies where u/K > threshold."""
    labels, count = ndimage.label(frame > threshold)
    if count == 0:
        return np.array([])
    return np.bincount(labels.ravel())[1:] * dx**2


def animate(p, xs, times, frames, out):
    dx = xs[1] - xs[0]
    areas = [colony_areas(f, p.threshold, dx) for f in frames]
    a_max = max((a.max() for a in areas if a.size), default=1.0)
    bins = np.linspace(0, a_max, 25)
    extent = [0, p.L, 0, p.L]

    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.8),
                                          gridspec_kw={"width_ratios": [1, 1.1]})
    im = ax_map.imshow(frames[0], origin="lower", extent=extent,
                       cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax_map, label="density  u / K")
    ax_map.set_xlabel("x [mm]")
    ax_map.set_ylabel("y [mm]")
    title = ax_map.set_title("")

    def draw(i):
        im.set_data(frames[i])
        title.set_text(f"t = {times[i]:.1f} h   coverage = {np.mean(frames[i] > p.threshold):.0%}")
        ax_hist.cla()
        ax_hist.hist(areas[i], bins=bins, color="tab:green", edgecolor="black")
        ax_hist.set_xlim(0, a_max)
        ax_hist.set_xlabel("colony area [mm²]")
        ax_hist.set_ylabel("number of colonies")
        ax_hist.set_title(f"size distribution ({areas[i].size} colonies)")
        return im, title

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"saved {out}")


def snapshot_figure(p, xs, times, frames, out, n_snap=4):
    dx = xs[1] - xs[0]
    idx = np.linspace(0, len(frames) - 1, n_snap).astype(int)
    fig, axes = plt.subplots(2, n_snap, figsize=(3.2 * n_snap, 6.4))
    for col, i in enumerate(idx):
        axes[0, col].imshow(frames[i], origin="lower", extent=[0, p.L, 0, p.L],
                            cmap="viridis", vmin=0, vmax=1)
        axes[0, col].set_title(f"t = {times[i]:.1f} h")
        a = colony_areas(frames[i], p.threshold, dx)
        axes[1, col].hist(a, bins=15, color="tab:green", edgecolor="black")
        axes[1, col].set_xlabel("colony area [mm²]")
    axes[1, 0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def front_speed_check(p):
    """Single-colony run: compare measured front speed with Fisher-KPP 2*sqrt(D r)."""
    q = Params(**{**p.__dict__, "n_colonies": 0, "m": 0.0, "nutrient": False})
    dx = q.L / q.N
    xs = (np.arange(q.N) + 0.5) * dx
    x, y = np.meshgrid(xs, xs)
    u0 = q.K * np.exp(-((x - q.L / 2) ** 2 + (y - q.L / 2) ** 2) / (2 * q.seed_radius**2))
    _, times, frames = simulate(q, u0=u0)
    radius = np.sqrt([np.sum(f > q.threshold) * dx**2 / np.pi for f in frames])
    # Fit only while the front is well inside the domain and past the transient
    ok = (radius > 1.0) & (radius < 0.4 * q.L)
    speed = np.polyfit(times[ok], radius[ok], 1)[0]
    theory = 2 * np.sqrt(q.D0 * q.r)
    print(f"front speed: measured {speed:.4g}, Fisher-KPP theory {theory:.4g} "
          f"({speed / theory:.0%} of theory; 2D curvature + transients make it slightly lower)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Params()
    for name, val in defaults.__dict__.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true", default=val)
        else:
            ap.add_argument(flag, type=type(val), default=val)
    ap.add_argument("--out", default="colony_growth", help="output file prefix")
    ap.add_argument("--check", action="store_true",
                    help="run the single-colony front-speed check instead")
    args = vars(ap.parse_args())
    out, check = args.pop("out"), args.pop("check")
    p = Params(**args)

    if check:
        front_speed_check(p)
        return
    xs, times, frames = simulate(p)
    animate(p, xs, times, frames, f"{out}.gif")
    snapshot_figure(p, xs, times, frames, f"{out}_snapshots.png")


if __name__ == "__main__":
    main()
