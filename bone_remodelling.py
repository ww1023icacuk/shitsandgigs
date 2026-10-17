"""
Bone remodelling: osteoclast / osteoblast model of Komarova et al. (2003),
"Mathematical model predicts a critical role for osteoclast autocrine regulation
in the control of bone remodeling", Bone 33:206-215.

    dC/dt = a1 * C^g11 * B^g21 - b1 * C          osteoclasts
    dB/dt = a2 * C^g12 * B^g22 - b2 * B          osteoblasts
    dz/dt = -k1 * max(C - Cs, 0) + k2 * max(B - Bs, 0)    bone mass [% of initial]

g11 = osteoclast autocrine, g21 = osteoblast -> osteoclast paracrine (RANKL/OPG),
g12 = osteoclast -> osteoblast paracrine (coupling), g22 = osteoblast autocrine.
(Cs, Bs) is the non-trivial steady state.

Two modes:
  default    the original well-mixed ODE model (one remodelling site)
  --spatial  an illustrative 2D extension (not in the paper): the same kinetics
             at every point on a bone surface, cells diffuse, and remodelling
             events are triggered at random places and times.
"""

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

from colony_growth import div_D_grad


@dataclass
class Params:
    # Komarova et al. (2003), Table 1
    a1: float = 3.0       # osteoclast production [1/day]
    a2: float = 4.0       # osteoblast production [1/day]
    b1: float = 0.2       # osteoclast removal [1/day]
    b2: float = 0.02      # osteoblast removal [1/day]
    g11: float = 0.5      # osteoclast autocrine
    g21: float = -0.5     # osteoblast -> osteoclast paracrine
    g12: float = 1.0      # osteoclast -> osteoblast paracrine
    g22: float = 0.0      # osteoblast autocrine
    k1: float = 0.24      # resorption rate [% / day / cell]
    k2: float = 0.0017    # formation rate [% / day / cell]
    dC0: float = 10.0     # initial osteoclast perturbation above steady state
    t_end: float = 1000.0 # [days]


def steady_state(p):
    """Non-trivial steady state (Cs, Bs) from setting dC/dt = dB/dt = 0."""
    gam = p.g12 * p.g21 - (1 - p.g11) * (1 - p.g22)
    r1, r2 = p.b1 / p.a1, p.b2 / p.a2
    Cs = r1 ** ((1 - p.g22) / gam) * r2 ** (p.g21 / gam)
    Bs = r1 ** (p.g12 / gam) * r2 ** ((1 - p.g11) / gam)
    return Cs, Bs


def stability(p):
    """Eigenvalues of the (C, B) Jacobian at the steady state."""
    Cs, Bs = steady_state(p)
    J = np.array([[p.b1 * (p.g11 - 1), p.b1 * p.g21 * Cs / Bs],
                  [p.b2 * p.g12 * Bs / Cs, p.b2 * (p.g22 - 1)]])
    return np.linalg.eigvals(J)


def kinetics(C, B, p, Cs, Bs):
    """Right-hand sides of the Komarova model; works on scalars or arrays."""
    C = np.maximum(C, 1e-12)
    B = np.maximum(B, 1e-12)
    dC = p.a1 * C**p.g11 * B**p.g21 - p.b1 * C
    dB = p.a2 * C**p.g12 * B**p.g22 - p.b2 * B
    dz = -p.k1 * np.maximum(C - Cs, 0) + p.k2 * np.maximum(B - Bs, 0)
    return dC, dB, dz


def run_ode(p, out):
    Cs, Bs = steady_state(p)
    eig = stability(p)
    if np.all(np.abs(eig.real) < 1e-9):
        kind = "neutral (sustained cycles)"
    elif np.all(eig.real < 0):
        kind = "stable (damped)"
    else:
        kind = "unstable (oscillations grow)"
    print(f"steady state: C* = {Cs:.4g}, B* = {Bs:.4g}")
    print(f"eigenvalues: {np.round(eig, 5)}  -> {kind}")

    sol = solve_ivp(lambda t, y: kinetics(*y[:2], p, Cs, Bs),
                    (0, p.t_end), [Cs + p.dC0, Bs, 100.0],
                    method="LSODA", max_step=0.5, rtol=1e-8, atol=1e-10)
    t, (C, B, z) = sol.t, sol.y
    print(f"bone mass at t = {p.t_end:g} d: {z[-1]:.2f} %  (min {z.min():.2f} %)")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(t, C, color="tab:red")
    ax[0, 0].axhline(Cs, ls="--", color="grey")
    ax[0, 0].set_ylabel("osteoclasts C")
    ax[0, 1].plot(t, B, color="tab:blue")
    ax[0, 1].axhline(Bs, ls="--", color="grey")
    ax[0, 1].set_ylabel("osteoblasts B")
    ax[1, 0].plot(t, z, color="black")
    ax[1, 0].set_ylabel("bone mass z [%]")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("time [days]")
    ax[1, 1].plot(C, B, color="tab:purple")
    ax[1, 1].plot(Cs, Bs, "ko", label="steady state")
    ax[1, 1].plot(C[0], B[0], "g^", label="start")
    ax[1, 1].set_xlabel("osteoclasts C")
    ax[1, 1].set_ylabel("osteoblasts B")
    ax[1, 1].legend()
    fig.suptitle(f"Komarova et al. (2003) model, g11 = {p.g11}: {kind}")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"saved {out}.png")


@dataclass
class SpatialParams:
    L: float = 2.0           # side length of the bone surface patch [mm]
    N: int = 100             # grid points per side
    DC: float = 1e-5         # osteoclast diffusivity [mm^2/day] (osteoclasts migrate)
    DB: float = 1e-6         # osteoblast diffusivity [mm^2/day] (nearly sessile; larger values
                             # spread formation away from the pit so it never fully refills)
    event_rate: float = 0.05 # new remodelling events per day on the whole patch
    event_radius: float = 0.08  # size of the initial osteoclast cluster [mm]
    t_end: float = 600.0     # [days]
    n_frames: int = 120
    rng_seed: int = 0


def simulate_spatial(p, s):
    """Integrate the 2D model. Returns snapshot times and z, C, B frames.

    Remodelling events depend only on s.rng_seed, so runs with different
    kinetic parameters but the same seed see identical events."""
    Cs, Bs = steady_state(p)
    rng = np.random.default_rng(s.rng_seed)
    dx = s.L / s.N
    xs = (np.arange(s.N) + 0.5) * dx
    x, y = np.meshgrid(xs, xs)

    C = np.full((s.N, s.N), Cs)
    B = np.full((s.N, s.N), Bs)
    z = np.full((s.N, s.N), 100.0)
    DC = np.full_like(C, s.DC)
    DB = np.full_like(B, s.DB)

    dt = min(0.2 * dx**2 / max(s.DC, s.DB), 0.1)
    n_steps = int(np.ceil(s.t_end / dt))
    dt = s.t_end / n_steps
    save_every = max(1, n_steps // s.n_frames)
    print(f"dx = {dx:.4g} mm, dt = {dt:.4g} d, steps = {n_steps}")

    times, z_frames, C_frames, B_frames = [], [], [], []
    n_events = 0
    for step in range(n_steps + 1):
        if step % save_every == 0 or step == n_steps:
            times.append(step * dt)
            z_frames.append(z.copy())
            C_frames.append(C.copy())
            B_frames.append(B.copy())
        if step == n_steps:
            break
        # Poisson-distributed remodelling events (e.g. microdamage) start BMUs
        for _ in range(rng.poisson(s.event_rate * dt)):
            cx, cy = rng.uniform(0, s.L, size=2)
            C += p.dC0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s.event_radius**2))
            n_events += 1
        dC, dB, dz = kinetics(C, B, p, Cs, Bs)
        C = np.maximum(C + dt * (dC + div_D_grad(C, DC, dx)), 1e-12)
        B = np.maximum(B + dt * (dB + div_D_grad(B, DB, dx)), 1e-12)
        z = z + dt * dz

    print(f"{n_events} remodelling events; mean bone mass at end {z.mean():.2f} %")
    return np.array(times), *map(np.array, (z_frames, C_frames, B_frames))


def run_spatial(p, s, out):
    Cs, Bs = steady_state(p)
    times, z_frames, C_frames, B_frames = simulate_spatial(p, s)
    z_lo = min(z_frames.min(), 99.0)
    z_hi = max(z_frames.max(), 101.0)
    extent = [0, s.L, 0, s.L]
    fig, (ax_z, ax_c, ax_b) = plt.subplots(1, 3, figsize=(16, 4.8))
    im_z = ax_z.imshow(z_frames[0], origin="lower", extent=extent, cmap="bone",
                       vmin=z_lo, vmax=z_hi)
    fig.colorbar(im_z, ax=ax_z, label="bone mass z [%]")
    im_c = ax_c.imshow(C_frames[0], origin="lower", extent=extent, cmap="Reds",
                       vmin=0, vmax=C_frames.max())
    fig.colorbar(im_c, ax=ax_c, label="osteoclasts C")
    im_b = ax_b.imshow(B_frames[0], origin="lower", extent=extent, cmap="Blues",
                       vmin=Bs, vmax=B_frames.max())
    fig.colorbar(im_b, ax=ax_b, label="osteoblasts B")
    for a in (ax_z, ax_c, ax_b):
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")
    title = fig.suptitle("")

    def draw(i):
        im_z.set_data(z_frames[i])
        im_c.set_data(C_frames[i])
        im_b.set_data(B_frames[i])
        title.set_text(f"t = {times[i]:.0f} days   mean bone mass = {z_frames[i].mean():.2f} %")
        return im_z, im_c, im_b, title

    fig.tight_layout()
    FuncAnimation(fig, draw, frames=len(times), interval=80).save(
        f"{out}.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"saved {out}.gif")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in {**Params().__dict__, **SpatialParams().__dict__}.items():
        if name == "t_end":
            continue
        ap.add_argument("--" + name.replace("_", "-"), type=type(val), default=val)
    ap.add_argument("--t-end", type=float, default=None,
                    help="days to simulate (default 1000 for ODE, 600 for --spatial)")
    ap.add_argument("--spatial", action="store_true", help="run the 2D extension")
    ap.add_argument("--out", default=None, help="output file prefix")
    args = vars(ap.parse_args())
    spatial, out, t_end = args.pop("spatial"), args.pop("out"), args.pop("t_end")

    p = Params(**{k: v for k, v in args.items() if k in Params.__dataclass_fields__})
    if spatial:
        s = SpatialParams(**{k: v for k, v in args.items()
                             if k in SpatialParams.__dataclass_fields__})
        if t_end is not None:
            s.t_end = t_end
        run_spatial(p, s, out or "bone_remodelling_2d")
    else:
        if t_end is not None:
            p.t_end = t_end
        run_ode(p, out or "bone_remodelling")


if __name__ == "__main__":
    main()
