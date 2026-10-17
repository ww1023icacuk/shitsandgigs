"""
Demo: how osteoclast and osteoblast activity drive net bone growth or loss
on a neutral substrate.

"Neutral substrate" means a uniform bone surface with an even nutrient supply,
so every kinetic parameter is the same everywhere. The only spatial structure
comes from where remodelling events happen to start. All four scenarios use the
same random events (same seed), so any difference between them comes only from
the change in cell activity.

Scenarios (k1 = bone resorbed per osteoclast, k2 = bone formed per osteoblast):
    balanced              paper values: each cycle refills what it resorbed
    osteoclast-dominant   k1 x 1.5: pits are dug deeper than they are refilled
    osteoblast-dominant   k2 x 1.5: pits are overfilled -> net bone gain
    high turnover         k1 and k2 x 1.5: deeper pits, still fully refilled

Outputs bone_demo.png (summary figure) and bone_demo.gif (side-by-side animation).
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

from bone_remodelling import Params, SpatialParams, kinetics, simulate_spatial, steady_state

BASE = Params()
SCENARIOS = {
    "balanced": BASE,
    "osteoclast-dominant\n(k1 × 1.5)": replace(BASE, k1=1.5 * BASE.k1),
    "osteoblast-dominant\n(k2 × 1.5)": replace(BASE, k2=1.5 * BASE.k2),
    "high turnover\n(k1, k2 × 1.5)": replace(BASE, k1=1.5 * BASE.k1, k2=1.5 * BASE.k2),
}
COLOURS = ["black", "tab:red", "tab:blue", "tab:purple"]


def single_site(p, t_end=200.0):
    """One remodelling cycle at a single site (the original ODE model)."""
    Cs, Bs = steady_state(p)
    sol = solve_ivp(lambda t, y: kinetics(*y[:2], p, Cs, Bs), (0, t_end),
                    [Cs + p.dC0, Bs, 100.0], method="LSODA", max_step=0.5,
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-rate", type=float, default=0.3,
                    help="remodelling events per day on the patch (default 0.3)")
    ap.add_argument("--t-end", type=float, default=720.0, help="days (default 720)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="bone_demo")
    args = ap.parse_args()

    s = SpatialParams(event_rate=args.event_rate, t_end=args.t_end,
                      n_frames=90, rng_seed=args.rng_seed)
    runs = {}
    for name, p in SCENARIOS.items():
        print(f"--- {name.splitlines()[0]}")
        times, z, _, _ = simulate_spatial(p, s)
        runs[name] = z - 100.0          # net change in bone mass [% points]
    lim = max(np.abs(z).max() for z in runs.values())
    extent = [0, s.L, 0, s.L]

    # Summary figure: final maps, then mean bone mass and single-site cycles
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8])
    for col, (name, dz) in enumerate(runs.items()):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(dz[-1], origin="lower", extent=extent, cmap="RdBu",
                       vmin=-lim, vmax=lim)
        ax.set_title(f"{name}\nmean change {dz[-1].mean():+.2f} %")
        ax.set_xlabel("x [mm]")
        if col == 0:
            ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=fig.axes[:4], label="bone mass change [% points]\n"
                 "red = net loss, blue = net gain", shrink=0.9)

    ax_mean = fig.add_subplot(gs[1, :2])
    ax_site = fig.add_subplot(gs[1, 2:])
    for (name, dz), colour in zip(runs.items(), COLOURS):
        label = name.replace("\n", " ")
        ax_mean.plot(times, 100 + dz.mean(axis=(1, 2)), color=colour, label=label)
        t, z = single_site(SCENARIOS[name])
        ax_site.plot(t, z, color=colour, label=label)
    ax_mean.set_xlabel("time [days]")
    ax_mean.set_ylabel("mean bone mass over patch [%]")
    ax_mean.set_title("whole surface: effects add up over many events")
    ax_mean.legend(fontsize=8)
    ax_site.set_xlabel("time [days]")
    ax_site.set_ylabel("bone mass at one site [%]")
    ax_site.set_title("one remodelling cycle: resorption, then formation")
    ax_site.axhline(100, color="grey", lw=0.8, ls="--")
    fig.suptitle(f"Osteoclast vs osteoblast activity on a neutral substrate "
                 f"(same {s.event_rate:g} events/day in every run)")
    fig.savefig(f"{args.out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}.png")

    # Side-by-side animation of the four scenarios
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    ims = []
    for ax, name in zip(axes, runs):
        ims.append(ax.imshow(runs[name][0], origin="lower", extent=extent,
                             cmap="RdBu", vmin=-lim, vmax=lim))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[0], ax=axes, label="bone mass change [% points]", shrink=0.85)
    title = fig.suptitle("")

    def draw(i):
        for ax, im, (name, dz) in zip(axes, ims, runs.items()):
            im.set_data(dz[i])
            ax.set_title(f"{name}\nmean {dz[i].mean():+.2f} %", fontsize=10)
        title.set_text(f"t = {times[i]:.0f} days")
        return ims

    FuncAnimation(fig, draw, frames=len(times), interval=90).save(
        f"{args.out}.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved {args.out}.gif")


if __name__ == "__main__":
    main()
