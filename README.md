# shitsandgigs: growth models on surfaces

Two models:

1. **Colony / biofilm spreading** ([colony_growth.py](colony_growth.py)): a Fisher–KPP
   reaction–diffusion model with a 2D density map and the colony size distribution.
2. **Bone remodelling** ([bone_remodelling.py](bone_remodelling.py)): the osteoclast–osteoblast
   model of Komarova et al. (2003), plus an illustrative 2D extension. See
   [below](#bone-remodelling-komarova-et-al-2003).

---

## Governing equation

Let `u(x, y, t)` be the cell (biomass) density per unit area. Conservation of cells in any
patch of surface says the density changes only because cells move in or out (a flux **J**)
or are born (a source `f`):

    ∂u/∂t = −∇·J + f(u)

**Movement:** random motility gives a Fickian flux, `J = −D(u) ∇u`.
**Growth:** logistic growth: exponential at low density, saturating at a carrying capacity `K`
(limited by space and resources): `f(u) = r u (1 − u/K)`.

Putting them together:

    ∂u/∂t = ∇·( D(u) ∇u ) + r u (1 − u/K)

* With constant `D(u) = D₀` this is the **Fisher–KPP equation**. It gives travelling fronts
  with speed **c = 2√(D₀ r)**, which is the main check that the code is right.
* With `D(u) = D₀ (u/K)^m`, `m > 0`, cells only spread where there are already many of them
  (e.g. collective motion or EPS-driven expansion in a biofilm). This gives a sharp colony edge
  instead of a diffuse one.

### Optional nutrient limitation

A nutrient field `n` diffuses through the substrate and is consumed by growth:

    ∂u/∂t = ∇·( D(u) ∇u ) + r g(n) u (1 − u/K)
    ∂n/∂t = Dₙ ∇²n − γ r g(n) u (1 − u/K),        g(n) = n / (n + Kₙ)   (Monod)

Colonies slow down as they use up the nutrient around them.

### Boundary and initial conditions

* No-flux on all edges (`∂u/∂n = 0`): cells can't leave the surface.
* `n_colonies` small Gaussian seeds placed at random positions.

### Non-dimensional form

Scaling `u → u/K`, `t → r t`, `x → x √(r/D₀)` gives `∂u/∂t = ∇²u + u(1 − u)`.
So with no nutrient and `m = 0`, the only things that change the picture are the domain size
relative to `√(D₀/r)` and how many seeds there are.

## Numerical method

* Finite differences on an `N × N` grid. The diffusion term uses the conservative,
  face-centred form, so it also handles `D(u)`.
* Forward Euler time stepping, with `dt ≤ 0.2 dx² / D_max` (stability needs `dx²/4D`) and
  `dt ≤ 0.05 / r`.
* **Colony sizes:** pixels with `u/K > threshold` are grouped into connected regions
  (`scipy.ndimage.label`) and the area of each region is measured. When colonies merge, the
  count drops.

## Usage

```bash
pip install -r requirements.txt
python colony_growth.py                        # Fisher-KPP, 40 random colonies
python colony_growth.py --check                # front speed vs 2√(D r)
python colony_growth.py --nutrient --m 1 --D0 0.02 --out biofilm
python colony_growth.py --help                 # all parameters
```

Each run writes `<out>.gif` (an animated density map next to a live size histogram), the same
animation as `<out>.mp4` (H.264), and `<out>_snapshots.png`. The MP4 uses a system `ffmpeg` if
one is on the PATH, otherwise the binary bundled with `imageio-ffmpeg`; with neither, it is skipped.

Default units are mm and hours: `D₀ = 0.01 mm²/h` and `r = 1 /h`, so fronts move at 0.2 mm/h.
These are placeholders; swap in measured values for your organism.

## Why Python rather than MATLAB

* **Free and portable:** anyone can clone and run it without a licence.
* `numpy` / `scipy` / `matplotlib` cover everything needed here, including connected-component
  labelling for colony sizes and GIF export.
* MATLAB would work just as well numerically. Its built-in `pdepe` is 1D only, so you'd write
  the same finite-difference loop by hand either way. It's mainly worth choosing if your group
  or course already standardises on it.

---

## Bone remodelling: Komarova et al. (2003)

Komarova S.V., Smith R.J., Dixon S.J., Sims S.M., Wahl L.M. (2003). *Mathematical model predicts a
critical role for osteoclast autocrine regulation in the control of bone remodeling.*
Bone 33(2):206–215.

### Equations

At one remodelling site, the osteoclast count `C`, the osteoblast count `B` and the bone mass `z`
(% of initial) follow

    dC/dt = α₁ C^g₁₁ B^g₂₁ − β₁ C
    dB/dt = α₂ C^g₁₂ B^g₂₂ − β₂ B
    dz/dt = −k₁ max(C − C*, 0) + k₂ max(B − B*, 0)

* The first term in each cell equation is production. The power laws describe signalling between
  the cell types, with each exponent setting the strength and sign of one interaction:
  * `g₁₁ = 0.5`: osteoclast autocrine signalling (osteoclasts stimulating more osteoclasts).
  * `g₂₁ = −0.5`: osteoblasts inhibiting osteoclast formation, via the RANKL/OPG balance.
  * `g₁₂ = 1`: osteoclasts stimulating osteoblasts. This couples resorption to formation (e.g.
    through factors released from the matrix such as TGF-β).
  * `g₂₂ = 0`: no osteoblast autocrine signalling.
* The second term in each cell equation is cell removal (apoptosis).
* Bone is resorbed in proportion to the osteoclasts above their steady-state number, and formed in
  proportion to the osteoblasts above theirs.

**Steady state:** setting both cell equations to zero, with `Γ = g₁₂g₂₁ − (1−g₁₁)(1−g₂₂)`, gives

    C* = (β₁/α₁)^((1−g₂₂)/Γ) · (β₂/α₂)^(g₂₁/Γ)
    B* = (β₁/α₁)^(g₁₂/Γ) · (β₂/α₂)^((1−g₁₁)/Γ)

**Stability:** set by the eigenvalues of the Jacobian at `(C*, B*)`. The trace is
`β₁(g₁₁−1) + β₂(g₂₂−1)`, so osteoclast autocrine signalling (`g₁₁`) controls whether a perturbation
decays (a single remodelling cycle) or keeps cycling. That is the paper's main conclusion.

### Results with the paper's parameters

`α₁=3, α₂=4, β₁=0.2, β₂=0.02, k₁=0.24, k₂=0.0017`, with an initial perturbation of 10 extra
osteoclasts:

| `g₁₁` | Behaviour | Bone mass |
|---|---|---|
| 0.5 (default) | A single damped cycle: resorption, then formation | Drops to ~90%, returns to ~100% |
| 1.1 | Eigenvalues purely imaginary, so the cycles never die out | Each cycle loses ~10%, because `k₁`/`k₂` are balanced for the `g₁₁ = 0.5` case |

The code computes `C* ≈ 1.06` and `B* ≈ 212` for the default parameters, matching the paper.

### 2D extension (not in the paper)

`--spatial` uses the same kinetics at every point of a 2 mm × 2 mm bone surface, with cells moving
by diffusion, and starts remodelling events (e.g. after microdamage) at random places and times:

    ∂C/∂t = D_C ∇²C + (Komarova kinetics)
    ∂B/∂t = D_B ∇²B + (Komarova kinetics)

The animation shows resorption pits opening as osteoclasts gather, osteoblasts arriving, and the
pits refilling.

Osteoblasts have to stay close to where they formed. If `D_B` is too large, they spread out and
deposit bone in a ring around the pit, so the pit never fully refills. The defaults
(`D_C = 1e-5`, `D_B = 1e-6` mm²/day) give ~99.8% refill. This is an illustrative extension, not a
validated model. For published spatial versions, see Ryser, Nigam & Komarova (2009, J. Bone
Miner. Res.) and Buenzli et al. (2011, J. Theor. Biol.), which model a moving BMU.

### Usage

```bash
python bone_remodelling.py                 # original ODE model -> bone_remodelling.png
python bone_remodelling.py --g11 1.1       # sustained remodelling cycles
python bone_remodelling.py --spatial       # 2D bone surface -> bone_remodelling_2d.gif
python bone_remodelling.py --help
```
