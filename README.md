# shitsandgigs: colony growth over a surface

Simulates and visualises the density of a colony / biofilm spreading over a 2D surface,
plus the size distribution of the colonies as they grow and merge.

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

Each run writes `<out>.gif` (an animated density map next to a live size histogram) and
`<out>_snapshots.png`.

Default units are mm and hours: `D₀ = 0.01 mm²/h` and `r = 1 /h`, so fronts move at 0.2 mm/h.
These are placeholders; swap in measured values for your organism.

## Why Python rather than MATLAB

* **Free and portable:** anyone can clone and run it without a licence.
* `numpy` / `scipy` / `matplotlib` cover everything needed here, including connected-component
  labelling for colony sizes and GIF export.
* MATLAB would work just as well numerically. Its built-in `pdepe` is 1D only, so you'd write
  the same finite-difference loop by hand either way. It's mainly worth choosing if your group
  or course already standardises on it.
