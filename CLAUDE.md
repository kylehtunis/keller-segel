# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

Run the simulation directly from Python:
```python
from keller_segel import KellerSegelParams, run_simulation
result = run_simulation(KellerSegelParams(), progbar=True)
```

Run the full analysis notebook:
```bash
jupyter notebook analysis.ipynb
# or
jupyter nbconvert --to notebook --execute analysis.ipynb
```

## Dependencies

- `fipy` — finite volume PDE solver (fallback backend)
- `numpy`, `matplotlib` — numerics and visualization
- `tqdm` — optional progress bar (activated via `progbar=True`); works with both C++ and FiPy backends

## Architecture

### Model (`keller_segel.py`)

A three-variable Keller-Segel PDE system for *Dictyostelium* chemotaxis on a 2D FiPy `Grid2D`:

- **`KellerSegelParams`** — dataclass holding all model parameters (domain geometry, PDE coefficients, numerical settings, ICs). The three PDE variables have distinct boundary condition types:
  - `rho` (cell density): zero-flux (FiPy default), Gaussian bump IC
  - `c` (cAMP chemoattractant): zero-flux, starts at 0, self-produced by cells
  - `s` (nutrient): Dirichlet BCs (configurable per-edge via `s_boundary` dict), starts at 0

- **`run_simulation()`** — main loop. Each timestep:
  1. Solve `c` to quasi-steady-state (`build_c_equation_steady`)
  2. Solve `s` to quasi-steady-state (`build_s_equation_steady`)
  3. Advance `rho` with `sweep_count` nonlinear sweeps (`build_rho_equation`)
  4. Clamp `rho >= 0`

- **Chemotaxis term**: uses `ExponentialConvectionTerm` (Scharfetter-Gummel scheme) with **volume-filling** (Painter & Hillen 2002): χ is scaled by `(1 - ρ/ρ_max)` so chemotactic flux → 0 as density approaches carrying capacity, preventing finite-time blow-up. Both FiPy and C++ backends implement this.

- **Growth**: Monod-logistic — `(mu_max/Y)·s·ρ·(1 - ρ/rho_max)` — split into two `ImplicitSourceTerm`s for numerical stability.

### Key design separation

`c` drives chemotaxis (self-produced, no-flux BCs, not consumed) and `s` drives growth (externally supplied via Dirichlet BCs, consumed by cells). Mixing these roles breaks the gradient driving aggregation.

### Notebook (`analysis.ipynb`)

Imports `keller_segel.py`, configures a `KellerSegelParams`, runs the simulation, and produces:
- Static 3×3 snapshot grid (`snapshots.png`)
- Diagnostics plot: total mass and max density over time (`diagnostics.png`)
- Animated HTML5 video of all three fields

Output PNGs are gitignored.

## Numerical Notes

- Reduce `dt` (e.g., `0.001`) when `chi` is large to avoid blow-up
- `sweep_count=5` nonlinear sweeps per step handles the nonlinear coupling between `rho` and `c`
- `get_2d_array` reshapes FiPy's flat cell data to `(ny, nx)` with Fortran order: `reshape((nx, ny), order="F").T`
- `s_boundary` accepts either a float (uniform on all edges) or a dict with keys `"left"`, `"right"`, `"top"`, `"bottom"` for spatially heterogeneous nutrient supply
