# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

Run the simulation directly from Python:
```python
from keller_segel import DimensionalParams, run_simulation
dim = DimensionalParams.growth_phase()      # or .aggregation_phase()
result = run_simulation(dim.to_dimensionless(), progbar=True)
```

Or with raw dimensionless params:
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

## Building the C++ Backend

The C++ Eigen/pybind11 solver is optional but ~10-50× faster than FiPy. Build with:
```bash
pip install -e .
```
This uses scikit-build-core + CMake. Eigen 3.4 is fetched automatically. If `_ks_core` fails to import, the code falls back to FiPy transparently.

## Dependencies

- **C++ backend** (preferred): Eigen 3.4 (fetched by CMake), pybind11, scikit-build-core
- `fipy` — finite volume PDE solver (fallback backend)
- `numpy`, `matplotlib` — numerics and visualization
- `tqdm` — optional progress bar (activated via `progbar=True`); works with both C++ and FiPy backends

## Architecture

### Model (`keller_segel.py`)

A three-variable Keller-Segel PDE system for *Dictyostelium* chemotaxis on a 2D grid:

- **`DimensionalParams`** — dataclass holding physical-unit parameters (mm, min, nM, µg/mL, cells/mm²). Has `.to_dimensionless()` to convert to `KellerSegelParams`. Two presets:
  - `.growth_phase()` — fed cells, weak χ (1.8e-4), nutrient at boundaries, 10 hr
  - `.aggregation_phase()` — starved cells, strong χ (1.8e-3), no nutrient, 2 hr

- **`KellerSegelParams`** — dimensionless solver params. Nondimensionalization uses L₀=√(D_c/β), T₀=1/β, ρ₀=ρ_max, s₀=s_boundary, so D_c=β=α=ρ_max=1 in code units.

- **`run_simulation()`** — dispatches to C++ (`_ks_core`) or FiPy. Each timestep:
  1. Solve `c` to quasi-steady-state (`build_c_equation_steady`)
  2. Solve `s` to quasi-steady-state (`build_s_equation_steady`)
  3. Advance `rho` with `sweep_count` nonlinear sweeps (`build_rho_equation`)
  4. Clamp `rho >= 0`

- **Chemotaxis term**: Scharfetter-Gummel scheme (`ExponentialConvectionTerm` in FiPy, Bernoulli-function discretization in C++) with **volume-filling** (Painter & Hillen 2002): χ scaled by `(1 - ρ/ρ_max)` so chemotactic flux → 0 as density approaches carrying capacity, preventing finite-time blow-up. Both backends implement this.

- **Growth**: Monod-logistic — `(mu_max/Y)·s·ρ·(1 - ρ/rho_max)` — split into two `ImplicitSourceTerm`s for numerical stability.

- **Custom initial conditions**: `rho_initial` (2D array, shape `(ny, nx)`) can be passed to override the default Gaussian bump IC. `DimensionalParams` accepts `rho_initial_cells_per_mm2` in physical units.

- **Multi-bump IC**: `n_bumps` (int, default 1) places that many Gaussian bumps at random positions. With `n_bumps=1` the bump stays at the domain centre (legacy behaviour). `rho_bump_seed` (int or None) seeds the RNG for reproducibility. The helper `_make_rho_ic(params)` generates the `(ny, nx)` IC array and is used by both backends.

- **s carry-over between phases**: `s_initial` (2D array) sets the initial `s` field (useful for chaining phases). `s_no_flux_bc` (bool, default False) switches `s` from Dirichlet to zero-Neumann boundary conditions — used for the aggregation phase where starvation seals the domain (no nutrient source at boundaries). `DimensionalParams` exposes `s_initial_ug_per_mL` and `s_no_flux_bc`. The notebook chains the two phases by passing `result_growth.s_snapshots[-1] * dim_growth.s_boundary_ug_per_mL` as `s_initial_ug_per_mL`.

- **Starvation-dependent chemotaxis**: `chi_s_half_sat` (dimensionless) modulates χ by local nutrient level: `χ_eff = χ / (1 + s / chi_s_half_sat)`. When `s = 0` (starvation), cells chemotax at full strength χ; when `s = chi_s_half_sat`, at half strength. `DimensionalParams` exposes `chi_s_half_sat_ug_per_mL` (physical units; a natural starting value is `K_s_ug_per_mL = 2.0 µg/mL`). Default `0.0` disables the feature (backward-compatible). **C++ backend only** — FiPy backend has a TODO stub.

### Key design separation

`c` drives chemotaxis (self-produced, no-flux BCs, not consumed) and `s` drives growth (externally supplied via Dirichlet BCs, consumed by cells). Mixing these roles breaks the gradient driving aggregation.

### C++ Backend (`src/ks_core/`)

- `ks_solver.hpp/.cpp` — `KSSolver` class: assembles and solves three Eigen sparse linear systems via `SparseLU`
- `sg_scheme.hpp` — Bernoulli function `B(x) = x/(e^x - 1)` with Taylor/asymptotic handling
- `bindings.cpp` — pybind11 module exposing `run_simulation(dict, progress_cb) -> dict`

Three linear systems with different update strategies:
| System | Matrix update | Strategy |
|--------|--------------|----------|
| `c` (cAMP) | Never — `D_c·∇² − β·I` is constant | Factored once at startup |
| `s` (nutrient) | Every step — diagonal depends on ρ | Pattern analyzed once, refactored each step |
| `ρ` (cells) | Every sweep — SG coefficients change | Assembled + factored each sweep |

Progress reporting: C++ calls an optional Python callback `(step, total)` with GIL released during the solve, re-acquired only for the brief callback.

### Notebook (`analysis.ipynb`)

Imports `keller_segel.py`, configures `DimensionalParams` with phase presets, runs the simulation, and produces:
- Static 3×3 snapshot grid (`snapshots.png`)
- Diagnostics plot: total mass and max density over time (`diagnostics.png`)
- Animated HTML5 video of all three fields

Output PNGs are gitignored.

## Grid Indexing Convention

Both backends use flat index `k = j * nx + i` (row-major for a (ny, nx) grid). This matches FiPy's `Grid2D` convention. Snapshots are stored as `(ny, nx)` numpy arrays. The FiPy helper `get_2d_array` does `reshape((nx, ny), order="F").T`, which is equivalent to `reshape((ny, nx))` in C order.

## Numerical Notes

- Reduce `dt` (e.g., `0.001`) when `chi` is large to avoid blow-up
- `sweep_count=5` nonlinear sweeps per step handles the nonlinear coupling between `rho` and `c`
- `s_boundary` accepts either a float (uniform on all edges) or a dict with keys `"left"`, `"right"`, `"top"`, `"bottom"` for spatially heterogeneous nutrient supply
- `s_no_flux_bc=True` overrides Dirichlet with zero-Neumann BCs for `s` (aggregation/starvation phase); `s_initial` carries over the field from a prior phase
- Aggregation instability requires `αχρ/(D_ρβ) >> 1`; if cells just diffuse, this ratio is too low
- Volume-filling prevents blow-up but max density will asymptote cleanly to `rho_max`
