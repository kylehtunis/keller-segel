"""
Three-Variable Keller-Segel Slime Mold Chemotaxis Simulation

Implements a three-variable Keller-Segel model for Dictyostelium chemotaxis with
separate nutrient and self-produced chemoattractant, solved with FVM via FiPy on a
2D rectangular domain.

Three coupled PDEs:
  ∂ρ/∂t = D_ρ ∇²ρ - χ(1-ρ/ρ_max)∇·(ρ ∇c) + (μ_max/Y)·s·ρ·(1 - ρ/ρ_max)  (cell density)
  0 = D_c ∇²c - βc + αρ                                           (cAMP, quasi-SS)
  0 = D_s ∇²s - μ_max·s·ρ                                         (nutrient, quasi-SS)

Cells chemotax up the cAMP gradient (c), which they produce themselves (rate α),
creating a positive feedback loop for aggregation. Growth is fueled by nutrient (s),
supplied at boundaries (Dirichlet BCs) and consumed by cells (Monod kinetics).
"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from fipy import (
    CellVariable,
    DiffusionTerm,
    ExponentialConvectionTerm,
    Grid2D,
    ImplicitSourceTerm,
    TransientTerm,
)


@dataclass
class KellerSegelParams:
    """All model and numerical parameters in physical units (mm, min).

    Spatial resolution: dx = dy = 0.01 mm = 10 μm (one Dictyostelium cell length).
    Domain: 1 mm x 1 mm
    """

    # Domain geometry (mm)
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 100
    ny: int = 100

    # Cell density parameters
    D_rho: float = 5e-4       # cell random motility (mm²/min), ~50 μm²/min
    chi: float = 0.015        # chemotactic sensitivity (mm²/(min·conc))
    mu_max: float = 0.005     # max specific growth rate (min⁻¹), doubling ~2-3 hrs
    Y: float = 0.5            # yield coefficient (substrate per unit biomass)
    rho_max: float = 5.0      # carrying capacity (normalized density)

    # cAMP chemoattractant parameters (self-produced, no-flux BCs)
    D_c: float = 0.024        # cAMP diffusion (mm²/min), literature value
    beta: float = 0.5         # PDE degradation (min⁻¹), half-life ~1.4 min
    alpha: float = 1.0        # cAMP production rate by cells (conc/(min·density))

    # Nutrient/substrate parameters (external, Dirichlet BCs)
    D_s: float = 0.012        # nutrient diffusion (mm²/min), ~half D_c
    # Boundary conditions for s (Dirichlet)
    # Scalar: same value on all edges
    # Dict: per-edge, e.g. {"left": 1.0, "right": 0.0, "top": 0.5, "bottom": 0.5}
    s_boundary: Union[float, dict] = 1.0

    # Numerical parameters (min)
    dt: float = 0.1           # timestep (min)
    total_time: float = 480.0 # total simulation time (min), 8 hours
    sweep_count: int = 3      # nonlinear sweeps per timestep
    snapshot_interval: int = 48  # save every 48 steps (~4.8 min real time)

    # Initial conditions
    rho_background: float = 0.1
    rho_bump_amplitude: float = 1.0
    rho_bump_sigma: float = 0.1  # mm, ~100 μm = ~10 cell lengths

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def n_steps(self) -> int:
        return int(self.total_time / self.dt)


def create_mesh_and_variables(
    params: KellerSegelParams,
) -> tuple[Grid2D, CellVariable, CellVariable, CellVariable]:
    """Create the FiPy mesh and cell variables with initial/boundary conditions.

    Returns:
        (mesh, rho, c, s) where rho is cell density, c is cAMP chemoattractant,
        and s is nutrient/substrate concentration.
    """
    mesh = Grid2D(dx=params.dx, dy=params.dy, nx=params.nx, ny=params.ny)

    # Cell density: hasOld=True for TransientTerm sweeping
    rho = CellVariable(name="cell_density", mesh=mesh, value=0.0, hasOld=True)

    # cAMP chemoattractant: no-flux BCs (FiPy default), starts at 0
    c = CellVariable(name="cAMP", mesh=mesh, value=0.0, hasOld=True)

    # Nutrient/substrate: Dirichlet BCs (food sources at boundaries)
    s = CellVariable(name="nutrient", mesh=mesh, value=0.0, hasOld=True)

    # Initial condition for rho: uniform background + Gaussian bump at center
    x, y = mesh.cellCenters
    cx, cy = params.Lx / 2.0, params.Ly / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    rho.setValue(
        params.rho_background
        + params.rho_bump_amplitude * np.exp(-r2 / (2 * params.rho_bump_sigma**2))
    )

    # c starts at 0.0 — will build up from cell production (no-flux BCs)
    c.setValue(0.0)

    # s starts at 0.0 — will reach steady state from Dirichlet BCs
    s.setValue(0.0)
    _apply_dirichlet_bcs(s, mesh, params.s_boundary)

    # rho: no-flux (zero Neumann) is FiPy's default -- no constrain needed

    return mesh, rho, c, s


def _apply_dirichlet_bcs(
    var: CellVariable,
    mesh: Grid2D,
    boundary: Union[float, dict],
) -> None:
    """Apply Dirichlet BCs to a variable on each edge."""
    if isinstance(boundary, (int, float)):
        bc_dict = {
            "left": float(boundary),
            "right": float(boundary),
            "top": float(boundary),
            "bottom": float(boundary),
        }
    else:
        bc_dict = boundary

    face_map = {
        "left": mesh.facesLeft,
        "right": mesh.facesRight,
        "top": mesh.facesTop,
        "bottom": mesh.facesBottom,
    }

    for edge, value in bc_dict.items():
        var.constrain(value, where=face_map[edge])


def build_c_equation_steady(
    c: CellVariable,
    rho: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the quasi-steady-state cAMP equation.

    Solves: 0 = D_c ∇²c - βc + αρ

    cAMP is self-produced by cells (αρ) and degraded by phosphodiesterase (βc).
    No-flux BCs — signal stays in domain. The αρ term is explicit (depends on ρ).
    """
    return (
        DiffusionTerm(coeff=params.D_c, var=c)
        + ImplicitSourceTerm(coeff=-params.beta, var=c)
        + params.alpha * rho
    )


def build_s_equation_steady(
    s: CellVariable,
    rho: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the quasi-steady-state nutrient equation.

    Solves: 0 = D_s ∇²s - μ_max·s·ρ

    Nutrient diffuses from Dirichlet boundaries and is consumed by cells.
    Consumption is implicit in s (good for stability).
    """
    return DiffusionTerm(coeff=params.D_s, var=s) + ImplicitSourceTerm(
        coeff=-params.mu_max * rho, var=s
    )


def build_rho_equation(
    rho: CellVariable,
    c: CellVariable,
    s: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the cell density equation for one timestep.

    ∂ρ/∂t = D_ρ ∇²ρ - χ(1 - ρ/ρ_max)∇·(ρ ∇c) + (μ_max/Y)·s·ρ·(1 - ρ/ρ_max)

    - Volume-filling chemotaxis (Painter & Hillen 2002): sensitivity is multiplied
      by (1 - ρ/ρ_max) so chemotactic flux → 0 as density approaches carrying capacity.
      This prevents the finite-time blow-up of classical Keller-Segel.
    - Monod-logistic growth: (μ_max/Y)·s is the biomass growth rate from nutrient,
      split into implicit linear source and implicit quadratic sink for stability
    """
    growth_rate = (params.mu_max / params.Y) * s  # nutrient-driven growth
    # Volume-filling: chemotactic velocity scaled by available space
    volume_factor = 1.0 - rho.faceValue / params.rho_max
    chemotaxis_coeff = params.chi * volume_factor * c.faceGrad
    return (
        TransientTerm(var=rho)
        == DiffusionTerm(coeff=params.D_rho, var=rho)
        - ExponentialConvectionTerm(coeff=chemotaxis_coeff, var=rho)
        + ImplicitSourceTerm(coeff=growth_rate, var=rho)
        + ImplicitSourceTerm(coeff=-growth_rate * rho / params.rho_max, var=rho)
    )


@dataclass
class SimulationResult:
    """Container for simulation output data."""

    times: list[float] = field(default_factory=list)
    rho_snapshots: list[np.ndarray] = field(default_factory=list)
    c_snapshots: list[np.ndarray] = field(default_factory=list)
    s_snapshots: list[np.ndarray] = field(default_factory=list)
    total_mass: list[float] = field(default_factory=list)
    max_density: list[float] = field(default_factory=list)
    params: KellerSegelParams = field(default_factory=KellerSegelParams)


def run_simulation(
    params: KellerSegelParams | None = None,
    progbar: bool = False,
) -> SimulationResult:
    """Run the full Keller-Segel simulation with pseudo-steady-state timestepping.

    At each biological timestep:
      1. Solve c to steady state (single elliptic solve)
      2. Advance ρ one timestep using steady-state c gradient
      3. Clamp ρ >= 0
      4. Record diagnostics and snapshots
    """
    if params is None:
        params = KellerSegelParams()

    if progbar:
        from tqdm.auto import tqdm

    mesh, rho, c, s = create_mesh_and_variables(params)

    result = SimulationResult(params=params)

    # Solve initial quasi-steady-state c and s (given initial rho)
    eq_c = build_c_equation_steady(c, rho, params)
    eq_c.solve(var=c)
    eq_s = build_s_equation_steady(s, rho, params)
    eq_s.solve(var=s)

    # Record initial state
    _record_snapshot(result, 0.0, rho, c, s, mesh, params)

    steps = range(1, params.n_steps + 1)
    if progbar:
        steps = tqdm(steps, desc="Keller-Segel", unit="step")

    for step in steps:
        t = step * params.dt

        # 1. Solve c to quasi-steady-state (cAMP: self-produced by cells)
        eq_c = build_c_equation_steady(c, rho, params)
        eq_c.solve(var=c)

        # 2. Solve s to quasi-steady-state (nutrient: consumed by cells)
        eq_s = build_s_equation_steady(s, rho, params)
        eq_s.solve(var=s)

        # 3. Advance rho using c gradient (chemotaxis) and s (growth)
        rho.updateOld()
        for _sweep in range(params.sweep_count):
            eq_rho = build_rho_equation(rho, c, s, params)
            eq_rho.sweep(var=rho, dt=params.dt)

        # 4. Clamp rho >= 0 to prevent negative densities
        rho.setValue(np.maximum(rho.value, 0.0))

        # 5. Record diagnostics
        result.total_mass.append(compute_total_mass(rho, mesh))
        result.max_density.append(compute_max_density(rho))

        # 6. Save snapshot at intervals
        if step % params.snapshot_interval == 0:
            _record_snapshot(result, t, rho, c, s, mesh, params)

        if not progbar and step % (params.n_steps // 10 or 1) == 0:
            print(
                f"Step {step}/{params.n_steps} (t={t:.2f}): "
                f"mass={result.total_mass[-1]:.4f}, "
                f"max_rho={result.max_density[-1]:.4f}"
            )

    return result


def _record_snapshot(
    result: SimulationResult,
    t: float,
    rho: CellVariable,
    c: CellVariable,
    s: CellVariable,
    mesh: Grid2D,
    params: KellerSegelParams,
) -> None:
    """Save current state as a snapshot."""
    result.times.append(t)
    result.rho_snapshots.append(get_2d_array(rho, params.nx, params.ny))
    result.c_snapshots.append(get_2d_array(c, params.nx, params.ny))
    result.s_snapshots.append(get_2d_array(s, params.nx, params.ny))
    result.total_mass.append(compute_total_mass(rho, mesh))
    result.max_density.append(compute_max_density(rho))


def get_2d_array(var: CellVariable, nx: int, ny: int) -> np.ndarray:
    """Reshape flat CellVariable data to (ny, nx) array for imshow."""
    return np.array(var.value).reshape((nx, ny), order="F").T


def compute_total_mass(rho: CellVariable, mesh: Grid2D) -> float:
    """Compute total cell mass (integral of ρ over domain)."""
    return float((rho * mesh.cellVolumes).sum())


def compute_max_density(rho: CellVariable) -> float:
    """Compute maximum cell density (blow-up detection)."""
    return float(max(rho.value))
