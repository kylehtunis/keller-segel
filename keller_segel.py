"""
Keller-Segel Slime Mold Chemotaxis Simulation

Implements the Keller-Segel model for Dictyostelium chemotaxis toward an
external chemical attractant, solved with FVM via FiPy on a 2D rectangular domain.

Two coupled PDEs:
  ∂ρ/∂t = D_ρ ∇²ρ - χ ∇·(ρ ∇c) + (μ_max/Y)·c·ρ·(1 - ρ/ρ_max)  (cell density)
  ∂c/∂t = D_c ∇²c - βc - μ_max·c·ρ                              (chemoattractant)

Growth follows Monod kinetics: cells consume substrate at rate μ_max·c·ρ. The yield
coefficient Y (substrate consumed per unit biomass) appears in the denominator of the
growth term — lower Y means more efficient conversion of substrate to biomass.
c is re-solved via quasi-steady-state at each timestep.
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
    """All model and numerical parameters with biologically-motivated defaults."""

    # Domain geometry
    Lx: float = 10.0
    Ly: float = 10.0
    nx: int = 100
    ny: int = 100

    # Cell density parameters
    D_rho: float = 1e-3       # cell diffusion coefficient
    chi: float = 5e-3         # chemotactic sensitivity
    mu_max: float = 0.02      # maximum specific growth rate (Monod)
    Y: float = 0.5            # yield coefficient (substrate consumed per unit biomass)
    rho_max: float = 5.0      # carrying capacity

    # Chemoattractant parameters
    D_c: float = 1e-2         # chemical diffusion coefficient
    beta: float = 0.05        # chemical decay rate

    # Boundary conditions for c (Dirichlet)
    # Scalar: same value on all edges
    # Dict: per-edge, e.g. {"left": 1.0, "right": 0.0, "top": 0.5, "bottom": 0.5}
    c_boundary: Union[float, dict] = 1.0

    # Numerical parameters
    dt: float = 0.01          # timestep for cell dynamics
    total_time: float = 50.0  # total simulation time
    sweep_count: int = 5      # nonlinear sweeps per timestep
    snapshot_interval: int = 50  # save snapshot every N steps

    # Initial conditions
    rho_background: float = 0.1
    rho_bump_amplitude: float = 1.0
    rho_bump_sigma: float = 1.0

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
) -> tuple[Grid2D, CellVariable, CellVariable]:
    """Create the FiPy mesh and cell variables with initial/boundary conditions.

    Returns:
        (mesh, rho, c) where rho is cell density and c is chemoattractant concentration.
    """
    mesh = Grid2D(dx=params.dx, dy=params.dy, nx=params.nx, ny=params.ny)

    # Cell density: hasOld=True for TransientTerm sweeping
    rho = CellVariable(name="cell_density", mesh=mesh, value=0.0, hasOld=True)
    c = CellVariable(name="chemoattractant", mesh=mesh, value=0.0, hasOld=True)

    # Initial condition for rho: uniform background + Gaussian bump at center
    x, y = mesh.cellCenters
    cx, cy = params.Lx / 2.0, params.Ly / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    rho.setValue(
        params.rho_background
        + params.rho_bump_amplitude * np.exp(-r2 / (2 * params.rho_bump_sigma**2))
    )

    # Initial condition for c: 0.0 (will reach steady state from BCs)
    c.setValue(0.0)

    # Boundary conditions for c: Dirichlet on all edges
    _apply_c_boundary_conditions(c, mesh, params.c_boundary)

    # rho: no-flux (zero Neumann) is FiPy's default -- no constrain needed

    return mesh, rho, c


def _apply_c_boundary_conditions(
    c: CellVariable,
    mesh: Grid2D,
    c_boundary: Union[float, dict],
) -> None:
    """Apply Dirichlet BCs to chemoattractant on each edge."""
    if isinstance(c_boundary, (int, float)):
        bc_dict = {
            "left": float(c_boundary),
            "right": float(c_boundary),
            "top": float(c_boundary),
            "bottom": float(c_boundary),
        }
    else:
        bc_dict = c_boundary

    face_map = {
        "left": mesh.facesLeft,
        "right": mesh.facesRight,
        "top": mesh.facesTop,
        "bottom": mesh.facesBottom,
    }

    for edge, value in bc_dict.items():
        c.constrain(value, where=face_map[edge])


def build_c_equation_steady(
    c: CellVariable,
    rho: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the quasi-steady-state chemoattractant equation.

    Solves: 0 = D_c ∇²c - βc - (μ_max/Y)·c·ρ

    The decay and consumption terms are both proportional to c, so they
    combine into a single ImplicitSourceTerm with coefficient -(β + μ_max · ρ).
    """
    return DiffusionTerm(coeff=params.D_c, var=c) + ImplicitSourceTerm(
        coeff=-(params.beta + params.mu_max * rho), var=c
    )


def build_rho_equation(
    rho: CellVariable,
    c: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the cell density equation for one timestep.

    ∂ρ/∂t = D_ρ ∇²ρ - χ ∇·(ρ ∇c) + μ_max·c·ρ·(1 - ρ/ρ_max)

    - Chemotaxis: ExponentialConvectionTerm with velocity χ∇c (Scharfetter-Gummel)
    - Monod-logistic growth: (μ_max/Y)·c is the biomass growth rate,
      split into implicit linear source ((μ_max/Y)·c·ρ) and implicit quadratic
      sink (-(μ_max/Y)·c·ρ²/ρ_max) for numerical stability
    """
    growth_rate = (params.mu_max / params.Y) * c  # substrate consumption rate / yield
    return (
        TransientTerm(var=rho)
        == DiffusionTerm(coeff=params.D_rho, var=rho)
        - ExponentialConvectionTerm(coeff=params.chi * c.faceGrad, var=rho)
        + ImplicitSourceTerm(coeff=growth_rate, var=rho)
        + ImplicitSourceTerm(coeff=-growth_rate * rho / params.rho_max, var=rho)
    )


@dataclass
class SimulationResult:
    """Container for simulation output data."""

    times: list[float] = field(default_factory=list)
    rho_snapshots: list[np.ndarray] = field(default_factory=list)
    c_snapshots: list[np.ndarray] = field(default_factory=list)
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

    mesh, rho, c = create_mesh_and_variables(params)

    result = SimulationResult(params=params)

    # Solve initial quasi-steady-state c (given initial rho)
    eq_c = build_c_equation_steady(c, rho, params)
    eq_c.solve(var=c)

    # Record initial state
    _record_snapshot(result, 0.0, rho, c, mesh, params)

    steps = range(1, params.n_steps + 1)
    if progbar:
        steps = tqdm(steps, desc="Keller-Segel", unit="step")

    for step in steps:
        t = step * params.dt

        # 1. Solve c to quasi-steady-state given current rho
        #    (must rebuild equation since consumption depends on rho)
        eq_c = build_c_equation_steady(c, rho, params)
        eq_c.solve(var=c)

        # 2. Advance rho using updated c gradient
        rho.updateOld()
        for _sweep in range(params.sweep_count):
            eq_rho = build_rho_equation(rho, c, params)
            eq_rho.sweep(var=rho, dt=params.dt)

        # 3. Clamp rho >= 0 to prevent negative densities
        rho.setValue(np.maximum(rho.value, 0.0))

        # 4. Record diagnostics
        result.total_mass.append(compute_total_mass(rho, mesh))
        result.max_density.append(compute_max_density(rho))

        # 5. Save snapshot at intervals
        if step % params.snapshot_interval == 0:
            _record_snapshot(result, t, rho, c, mesh, params)

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
    mesh: Grid2D,
    params: KellerSegelParams,
) -> None:
    """Save current state as a snapshot."""
    result.times.append(t)
    result.rho_snapshots.append(get_2d_array(rho, params.nx, params.ny))
    result.c_snapshots.append(get_2d_array(c, params.nx, params.ny))
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
