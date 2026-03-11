"""
Three-Variable Keller-Segel Slime Mold Chemotaxis Simulation

Implements a three-variable Keller-Segel model for Dictyostelium chemotaxis with
separate nutrient and self-produced chemoattractant, solved with FVM via FiPy on a
2D rectangular domain.

Three coupled PDEs (dimensionless form used by the solver):
  ∂ρ/∂t = D_ρ ∇²ρ - χ(1-ρ/ρ_max)∇·(ρ ∇c) + (μ_max/Y)·s·ρ·(1 - ρ/ρ_max)  (cell density)
  0      = D_c ∇²c - βc + αρ                                       (cAMP, quasi-SS)
  ∂s/∂t  = D_s ∇²s - μ_max·s·ρ                                     (nutrient, transient)

Cells chemotax up the cAMP gradient (c), which they produce themselves (rate α),
creating a positive feedback loop for aggregation. Growth is fueled by nutrient (s),
supplied at boundaries (Dirichlet BCs) and consumed by cells (Monod kinetics).

Non-dimensionalization (used by DimensionalParams.to_dimensionless):
  All three PDEs share the same reference scales:
    L₀ = √(D_c / β)          cAMP spatial decay length [mm]
    T₀ = 1 / β               cAMP degradation timescale [min]
    c₀ = c_ref               equilibrium [cAMP] at ρ_max [nM]
    α  = c_ref · β / ρ_max   derived (not a free parameter)

  Dimensionless solver parameters derived from physical ones:
    D_rho  = D_ρ / D_c                   (ratio of diffusivities)
    chi    = χ · c_ref / D_c             (chemotactic Péclet parameter)
    D_c    = 1.0   (fixed by scale choice)
    beta   = 1.0   (fixed by scale choice)
    alpha  = 1.0   (fixed by scale choice)
    D_s    = D_s_dim / D_c               (nutrient/cAMP diffusivity ratio)
    mu_max = μ_max · T₀ = μ_max / β     (growth-to-degradation ratio)
    Y      = K_s / s_boundary            (normalised Monod half-saturation)
    rho_max = 1.0  (fixed by scale choice)
    s_boundary = 1.0  (fixed by scale choice)
    Lx/Ly  = L_domain / L₀              (domain in units of L₀)
    dt     = dt_min · β                  (timestep in units of T₀)

Use DimensionalParams for a biologically interpretable interface:
  p = DimensionalParams.aggregation_phase()
  result = run_simulation(p.to_dimensionless())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Try the fast C++ backend first; fall back to FiPy if unavailable
# ---------------------------------------------------------------------------
try:
    import _ks_core  # type: ignore[import-not-found]
    _HAS_CPP = True
except ImportError:
    _ks_core = None  # type: ignore[assignment]
    _HAS_CPP = False

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
    D_s: float = 0.001        # nutrient diffusion (mm²/min), slow (bacteria)
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
    # Optional custom IC: 2D array (ny, nx) overriding Gaussian bump when set
    rho_initial: object = None  # np.ndarray or None

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def n_steps(self) -> int:
        return int(self.total_time / self.dt)


@dataclass
class DimensionalParams:
    """Biologically calibrated parameters in physical units (mm, min, nM, µg/mL).

    Use .to_dimensionless() to convert to KellerSegelParams for the solver.

    Two convenience constructors capture the two biological phases:
      DimensionalParams.growth_phase()       — cells dividing on nutrient, weak chemotaxis
      DimensionalParams.aggregation_phase()  — starved cells, strong cAMP-driven aggregation

    Literature sources
    ------------------
    D_c  : Tomchik & Devreotes 1981 (Science 212:443)
    beta : Martiel & Goldbeter 1987 (Biophys J 52:807)
    D_rho: Devreotes & Zigmond 1988 (Annu Rev Cell Biol 4:649)
    chi  : Van Haastert & Veltman 2007 (Nat Rev Mol Cell Biol 8:353)
    mu_max, rho_max: standard AX2 culture references
    """

    # --- cAMP — sets all reference scales (L₀, T₀, c₀) ---
    D_c_mm2_per_min: float = 0.024       # 400 µm²/s diffusion in aqueous/agar
    beta_per_min: float = 0.5            # phosphodiesterase degradation rate
    c_ref_nM: float = 10.0               # target equilibrium [cAMP] at ρ_max

    # --- Cell density ---
    D_rho_mm2_per_min: float = 5e-5      # random motility (50 µm²/min)
    chi_mm2_per_nM_per_min: float = 1.8e-3   # chemotactic sensitivity (aggregation)
    rho_max_cells_per_mm2: float = 1e4   # close-packed monolayer (~10 µm cells)
    rho_background_cells_per_mm2: float = 1e3   # uniform background (10% of max)
    rho_bump_amplitude_cells_per_mm2: float = 2e3   # Gaussian IC perturbation
    rho_bump_sigma_mm: float = 0.2       # ~1 aggregation territory radius
    # Optional custom IC: 2D array (ny, nx) in cells/mm². Overrides Gaussian bump.
    rho_initial_cells_per_mm2: object = None  # np.ndarray or None

    # --- Nutrient ---
    D_s_mm2_per_min: float = 0.001       # ~17 µm²/s slow bacterial diffusion
    K_s_ug_per_mL: float = 2.0           # Monod half-saturation constant (= s_boundary)
    s_boundary_ug_per_mL: float = 2.0    # Dirichlet boundary nutrient concentration
    mu_max_per_min: float = 0.002        # max specific growth rate (~5.8 hr doubling)

    # --- Physical domain ---
    Lx_mm: float = 5.0
    Ly_mm: float = 5.0
    nx: int = 100
    ny: int = 100

    # --- Simulation time ---
    total_time_min: float = 300.0        # 5 hours physical time
    dt_min: float = 0.04                 # timestep in minutes (~2.4 s)

    # --- Numerical (passed through unchanged) ---
    sweep_count: int = 5
    snapshot_interval: int = 50

    # --- Boundary conditions for s (passed through to KellerSegelParams) ---
    s_boundary_dict: Union[float, dict, None] = None  # None → use s_boundary_ug_per_mL uniformly

    @property
    def L0_mm(self) -> float:
        """cAMP spatial decay length [mm]: L₀ = √(D_c / β)."""
        return np.sqrt(self.D_c_mm2_per_min / self.beta_per_min)

    @property
    def T0_min(self) -> float:
        """cAMP degradation timescale [min]: T₀ = 1 / β."""
        return 1.0 / self.beta_per_min

    @property
    def alpha_nM_mm2_per_cell_per_min(self) -> float:
        """Derived cAMP production rate per cell: α = c_ref · β / ρ_max.

        Chosen so that the quasi-steady-state cAMP at uniform ρ_max equals c_ref_nM.
        Not a free parameter — adjust c_ref_nM to change cAMP levels.
        """
        return self.c_ref_nM * self.beta_per_min / self.rho_max_cells_per_mm2

    def to_dimensionless(self) -> KellerSegelParams:
        """Convert to dimensionless KellerSegelParams for the FiPy solver.

        Applies the non-dimensionalization:
          length  → x / L₀,   L₀ = √(D_c / β)
          time    → t · β
          density → ρ / ρ_max
          cAMP    → c / c_ref
          nutrient→ s / s_boundary
        """
        D_c = self.D_c_mm2_per_min
        beta = self.beta_per_min
        L0 = self.L0_mm
        rho0 = self.rho_max_cells_per_mm2
        s0 = self.s_boundary_ug_per_mL

        # Dimensionless PDE coefficients
        D_rho_nd = self.D_rho_mm2_per_min / D_c
        chi_nd = self.chi_mm2_per_nM_per_min * self.c_ref_nM / D_c
        D_s_nd = self.D_s_mm2_per_min / D_c
        mu_max_nd = self.mu_max_per_min / beta          # = mu_max * T₀
        # Y = K_s / s_boundary (normalised half-saturation).
        # When s_boundary = 0 (no nutrient), growth is absent so Y is irrelevant;
        # fall back to 1.0 to avoid division by zero.
        Y_nd = self.K_s_ug_per_mL / s0 if s0 > 0.0 else 1.0

        # Domain and time in code units
        Lx_nd = self.Lx_mm / L0
        Ly_nd = self.Ly_mm / L0
        total_time_nd = self.total_time_min * beta      # = total_time / T₀
        dt_nd = self.dt_min * beta

        # Initial conditions (normalised)
        rho_bg_nd = self.rho_background_cells_per_mm2 / rho0
        rho_bump_nd = self.rho_bump_amplitude_cells_per_mm2 / rho0
        rho_sigma_nd = self.rho_bump_sigma_mm / L0
        rho_init_nd = None
        if self.rho_initial_cells_per_mm2 is not None:
            rho_init_nd = np.asarray(self.rho_initial_cells_per_mm2) / rho0

        # Boundary condition for s: normalise scalar values; pass dict through.
        # When s0=0 (no nutrient), boundary is 0 regardless of s_boundary_dict.
        if s0 == 0.0:
            s_bc = 0.0
        elif self.s_boundary_dict is not None:
            if isinstance(self.s_boundary_dict, dict):
                s_bc = {k: v / s0 for k, v in self.s_boundary_dict.items()}
            else:
                s_bc = float(self.s_boundary_dict) / s0
        else:
            s_bc = 1.0  # normalised boundary = 1 by construction

        return KellerSegelParams(
            Lx=Lx_nd,
            Ly=Ly_nd,
            nx=self.nx,
            ny=self.ny,
            D_rho=D_rho_nd,
            chi=chi_nd,
            mu_max=mu_max_nd,
            Y=Y_nd,
            rho_max=1.0,           # fixed by ρ₀ = ρ_max
            D_c=1.0,               # fixed by L₀ = √(D_c/β)
            beta=1.0,              # fixed by T₀ = 1/β
            alpha=1.0,             # fixed by c₀ = α·ρ_max/β
            D_s=D_s_nd,
            s_boundary=s_bc,
            dt=dt_nd,
            total_time=total_time_nd,
            sweep_count=self.sweep_count,
            snapshot_interval=self.snapshot_interval,
            rho_background=rho_bg_nd,
            rho_bump_amplitude=rho_bump_nd,
            rho_bump_sigma=rho_sigma_nd,
            rho_initial=rho_init_nd,
        )

    @classmethod
    def growth_phase(cls) -> "DimensionalParams":
        """Preset for the vegetative growth phase.

        Cells are dividing on nutrient; chemotaxis is weak (10× lower chi than
        aggregation phase). Growth dominates dynamics. Domain kept smaller for
        faster iteration.
        """
        return cls(
            chi_mm2_per_nM_per_min=1.8e-4,   # 10× weaker than aggregation
            mu_max_per_min=0.002,
            total_time_min=600.0,             # 10 hours to see colony growth
        )

    @classmethod
    def aggregation_phase(cls) -> "DimensionalParams":
        """Preset for the starvation/aggregation phase.

        Cells are starved (nutrient depleted), cAMP relay is active, strong
        chemotaxis drives stream formation and aggregate nucleation. Growth is
        negligible (mu_max ≈ 0). Use small dt if chi is increased further.
        """
        return cls(
            chi_mm2_per_nM_per_min=1.8e-3,   # full aggregation-phase sensitivity
            mu_max_per_min=0.0,               # no growth during starvation
            s_boundary_ug_per_mL=0.0,         # no nutrient at boundaries
            total_time_min=120.0,             # 2 hours to see aggregation
        )


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

    # Initial condition for rho
    if params.rho_initial is not None:
        # Custom IC: (ny, nx) array. FiPy flat order = C-ravel of (ny, nx).
        rho.setValue(np.asarray(params.rho_initial).ravel())
    else:
        # Default: uniform background + Gaussian bump at center
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


def build_s_equation(
    s: CellVariable,
    rho: CellVariable,
    params: KellerSegelParams,
) -> object:
    """Build the transient nutrient equation.

    Solves: ∂s/∂t = D_s ∇²s - μ_max·s·ρ

    Nutrient diffuses from Dirichlet boundaries and is consumed by cells.
    Transient (not quasi-SS) because nutrient diffusion is slow relative to
    cell movement. Consumption is implicit in s for stability.
    """
    return (
        TransientTerm(var=s)
        == DiffusionTerm(coeff=params.D_s, var=s)
        + ImplicitSourceTerm(coeff=-params.mu_max * rho, var=s)
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


def _run_simulation_cpp(
    params: KellerSegelParams,
    progbar: bool = False,
) -> SimulationResult:
    """Call the C++ Eigen/pybind11 backend."""
    assert _ks_core is not None, "_ks_core extension not loaded"
    d = {
        "nx":               params.nx,
        "ny":               params.ny,
        "Lx":               params.Lx,
        "Ly":               params.Ly,
        "D_rho":            params.D_rho,
        "chi":              params.chi,
        "mu_max":           params.mu_max,
        "Y":                params.Y,
        "rho_max":          params.rho_max,
        "D_c":              params.D_c,
        "beta":             params.beta,
        "alpha":            params.alpha,
        "D_s":              params.D_s,
        "s_boundary":       params.s_boundary,
        "dt":               params.dt,
        "total_time":       params.total_time,
        "sweep_count":      params.sweep_count,
        "snapshot_interval": params.snapshot_interval,
        "rho_background":   params.rho_background,
        "rho_bump_amplitude": params.rho_bump_amplitude,
        "rho_bump_sigma":   params.rho_bump_sigma,
    }
    if params.rho_initial is not None:
        # Flat array in row-major C order matches idx = j*nx + i
        arr = np.asarray(params.rho_initial, dtype=np.float64)
        d["rho_initial"] = arr.ravel()

    progress_cb = None
    if progbar:
        from tqdm.auto import tqdm
        n_steps = int(params.total_time / params.dt)
        pbar = tqdm(total=n_steps, desc="C++ solver", unit="step")

        def progress_cb(step: int, total: int) -> None:
            pbar.update(1)
            if step == total:
                pbar.close()

    raw = _ks_core.run_simulation(d, progress_cb=progress_cb)
    return SimulationResult(
        times=list(raw["times"]),
        rho_snapshots=list(raw["rho_snapshots"]),
        c_snapshots=list(raw["c_snapshots"]),
        s_snapshots=list(raw["s_snapshots"]),
        total_mass=list(raw["total_mass"]),
        max_density=list(raw["max_density"]),
        params=params,
    )


def _run_simulation_fipy(
    params: KellerSegelParams,
    progbar: bool = False,
) -> SimulationResult:
    """FiPy fallback backend (original implementation)."""
    if progbar:
        from tqdm.auto import tqdm

    mesh, rho, c, s = create_mesh_and_variables(params)

    result = SimulationResult(params=params)

    # Solve initial quasi-steady-state c (s starts at 0, builds up transiently)
    eq_c = build_c_equation_steady(c, rho, params)
    eq_c.solve(var=c)

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

        # 2. Advance s transiently (nutrient: diffuses slowly, consumed by cells)
        s.updateOld()
        eq_s = build_s_equation(s, rho, params)
        eq_s.sweep(var=s, dt=params.dt)

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


def run_simulation(
    params: KellerSegelParams | None = None,
    progbar: bool = False,
) -> SimulationResult:
    """Run the full Keller-Segel simulation.

    Dispatches to the C++ Eigen backend (_ks_core) when available, falling
    back to the FiPy implementation otherwise.  Both backends support the
    ``progbar`` argument for tqdm progress reporting.
    """
    if params is None:
        params = KellerSegelParams()
    if _HAS_CPP:
        return _run_simulation_cpp(params, progbar=progbar)
    return _run_simulation_fipy(params, progbar=progbar)


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


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Keller-Segel Dictyostelium simulation")
    parser.add_argument(
        "--phase",
        choices=["growth", "aggregation", "custom"],
        default="aggregation",
        help="Biological phase preset (default: aggregation)",
    )
    args = parser.parse_args()

    if args.phase == "growth":
        dim = DimensionalParams.growth_phase()
    elif args.phase == "aggregation":
        dim = DimensionalParams.aggregation_phase()
    else:
        dim = DimensionalParams()

    params = dim.to_dimensionless()
    L0 = dim.L0_mm
    T0 = dim.T0_min

    print(f"Phase          : {args.phase}")
    print(f"L₀ = {L0:.3f} mm,  T₀ = {T0:.2f} min")
    print(f"Domain         : {dim.Lx_mm} × {dim.Ly_mm} mm  ({params.nx}×{params.ny} cells)")
    print(f"Physical time  : {dim.total_time_min:.0f} min  ({params.n_steps} steps)")
    print(f"D_rho (code)   : {params.D_rho:.5f}")
    print(f"chi   (code)   : {params.chi:.5f}  (chi/D_rho = {params.chi/params.D_rho:.1f})")
    print()

    result = run_simulation(params, progbar=True)

    # --- quick summary plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    final_rho = result.rho_snapshots[-1]
    final_c   = result.c_snapshots[-1]
    final_s   = result.s_snapshots[-1]

    extent = [0, dim.Lx_mm, 0, dim.Ly_mm]
    for ax, data, title, cmap in zip(
        axes,
        [final_rho, final_c, final_s],
        [f"ρ (cells/mm², ×{dim.rho_max_cells_per_mm2:.0f})",
         f"c (nM, ×{dim.c_ref_nM:.0f})",
         f"s (µg/mL, ×{dim.s_boundary_ug_per_mL:.1f})"],
        ["viridis", "plasma", "cividis"],
    ):
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    t_final_min = result.times[-1] * T0
    fig.suptitle(
        f"Keller-Segel ({args.phase} phase)  —  t = {t_final_min:.0f} min", y=1.01
    )
    plt.tight_layout()
    out = f"keller_segel_{args.phase}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out}")
