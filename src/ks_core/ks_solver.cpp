#include "ks_solver.hpp"
#include "sg_scheme.hpp"

#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ===========================================================================
// Construction & initialisation
// ===========================================================================

KSSolver::KSSolver(const KSParams& p)
    : p_(p), N_(p.nx * p.ny),
      rho_(N_), rho_old_(N_), c_(N_), s_(N_)
{
    init_fields();
    assemble_c_matrix();
    // Analyse sparsity pattern of s matrix (numeric values assembled per-step)
    {
        VecXd dummy_rhs(N_);
        SpMat A_s = assemble_s_matrix(dummy_rhs);
        lu_s_.analyzePattern(A_s);
        s_pattern_analysed_ = true;
        A_s_pattern_ = A_s;  // keep for later re-use of pattern
    }
}

void KSSolver::init_fields() {
    const int nx = p_.nx, ny = p_.ny;
    const double cx = p_.Lx / 2.0;
    const double cy = p_.Ly / 2.0;
    const double sig2 = 2.0 * p_.rho_bump_sigma * p_.rho_bump_sigma;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = idx(i, j);
            double r2 = (xc(i) - cx) * (xc(i) - cx)
                      + (yc(j) - cy) * (yc(j) - cy);
            rho_[k] = p_.rho_background
                    + p_.rho_bump_amplitude * std::exp(-r2 / sig2);
        }
    }
    c_.setZero();
    s_.setZero();
    // Dirichlet BCs for s are enforced through the linear system, not pre-set
}

// ===========================================================================
// Matrix assembly — cAMP (c)
//
//   A_c · c = rhs_c
//   A_c = D_c·L_Neumann − β·I        (structure FIXED — factorise once)
//   rhs_c[k] = α·ρ[k]                (rebuilt each step)
//
// Zero-flux (Neumann) BCs: boundary faces simply absent from stencil.
// ===========================================================================

void KSSolver::assemble_c_matrix() {
    const int    nx = p_.nx, ny = p_.ny;
    const double dx2 = p_.dx * p_.dx;
    const double dy2 = p_.dy * p_.dy;
    const double Dc  = p_.D_c;
    const double b   = p_.beta;

    std::vector<Triplet> trips;
    trips.reserve(5 * N_);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = idx(i, j);
            double diag = b;   // diagonal accumulates from faces + β

            // West face
            if (i > 0) {
                trips.push_back({k, idx(i-1,j), -Dc/dx2});
                diag += Dc/dx2;
            }
            // East face
            if (i < nx-1) {
                trips.push_back({k, idx(i+1,j), -Dc/dx2});
                diag += Dc/dx2;
            }
            // South face
            if (j > 0) {
                trips.push_back({k, idx(i,j-1), -Dc/dy2});
                diag += Dc/dy2;
            }
            // North face
            if (j < ny-1) {
                trips.push_back({k, idx(i,j+1), -Dc/dy2});
                diag += Dc/dy2;
            }
            trips.push_back({k, k, diag});
        }
    }

    A_c_.resize(N_, N_);
    A_c_.setFromTriplets(trips.begin(), trips.end());
    lu_c_.compute(A_c_);
    if (lu_c_.info() != Eigen::Success)
        throw std::runtime_error("KSSolver: SparseLU factorisation of A_c failed");
}

// ===========================================================================
// Matrix assembly — nutrient (s)
//
//   A_s · s = rhs_s
//   A_s = D_s·L − μ_max·diag(ρ)   + Dirichlet modifications
//   rhs_s encodes the Dirichlet values
//
// Dirichlet ghost-cell on boundary face:
//   flux through face = D_s · (s_boundary − s_cell) / (h/2)
//               = (2·D_s/h²) · s_boundary  −  (2·D_s/h²) · s_cell
//   → diagonal += 2·D_s/h², rhs += 2·D_s/h² · s_val
// ===========================================================================

KSSolver::SpMat KSSolver::assemble_s_matrix(VecXd& rhs_s) const {
    const int    nx = p_.nx, ny = p_.ny;
    const double dx2 = p_.dx * p_.dx;
    const double dy2 = p_.dy * p_.dy;
    const double Ds  = p_.D_s;

    rhs_s.setZero(N_);
    std::vector<Triplet> trips;
    trips.reserve(5 * N_);

    auto bc = [&](const std::string& edge) -> double {
        auto it = p_.s_boundary.find(edge);
        return (it != p_.s_boundary.end()) ? it->second : 0.0;
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int    k    = idx(i, j);
            double diag = p_.mu_max * rho_[k];  // sink term

            // West
            if (i == 0) {
                double sv = bc("left");
                diag           += 2.0*Ds/dx2;
                rhs_s[k]       += 2.0*Ds/dx2 * sv;
            } else {
                trips.push_back({k, idx(i-1,j), -Ds/dx2});
                diag += Ds/dx2;
            }
            // East
            if (i == nx-1) {
                double sv = bc("right");
                diag           += 2.0*Ds/dx2;
                rhs_s[k]       += 2.0*Ds/dx2 * sv;
            } else {
                trips.push_back({k, idx(i+1,j), -Ds/dx2});
                diag += Ds/dx2;
            }
            // South
            if (j == 0) {
                double sv = bc("bottom");
                diag           += 2.0*Ds/dy2;
                rhs_s[k]       += 2.0*Ds/dy2 * sv;
            } else {
                trips.push_back({k, idx(i,j-1), -Ds/dy2});
                diag += Ds/dy2;
            }
            // North
            if (j == ny-1) {
                double sv = bc("top");
                diag           += 2.0*Ds/dy2;
                rhs_s[k]       += 2.0*Ds/dy2 * sv;
            } else {
                trips.push_back({k, idx(i,j+1), -Ds/dy2});
                diag += Ds/dy2;
            }
            trips.push_back({k, k, diag});
        }
    }

    SpMat A(N_, N_);
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// ===========================================================================
// Matrix assembly — cell density (ρ) — one sweep
//
//   A_rho · ρ^{n+1} = rhs_rho
//
//   Transient:   A[k,k] += 1/dt;   rhs[k] = ρ_old[k]/dt
//   Diffusion:   standard 5-pt, no-flux BCs
//   SG chemotaxis (x-face between k=(i,j) and kr=(i+1,j)):
//     Pe = χ·(c[kr]−c[k])/D_ρ
//     A[k, k]   += (D_ρ/dx²)·B( Pe)    outgoing from k
//     A[k, kr]  -= (D_ρ/dx²)·B(−Pe)    incoming from kr
//     A[kr, kr] += (D_ρ/dx²)·B(−Pe)    outgoing from kr
//     A[kr, k]  -= (D_ρ/dx²)·B( Pe)    incoming from k
//   Logistic growth (linearised around ρ_old):
//     g = (μ_max/Y)·s[k]
//     A[k,k] -= g·(1 − ρ_old[k]/ρ_max)
// ===========================================================================

KSSolver::SpMat KSSolver::assemble_rho_matrix(VecXd& rhs_rho) const {
    const int    nx   = p_.nx, ny = p_.ny;
    const double dx2  = p_.dx * p_.dx;
    const double dy2  = p_.dy * p_.dy;
    const double Drho    = p_.D_rho;
    const double chi     = p_.chi;
    const double rho_max = p_.rho_max;
    const double dt      = p_.dt;

    rhs_rho = rho_old_ / dt;

    // Accumulate diagonal separately to avoid repeated coeffRef
    VecXd diag = VecXd::Constant(N_, 1.0 / dt);
    std::vector<Triplet> trips;
    trips.reserve(5 * N_);

    // ----- x-direction faces -----
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx-1; ++i) {
            int k  = idx(i,   j);
            int kr = idx(i+1, j);

            // Pure diffusion contribution
            const double Ddx2 = Drho / dx2;
            diag[k]  += Ddx2;
            diag[kr] += Ddx2;
            trips.push_back({k,  kr, -Ddx2});
            trips.push_back({kr, k,  -Ddx2});

            // Volume-filling: scale chi by (1 - rho_face/rho_max)
            // Use average of the two cell values at the face
            const double vf_x = 1.0 - 0.5*(rho_old_[k] + rho_old_[kr]) / rho_max;
            const double chi_eff_x = chi * std::max(vf_x, 0.0);

            // SG correction (subtract diffusion, add SG)
            const double Pe   = chi_eff_x * (c_[kr] - c_[k]) / Drho;
            const double BPe  = bernoulli( Pe);
            const double BnPe = bernoulli(-Pe);

            // Net SG diagonal increment over pure diffusion
            diag[k]  += Ddx2 * (BPe  - 1.0);
            diag[kr] += Ddx2 * (BnPe - 1.0);

            // SG off-diagonal corrections
            trips.push_back({k,  kr,  Ddx2 * (1.0 - BnPe)});
            trips.push_back({kr, k,   Ddx2 * (1.0 - BPe)});
        }
    }

    // ----- y-direction faces -----
    for (int j = 0; j < ny-1; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k  = idx(i, j);
            int kt = idx(i, j+1);

            const double Ddy2 = Drho / dy2;
            diag[k]  += Ddy2;
            diag[kt] += Ddy2;
            trips.push_back({k,  kt, -Ddy2});
            trips.push_back({kt, k,  -Ddy2});

            // Volume-filling: scale chi by (1 - rho_face/rho_max)
            const double vf_y = 1.0 - 0.5*(rho_old_[k] + rho_old_[kt]) / rho_max;
            const double chi_eff_y = chi * std::max(vf_y, 0.0);

            const double Pe   = chi_eff_y * (c_[kt] - c_[k]) / Drho;
            const double BPe  = bernoulli( Pe);
            const double BnPe = bernoulli(-Pe);

            diag[k]  += Ddy2 * (BPe  - 1.0);
            diag[kt] += Ddy2 * (BnPe - 1.0);

            trips.push_back({k,  kt,  Ddy2 * (1.0 - BnPe)});
            trips.push_back({kt, k,   Ddy2 * (1.0 - BPe)});
        }
    }

    // ----- logistic growth (diagonal only) -----
    for (int k = 0; k < N_; ++k) {
        const double g = (p_.mu_max / p_.Y) * s_[k];
        diag[k] -= g * (1.0 - rho_old_[k] / p_.rho_max);
    }

    // Merge diagonal into triplet list
    for (int k = 0; k < N_; ++k)
        trips.push_back({k, k, diag[k]});

    SpMat A(N_, N_);
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// ===========================================================================
// Per-step solvers
// ===========================================================================

void KSSolver::solve_c() {
    // A_c_ is constant; only rebuild RHS
    VecXd rhs(N_);
    for (int k = 0; k < N_; ++k)
        rhs[k] = p_.alpha * rho_[k];
    c_ = lu_c_.solve(rhs);
}

void KSSolver::solve_s() {
    VecXd rhs_s;
    SpMat A_s = assemble_s_matrix(rhs_s);
    lu_s_.factorize(A_s);
    if (lu_s_.info() != Eigen::Success)
        throw std::runtime_error("KSSolver: SparseLU factorisation of A_s failed");
    s_ = lu_s_.solve(rhs_s);
}

void KSSolver::sweep_rho() {
    VecXd rhs_rho;
    SpMat A_rho = assemble_rho_matrix(rhs_rho);

    Eigen::SparseLU<SpMat> lu_rho;
    lu_rho.compute(A_rho);
    if (lu_rho.info() != Eigen::Success)
        throw std::runtime_error("KSSolver: SparseLU factorisation of A_rho failed");
    rho_ = lu_rho.solve(rhs_rho);
}

// ===========================================================================
// Diagnostics
// ===========================================================================

double KSSolver::compute_total_mass() const {
    // ∫ρ dA ≈ Σ ρ_k · dx · dy
    const double cell_vol = p_.dx * p_.dy;
    return rho_.sum() * cell_vol;
}

double KSSolver::compute_max_density() const {
    return rho_.maxCoeff();
}

std::vector<double> KSSolver::snapshot_flat(const VecXd& v) const {
    return std::vector<double>(v.data(), v.data() + N_);
}

// ===========================================================================
// Main time loop
// ===========================================================================

KSSolver::SnapshotData KSSolver::run() {
    SnapshotData out;

    // Initial quasi-steady-state solve
    solve_c();
    solve_s();

    // Record initial state (t = 0)
    out.times.push_back(0.0);
    out.rho_snaps.push_back(snapshot_flat(rho_));
    out.c_snaps.push_back(snapshot_flat(c_));
    out.s_snaps.push_back(snapshot_flat(s_));
    out.total_mass.push_back(compute_total_mass());
    out.max_density.push_back(compute_max_density());

    for (int step = 1; step <= p_.n_steps; ++step) {
        const double t = step * p_.dt;

        // 1. cAMP quasi-steady-state
        solve_c();

        // 2. Nutrient quasi-steady-state
        solve_s();

        // 3. Advance ρ with nonlinear sweeps
        rho_old_ = rho_;
        for (int sw = 0; sw < p_.sweep_count; ++sw)
            sweep_rho();

        // 4. Clamp ρ ≥ 0
        rho_ = rho_.cwiseMax(0.0);

        // 5. Diagnostics at every step
        out.total_mass.push_back(compute_total_mass());
        out.max_density.push_back(compute_max_density());

        // 6. Full snapshot at intervals
        if (step % p_.snapshot_interval == 0) {
            out.times.push_back(t);
            out.rho_snaps.push_back(snapshot_flat(rho_));
            out.c_snaps.push_back(snapshot_flat(c_));
            out.s_snaps.push_back(snapshot_flat(s_));
        }
    }

    return out;
}
