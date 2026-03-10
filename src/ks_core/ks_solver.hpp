#pragma once
#include <Eigen/Sparse>
#include <functional>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// KSParams — mirrors KellerSegelParams (dimensionless code units)
// ---------------------------------------------------------------------------
struct KSParams {
    // Domain geometry
    int    nx, ny;
    double dx, dy;   // = Lx/nx, Ly/ny
    double Lx, Ly;

    // Cell density (ρ) PDE
    double D_rho;    // random motility / D_c
    double chi;      // chemotactic sensitivity × c_ref / D_c
    double mu_max;   // max specific growth rate / β
    double Y;        // normalised Monod half-saturation  K_s / s_boundary
    double rho_max;  // always 1.0 in dimensionless form

    // cAMP (c) PDE
    double D_c;      // always 1.0 in dimensionless form
    double beta;     // always 1.0 in dimensionless form
    double alpha;    // always 1.0 in dimensionless form

    // Nutrient (s) PDE
    double D_s;
    // Dirichlet BCs for s — keys: "left","right","top","bottom"
    std::map<std::string, double> s_boundary;

    // Initial conditions
    double rho_background;
    double rho_bump_amplitude;
    double rho_bump_sigma;
    // Optional custom IC (flat, column-major: idx = j*nx + i). Empty → use Gaussian bump.
    std::vector<double> rho_initial;

    // Time stepping
    double dt;
    int    n_steps;
    int    sweep_count;
    int    snapshot_interval;
};

// ---------------------------------------------------------------------------
// KSSolver — assembles and solves the three coupled FVM systems
// ---------------------------------------------------------------------------
class KSSolver {
public:
    // Snapshot output (flat arrays, Fortran/column-major: idx = j*nx + i)
    struct SnapshotData {
        std::vector<double>              times;
        std::vector<std::vector<double>> rho_snaps;
        std::vector<std::vector<double>> c_snaps;
        std::vector<std::vector<double>> s_snaps;
        std::vector<double>              total_mass;
        std::vector<double>              max_density;
    };

    // Progress callback: called with (current_step, total_steps) after each step.
    using ProgressCB = std::function<void(int, int)>;

    explicit KSSolver(const KSParams& p);
    SnapshotData run(ProgressCB progress = nullptr);

private:
    using SpMat  = Eigen::SparseMatrix<double>;
    using VecXd  = Eigen::VectorXd;
    using Triplet = Eigen::Triplet<double>;

    KSParams p_;
    int      N_;   // nx * ny

    // Field arrays (col-major: idx(i,j) = j*nx + i)
    VecXd rho_, rho_old_, c_, s_;

    // ---- matrices --------------------------------------------------------
    // A_c_: D_c·L − β·I  (structure never changes → factorise once)
    SpMat A_c_;
    Eigen::SparseLU<SpMat> lu_c_;

    // A_s_: D_s·L − μ_max·diag(ρ) + Dirichlet modifications
    // Sparsity pattern fixed after construction; numeric values change with ρ
    SpMat                  A_s_pattern_;   // pattern only (analysed once)
    Eigen::SparseLU<SpMat> lu_s_;
    bool                   s_pattern_analysed_ = false;

    // ---- helpers ---------------------------------------------------------
    inline int    idx(int i, int j) const noexcept { return j * p_.nx + i; }
    inline double xc(int i)         const noexcept { return (i + 0.5) * p_.dx; }
    inline double yc(int j)         const noexcept { return (j + 0.5) * p_.dy; }

    // ---- initialisation --------------------------------------------------
    void init_fields();

    // ---- matrix assembly -------------------------------------------------
    void   assemble_c_matrix();
    SpMat  assemble_s_matrix(VecXd& rhs_s) const;
    SpMat  assemble_rho_matrix(VecXd& rhs_rho) const;

    // ---- per-step solvers ------------------------------------------------
    void solve_c();
    void solve_s();
    void sweep_rho();   // one nonlinear sweep

    // ---- diagnostics -----------------------------------------------------
    double compute_total_mass()  const;
    double compute_max_density() const;
    std::vector<double> snapshot_flat(const VecXd& v) const;
};
