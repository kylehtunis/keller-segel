#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ks_solver.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Build KSParams from the Python dict passed by _run_simulation_cpp()
// ---------------------------------------------------------------------------
static KSParams dict_to_params(const py::dict& d) {
    KSParams p;

    // Domain
    p.nx = d["nx"].cast<int>();
    p.ny = d["ny"].cast<int>();
    p.Lx = d["Lx"].cast<double>();
    p.Ly = d["Ly"].cast<double>();
    p.dx = p.Lx / p.nx;
    p.dy = p.Ly / p.ny;

    // Cell density
    p.D_rho   = d["D_rho"].cast<double>();
    p.chi     = d["chi"].cast<double>();
    p.mu_max  = d["mu_max"].cast<double>();
    p.Y       = d["Y"].cast<double>();
    p.rho_max = d["rho_max"].cast<double>();

    // cAMP
    p.D_c   = d["D_c"].cast<double>();
    p.beta  = d["beta"].cast<double>();
    p.alpha = d["alpha"].cast<double>();

    // Nutrient
    p.D_s = d["D_s"].cast<double>();

    // s_boundary: float/int → expand to all four edges; dict → pass through
    py::object sb = d["s_boundary"];
    if (py::isinstance<py::float_>(sb) || py::isinstance<py::int_>(sb)) {
        double v = sb.cast<double>();
        p.s_boundary = {{"left",v}, {"right",v}, {"top",v}, {"bottom",v}};
    } else {
        py::dict sbd = sb.cast<py::dict>();
        for (auto kv : sbd)
            p.s_boundary[kv.first.cast<std::string>()] = kv.second.cast<double>();
    }

    // Time stepping
    p.dt               = d["dt"].cast<double>();
    p.n_steps          = static_cast<int>(
                            d["total_time"].cast<double>() / p.dt);
    p.sweep_count      = d["sweep_count"].cast<int>();
    p.snapshot_interval = d["snapshot_interval"].cast<int>();

    // Initial conditions
    p.rho_background     = d["rho_background"].cast<double>();
    p.rho_bump_amplitude = d["rho_bump_amplitude"].cast<double>();
    p.rho_bump_sigma     = d["rho_bump_sigma"].cast<double>();

    // Optional custom IC (flat numpy array, column-major)
    if (d.contains("rho_initial")) {
        auto arr = d["rho_initial"].cast<py::array_t<double>>();
        auto buf = arr.unchecked<1>();
        p.rho_initial.resize(buf.shape(0));
        for (py::ssize_t k = 0; k < buf.shape(0); ++k)
            p.rho_initial[k] = buf(k);
    }

    // Optional s initial condition
    if (d.contains("s_initial")) {
        auto arr = d["s_initial"].cast<py::array_t<double>>();
        auto buf = arr.unchecked<1>();
        p.s_initial.resize(buf.shape(0));
        for (py::ssize_t k = 0; k < buf.shape(0); ++k)
            p.s_initial[k] = buf(k);
    }

    // No-flux BC flag for s
    if (d.contains("s_no_flux_bc"))
        p.s_no_flux_bc = d["s_no_flux_bc"].cast<bool>();

    return p;
}

// ---------------------------------------------------------------------------
// Convert a flat C++ vector (Fortran/col-major: idx = j*nx + i) to a
// (ny, nx) numpy array — equivalent to:
//   np.array(flat).reshape((nx, ny), order='F').T
// ---------------------------------------------------------------------------
static py::array_t<double> flat_to_2d(
        const std::vector<double>& flat, int nx, int ny) {
    py::array_t<double> arr({ny, nx});
    auto buf = arr.mutable_unchecked<2>();
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            buf(j, i) = flat[static_cast<std::size_t>(j * nx + i)];
    return arr;
}

// ---------------------------------------------------------------------------
// Main entry point exposed to Python
// ---------------------------------------------------------------------------
static py::dict run_simulation(const py::dict& params_dict,
                               py::object progress_cb = py::none()) {
    KSParams p = dict_to_params(params_dict);

    KSSolver solver(p);

    // Wrap the optional Python callback so the C++ side can call it.
    // The heavy solve runs with the GIL released; we re-acquire it only
    // for the brief callback invocation.
    KSSolver::ProgressCB cpp_cb = nullptr;
    if (!progress_cb.is_none()) {
        cpp_cb = [&progress_cb](int step, int total) {
            py::gil_scoped_acquire gil;
            progress_cb(step, total);
        };
    }

    KSSolver::SnapshotData data;
    {
        py::gil_scoped_release release;
        data = solver.run(cpp_cb);
    }

    const int nx = p.nx, ny = p.ny;

    py::list rho_snaps, c_snaps, s_snaps;
    for (std::size_t t = 0; t < data.rho_snaps.size(); ++t) {
        rho_snaps.append(flat_to_2d(data.rho_snaps[t], nx, ny));
        c_snaps.append(  flat_to_2d(data.c_snaps[t],   nx, ny));
        s_snaps.append(  flat_to_2d(data.s_snaps[t],   nx, ny));
    }

    py::dict result;
    result["times"]         = data.times;
    result["rho_snapshots"] = rho_snaps;
    result["c_snapshots"]   = c_snaps;
    result["s_snapshots"]   = s_snaps;
    result["total_mass"]    = data.total_mass;
    result["max_density"]   = data.max_density;
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_ks_core, m) {
    m.doc() = "Keller-Segel C++ solver (Eigen SparseLU + pybind11). "
              "Drop-in replacement for the FiPy backend.";
    m.def("run_simulation", &run_simulation,
          py::arg("params"),
          py::arg("progress_cb") = py::none(),
          "Run the Keller-Segel simulation.\n\n"
          "params      : dict     — all fields of KellerSegelParams\n"
          "progress_cb : callable — optional callback(step, total) for progress\n"
          "returns     : dict     — times, rho_snapshots, c_snapshots, s_snapshots, "
                                   "total_mass, max_density");
}
