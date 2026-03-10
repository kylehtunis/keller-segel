#pragma once
#include <cmath>

// ---------------------------------------------------------------------------
// Scharfetter-Gummel scheme for the chemotaxis convection-diffusion term
//
//   ∂ρ/∂t = ∇·(D_ρ ∇ρ  −  χ ρ ∇c)
//
// The SG flux at the face between cells L (left/south) and R (right/north) is
//
//   F_{L→R} = (D_ρ / h) · [ B(Pe)·ρ_L  −  B(−Pe)·ρ_R ]
//
// where  Pe = χ·(c_R − c_L) / D_ρ  is the local face Péclet number
// and    B(x) = x / (eˣ − 1)        is the Bernoulli function.
//
// B(0) = 1 by L'Hôpital.  B is evaluated in a numerically stable way:
//   • |x| < 1e-8  : Taylor expansion  B(x) ≈ 1 − x/2 + x²/12 − x⁴/720
//   •  x  > 500   : B(x) ≈ 0  (eˣ − 1 overflows double)
//   • otherwise   : x / expm1(x)  (expm1 avoids catastrophic cancellation)
// ---------------------------------------------------------------------------

inline double bernoulli(double x) noexcept {
    constexpr double eps = 1e-8;
    if (x > 500.0)  return 0.0;             // overflow clamp: B → 0
    if (x < -500.0) return -x;              // B(x) → -x for x ≪ 0
    if (std::abs(x) < eps)
        // Taylor: B(x) = 1 - x/2 + x²/12 - x⁴/720 + ...
        return 1.0 - x * (0.5 - x * (1.0/12.0 - x * x / 720.0));
    return x / std::expm1(x);
}

// SG face flux:  F = (D_rho/h) * [B(Pe)*rhoL − B(-Pe)*rhoR]
// Positive value means net transport from L to R.
inline double sg_flux(double rhoL, double rhoR,
                      double cL,   double cR,
                      double D_rho, double chi, double h) noexcept {
    const double Pe = chi * (cR - cL) / D_rho;
    return (D_rho / h) * (bernoulli(Pe) * rhoL - bernoulli(-Pe) * rhoR);
}
