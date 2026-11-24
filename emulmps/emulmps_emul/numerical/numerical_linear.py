"""
Numerical (not symbolic-regression) linear matter power spectrum baseline.

Two ingredients, both computed from physics directly rather than via a
narrow-domain machine-learned fit:

  1. Eisenstein & Hu 1998 (zero-baryon) transfer function -- a properly
     derived semi-analytic approximation (not a regression fit calibrated
     to a specific finite simulation suite), so unlike e.g. Bartlett et
     al.'s symbolic corrections (which have explicit kmin/kmax validity
     bounds and warn when extrapolated) it has no narrow calibration range
     to silently extrapolate badly outside of.

  2. The linear growth factor D(a), obtained by NUMERICALLY INTEGRATING
     its governing second-order ODE for the actual w0waCDM expansion
     history, rather than via a symbolic fit (like Bartlett et al.'s
     growth_correction_R) or an external package (colossus). This mirrors
     numerical_halofit.py's approach exactly: derive the needed quantity
     from first principles for whatever cosmology is handed in, so there's
     no separate domain of validity to fall outside of, and no extra
     runtime dependency.

The overall amplitude is set DIRECTLY from As (the primordial power
spectrum amplitude) via the standard Delta_R^2 normalization -- no
As<->sigma8 conversion (symbolic or otherwise) is needed anywhere.

Works entirely in physical units (k in 1/Mpc, P(k) in Mpc^3) throughout,
matching numerical_halofit.py's convention -- no h-dependent unit-
convention bugs.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

EPS = 1e-30


def eisenstein_hu_transfer(k_hmpc, Om, Ob, h):
    """Eisenstein & Hu 1998 zero-baryon transfer function T(k), k in h/Mpc."""
    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7  # Tcmb0 = 2.7255 K

    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + 0.38 * np.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k_hmpc * h * s) ** 4))

    q = k_hmpc * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.e + 1.8 * q)
    return L0 / (L0 + C0 * q**2)


def pk_lin_z0_from_As(k_hmpc, As9, ns, Om, Ob, h):
    """
    Linear P(k) at z=0 [(Mpc/h)^3], k in h/Mpc -- directly from As (no
    sigma8 intermediate step, no fit anywhere): the standard primordial
    Delta_R^2 normalization, the transfer function squared, and the
    standard sub-horizon Poisson-equation conversion factor relating the
    curvature and matter power spectra. As9 = 10^9 * true A_s.
    """
    kpivot = 0.05  # Mpc^-1, standard pivot-scale convention
    tk = eisenstein_hu_transfer(k_hmpc, Om, Ob, h)
    return (
        2 * np.pi**2 / k_hmpc**3
        * (As9 * 1e-9) * (k_hmpc * h / kpivot) ** (ns - 1)
        * (2 * k_hmpc**2 * 2998.0**2 / 5.0 / Om) ** 2
        * tk**2
    )


def _growth_rhs(a, y, Om, w0, wa):
    """
    RHS of the linear growth ODE in scale-factor form:
        D''(a) + [3/a + (1/2) dlnE2/da] D'(a) - (3/(2a^2)) Omega_m(a) D(a) = 0
    for a general CPL (w0waCDM) expansion history:
        E(a)^2 = Om/a^3 + (1-Om) a^(-3(1+w0+wa)) exp(-3 wa (1-a))
    Reduces exactly to the standard LCDM growth ODE at w0=-1, wa=0.
    """
    D, Dp = y
    a3 = a * a * a
    Om_a3 = Om / a3
    de_term = (1.0 - Om) * a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    E2 = Om_a3 + de_term
    Omega_a = Om_a3 / E2

    dE2_da = -3.0 * Om_a3 / a + de_term * (-3.0 * (1.0 + w0 + wa) / a + 3.0 * wa)
    dlnE2_da = dE2_da / E2

    Dpp = -(3.0 / a + 0.5 * dlnE2_da) * Dp + 1.5 * Omega_a / (a * a) * D
    return [Dp, Dpp]


def linear_growth_factor(Om, w0, wa, a_targets, a_start=1e-4):
    """
    D(a)/D(1) via direct numerical ODE integration for a general w0waCDM
    expansion history -- growing-mode solution, with the matter-domination
    initial condition D(a_start)=a_start, D'(a_start)=1 (exact in that
    limit for any reasonable w0/wa, since dark energy is negligible at
    a_start~1e-4 regardless of its equation of state).

    a_targets: array of scale factors to evaluate D at -- all evaluated
    from a SINGLE ODE solve (via t_eval), not one solve per target, so the
    cost is one integration per cosmology, not per (cosmology, redshift).
    """
    a_targets = np.atleast_1d(np.asarray(a_targets, dtype=np.float64))
    a_eval = np.unique(np.clip(np.concatenate([a_targets, [1.0]]), a_start, 1.0))
    a_eval_sorted = np.sort(a_eval)

    sol = solve_ivp(_growth_rhs, (a_start, 1.0), [a_start, 1.0], args=(Om, w0, wa),
                     t_eval=a_eval_sorted, method="RK45", rtol=1e-7, atol=1e-12)
    D_of_a = dict(zip(a_eval_sorted, sol.y[0]))
    D1 = D_of_a[1.0]
    D_targets = np.array([D_of_a[np.clip(a, a_start, 1.0)] for a in a_targets])
    return D_targets / D1


def linear_baseline_grid(ks_mpc, zs, As9, ns, H0, Ob, Om, w0, wa):
    """One cosmology -> P_lin(k,z) baseline [Mpc^3], shape (n_z, n_k)."""
    h = H0 / 100.0
    k_hmpc = np.asarray(ks_mpc, dtype=np.float64) / h
    pk_z0_hmpc = pk_lin_z0_from_As(k_hmpc, As9, ns, Om, Ob, h)  # (n_k,), (Mpc/h)^3

    a_targets = 1.0 / (1.0 + np.asarray(zs, dtype=np.float64))
    growth = linear_growth_factor(Om, w0, wa, a_targets)  # (n_z,)

    pk_hmpc_z = pk_z0_hmpc[None, :] * (growth[:, None] ** 2)  # (n_z, n_k)
    pk_mpc3 = pk_hmpc_z / h**3
    pk_mpc3 = np.nan_to_num(pk_mpc3, nan=EPS, posinf=1e30, neginf=EPS)
    return np.clip(pk_mpc3, EPS, 1e30)


def linear_baseline_batch(ks_mpc, zs, cosmos, as_idx, ns_idx, h_idx, ob_idx, om_idx, w0_idx, wa_idx):
    """
    cosmos: (N, n_params) -> (N, n_z, n_k) baseline. Looped per sample
    (each cosmology needs its own ODE solve for the growth factor; see
    linear_growth_factor) -- a one-off preprocessing cost at training time
    and a small per-query cost at inference, same pattern already used for
    the halofit baseline in numerical_halofit.py.
    """
    n = cosmos.shape[0]
    out = np.empty((n, len(zs), len(ks_mpc)))
    for i in range(n):
        out[i] = linear_baseline_grid(
            ks_mpc, zs,
            As9=cosmos[i, as_idx], ns=cosmos[i, ns_idx], H0=cosmos[i, h_idx],
            Ob=cosmos[i, ob_idx], Om=cosmos[i, om_idx], w0=cosmos[i, w0_idx], wa=cosmos[i, wa_idx],
        )
    return out