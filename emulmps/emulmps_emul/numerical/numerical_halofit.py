"""
Numerical halofit baseline: computes the Takahashi et al. 2012 halofit
nonlinear P(k) prediction from an ARBITRARY given P_lin(k), by numerically
computing the halofit "ingredients" (the nonlinear wavenumber k_sigma, the
effective spectral index n_eff, and the spectral curvature C) via direct
integration -- NOT via a symbolic-regression approximation of those
quantities as functions of cosmological parameters.

This matters because a symbolic fit (e.g. Bartlett et al.'s As->sigma8,
ksigma(sigma8,...), neff(sigma8,...), C(sigma8,...)) is only valid within
whatever domain it was calibrated on, and silently extrapolates badly
outside it (this is what broke an earlier version of this emulator). Here,
k_sigma/n_eff/C are derived directly from the actual P_lin via the same
integral definitions used to derive halofit in the first place (Smith et al.
2003 eq. 52-54 / Takahashi et al. 2012), so there is no separate domain of
validity to fall outside of -- it's exact (up to numerical integration/
root-finding precision) for whatever P_lin is handed to it.

Works entirely in physical units (k in 1/Mpc, P(k) in Mpc^3, R in Mpc) --
the halofit fitting-formula polynomials (in n_eff, C) and the y=k/k_sigma
ratio are unit-system-agnostic as long as k and 1/R share the same length
unit, so no h-dependent unit conversion is needed anywhere in this module.

Everything is vectorized over an arbitrary batch of (P_lin, Omega_m, a)
triples at once (e.g. (n_samples, n_z, n_k)), with the k_sigma root-find
done via table lookup + interpolation (never a per-sample iterative solve),
so this is fast enough to run inside training preprocessing and inside the
per-prediction inference path.
"""

from __future__ import annotations

import numpy as np

_TWO_PI2 = 2.0 * np.pi**2

# Standard Takahashi et al. 2012 halofit fitting-formula coefficients
# (identical physics content to baseline_pofk.py's _PARS_TAKAHASHI -- kept
# here too so this module has no dependency on baseline_pofk.py).
_PARS_TAKAHASHI = np.array([
    1.5222, 2.8553, 2.3706, 0.9903, 0.2250, 0.6083,
    -0.5642, 0.5864, 0.5716, -1.5474, 0.3698, 2.0404, 0.8161, 0.5869,
    0.1971, -0.0843, 0.8460, 5.2105, 3.6902, -0.0307, -0.0585,
    0.0743, 6.0835, 1.3373, -0.1959, -5.5274, 2.0379, -0.7354, 0.3157,
    1.2490, 0.3980, -0.1682
], dtype=np.float64)

EPS = 1e-30


def _sigma2_table(ks, pk, R_grid):
    """
    sigma^2(R) = integral (dk/k) * [k^3 P(k) / (2 pi^2)] * exp(-(kR)^2)

    ks: (n_k,)
    pk: (..., n_k)  -- any leading batch shape
    R_grid: (n_R,)

    Returns sigma2: (..., n_R)
    """
    logk = np.log(ks)
    # trapezoidal weights along the log-k grid
    w = np.empty_like(ks)
    w[1:-1] = (logk[2:] - logk[:-2]) / 2.0
    w[0] = (logk[1] - logk[0]) / 2.0
    w[-1] = (logk[-1] - logk[-2]) / 2.0

    delta2 = (ks**3) * pk / _TWO_PI2  # (..., n_k)
    window = np.exp(-np.outer(R_grid, ks) ** 2)  # (n_R, n_k)
    integrand_weight = window * w[None, :]  # (n_R, n_k)

    batch_shape = pk.shape[:-1]
    delta2_flat = delta2.reshape(-1, ks.size)  # (B, n_k)
    sigma2_flat = delta2_flat @ integrand_weight.T  # (B, n_R)
    return sigma2_flat.reshape(*batch_shape, R_grid.size)


def compute_halofit_ingredients(ks, pk_lin, R_grid=None):
    """
    ks: (n_k,) physical k grid [1/Mpc]
    pk_lin: (..., n_k) linear P(k) [Mpc^3], any leading batch shape (e.g.
        (n_samples, n_z, n_k))
    R_grid: optional (n_R,) smoothing-scale grid [Mpc] to search over for
        the sigma(R)=1 crossing. Default spans very early-time (small R,
        relevant at high z where structure is weakly evolved) to very
        late-time/large-scale (large R) so it should bracket the true
        nonlinear scale across the full z=0-50 range this emulator covers.

    Returns (ksigma, neff, C), each shape pk_lin.shape[:-1] (i.e. one value
    per leading-batch-index, with the k axis reduced out). Any (sample,z)
    where sigma(R) never reaches 1 across R_grid (e.g. a pathological or
    extremely early-time case) falls back to the nearest edge of R_grid,
    which keeps outputs finite -- these should be rare and are flagged by
    the caller if the fraction is large.
    """
    if R_grid is None:
        R_grid = np.logspace(-4.5, 3.5, 500)

    sigma2 = _sigma2_table(ks, pk_lin, R_grid)  # (..., n_R)
    sigma2 = np.clip(sigma2, EPS, None)
    ln_sigma2 = np.log(sigma2)  # (..., n_R)
    ln_R = np.log(R_grid)

    batch_shape = pk_lin.shape[:-1]
    ln_sigma2_flat = ln_sigma2.reshape(-1, R_grid.size)  # (B, n_R)
    B = ln_sigma2_flat.shape[0]

    # sigma2 is monotonically decreasing in R (more smoothing -> less
    # variance), so ln_sigma2 decreases with ln_R. Find where it crosses 0
    # (sigma=1) via linear interpolation in (ln_R, ln_sigma2) space --
    # fully vectorized across the batch (no per-row Python loop / no
    # per-row np.interp call), since this needs to stay fast even for
    # millions of (sample, z) pairs at once (e.g. training preprocessing).
    ln_sigma2_rev = ln_sigma2_flat[:, ::-1]  # increasing along axis 1 (R decreasing)
    ln_R_rev = ln_R[::-1]

    mask = ln_sigma2_rev >= 0.0  # (B, n_R); True from the crossing point onward
    any_pos = mask.any(axis=1)
    idx_cross = np.argmax(mask, axis=1)  # first True index (0 if row is all False)

    idx_hi = np.clip(idx_cross, 0, R_grid.size - 1)
    idx_lo = np.clip(idx_cross - 1, 0, R_grid.size - 1)
    rows = np.arange(B)
    y_lo, y_hi = ln_sigma2_rev[rows, idx_lo], ln_sigma2_rev[rows, idx_hi]
    x_lo, x_hi = ln_R_rev[idx_lo], ln_R_rev[idx_hi]

    denom = y_hi - y_lo
    frac = np.where(np.abs(denom) > 1e-300, -y_lo / np.where(denom == 0, 1.0, denom), 0.5)
    ln_R_sigma = x_lo + frac * (x_hi - x_lo)

    always_positive = mask[:, 0]                    # sigma>=1 even at the largest R probed
    never_positive = ~any_pos                        # sigma<1 even at the smallest R probed
    ln_R_sigma = np.where(always_positive, ln_R_rev[0], ln_R_sigma)
    ln_R_sigma = np.where(never_positive, ln_R_rev[-1], ln_R_sigma)
    hit_edge = always_positive | never_positive

    R_sigma = np.exp(ln_R_sigma)
    ksigma = 1.0 / R_sigma

    # neff = -3 - d ln(sigma2)/d ln(R) at R=R_sigma
    # C     = - d^2 ln(sigma2)/d ln(R)^2 at R=R_sigma
    # via central finite differences, evaluated with the same sigma2(R)
    # integral (fresh evaluations at R_sigma*exp(+-h), not grid interpolation,
    # for better accuracy since R_sigma generally doesn't sit on the grid).
    h = 0.01
    R_plus = (R_sigma * np.exp(h)).reshape(*batch_shape)
    R_minus = (R_sigma * np.exp(-h)).reshape(*batch_shape)
    R_mid = R_sigma.reshape(*batch_shape)

    def sigma2_at(R_target):
        # R_target: batch_shape array, one R per (sample,z) -- evaluate
        # sigma2 exactly at these (not grid-restricted) via the same
        # integral, batched over the flattened index with each row using
        # its own R.
        Rf = R_target.reshape(-1)
        pkf = pk_lin.reshape(-1, ks.size)
        logk = np.log(ks)
        w = np.empty_like(ks)
        w[1:-1] = (logk[2:] - logk[:-2]) / 2.0
        w[0] = (logk[1] - logk[0]) / 2.0
        w[-1] = (logk[-1] - logk[-2]) / 2.0
        delta2 = (ks**3) * pkf / _TWO_PI2  # (B, n_k)
        window = np.exp(-(np.outer(Rf, ks)) ** 2)  # (B, n_k)
        s2 = np.sum(delta2 * window * w[None, :], axis=1)
        return np.clip(s2, EPS, None).reshape(*batch_shape)

    ln_s2_plus = np.log(sigma2_at(R_plus))
    ln_s2_minus = np.log(sigma2_at(R_minus))
    ln_s2_mid = np.log(sigma2_at(R_mid))  # NOT assumed to be exactly 0: R_sigma comes from
    # linear interpolation (not an exact root), so sigma(R_sigma) is only
    # approximately 1. The first derivative is insensitive to that small
    # offset, but the second derivative (C) is not -- evaluating explicitly
    # here rather than assuming ln_s2_mid=0 was verified against an
    # independent brentq-based reference to fix a real, non-negligible
    # error in C specifically (neff/ksigma were already correct).

    d1 = (ln_s2_plus - ln_s2_minus) / (2 * h)
    d2 = (ln_s2_plus - 2 * ln_s2_mid + ln_s2_minus) / (h * h)

    neff = (-3.0 - d1).reshape(*batch_shape)
    C = (-d2).reshape(*batch_shape)
    ksigma = ksigma.reshape(*batch_shape)
    hit_edge = hit_edge.reshape(*batch_shape)

    return ksigma, neff, C, hit_edge


def takahashi_pnl(ks, pk_lin, ksigma, neff, C, Om, a, w0=-1.0, wa=0.0):
    """
    Apply the Takahashi et al. 2012 halofit fitting formula given
    numerically-derived ksigma/neff/C.

    ks: (n_k,)
    pk_lin: (..., n_k)
    ksigma, neff, C: (...,)  [batch shape matching pk_lin.shape[:-1]]
    Om: scalar or (...,) broadcastable -- Omega_m today (z=0)
    a: (...,) scale factor, broadcastable to the same batch shape as ksigma
    w0, wa: scalar or (...,) broadcastable -- CPL dark-energy equation of
        state, w(a) = w0 + wa*(1-a). Default w0=-1, wa=0 recovers flat LCDM.
        These feed into Omega_m(a) (via the actual w0waCDM expansion
        history), which sets the f1/f2/f3 correction factors that
        interpolate the halofit fit between matter- and dark-energy-
        dominated behavior -- using the LCDM expansion history for a
        cosmology with w0/wa far from (-1, 0) would bias this in a way that
        correlates with how non-LCDM the dark energy is.

    Returns pk_nl: (..., n_k), same units as pk_lin (physical Mpc^3).
    """
    pars = _PARS_TAKAHASHI
    batch_shape = pk_lin.shape[:-1]

    ne2 = neff * neff
    ne3 = ne2 * neff
    ne4 = ne2 * ne2

    an = 10.0 ** np.clip(pars[0] + pars[1]*neff + pars[2]*ne2 + pars[3]*ne3 + pars[4]*ne4 - pars[5]*C, -30, 30)
    bn = 10.0 ** np.clip(pars[6] + pars[7]*neff + pars[8]*ne2 + pars[9]*C, -30, 30)
    cn = 10.0 ** np.clip(pars[10] + pars[11]*neff + pars[12]*ne2 + pars[13]*C, -30, 30)
    gamma = pars[14] + pars[15]*neff + pars[16]*C
    nu = 10.0 ** np.clip(pars[17] + pars[18]*neff, -30, 30)

    a = np.broadcast_to(a, batch_shape).astype(np.float64)
    Om_b = np.broadcast_to(Om, batch_shape).astype(np.float64)
    w0_b = np.broadcast_to(w0, batch_shape).astype(np.float64)
    wa_b = np.broadcast_to(wa, batch_shape).astype(np.float64)

    # Full w0waCDM (CPL) expansion history, not the LCDM special case --
    # E(a)^2 = Om*a^-3 + (1-Om)*a^(-3(1+w0+wa))*exp(-3*wa*(1-a)); reduces
    # exactly to the LCDM formula below when w0=-1, wa=0.
    matter_term = Om_b / a**3
    de_term = (1.0 - Om_b) * a ** (-3.0 * (1.0 + w0_b + wa_b)) * np.exp(-3.0 * wa_b * (1.0 - a))
    Omz = matter_term / (matter_term + de_term)
    f1 = Omz ** pars[19]
    f2 = Omz ** pars[20]
    f3 = Omz ** pars[21]

    alpha = np.abs(pars[22] + pars[23]*neff + pars[24]*ne2 + pars[25]*C)
    beta = pars[26] + pars[27]*neff + pars[28]*ne2 + pars[29]*ne3 + pars[30]*ne4 + pars[31]*C

    ksigma_b = ksigma[..., None]  # (..., 1)
    y = ks[(None,) * len(batch_shape) + (slice(None),)] / np.clip(ksigma_b, EPS, 1e15)  # (..., n_k)
    y = np.clip(y, EPS, 1e15)

    k2 = ks * ks
    k3 = k2 * ks
    inv_k3 = 1.0 / k3

    deltaH2 = (an[..., None] * y ** (3.0 * f1[..., None])) / (
        1.0 + bn[..., None] * y ** (f2[..., None]) + (cn[..., None] * f3[..., None] * y) ** (3.0 - gamma[..., None])
    )
    deltaH2 = deltaH2 / (1.0 + nu[..., None] / (y * y))
    ph = deltaH2 * _TWO_PI2 * inv_k3[(None,) * len(batch_shape) + (slice(None),)]

    deltaL2 = (k3[(None,) * len(batch_shape) + (slice(None),)] * pk_lin) / _TWO_PI2
    np.clip(deltaL2, 0.0, 1e10, out=deltaL2)  # generous defensive bound; only engages for
    # deliberately-pathological/extrapolated inputs, silences the resulting overflow warning
    # from (1+deltaL2)**beta without affecting any physically plausible value
    pq = pk_lin * (1.0 + deltaL2) ** (beta[..., None]) / (1.0 + alpha[..., None] * deltaL2) * \
        np.exp(-y / 4.0 - (y * y) / 8.0)

    return ph + pq


def halofit_baseline(ks, pk_lin, Om, a, w0=-1.0, wa=0.0, R_grid=None, chunk_size=20000):
    """
    Convenience wrapper: pk_lin -> (pk_nl_baseline, diagnostics).

    ks: (n_k,)
    pk_lin: (..., n_k), physical Mpc^3
    Om: scalar or (...,) -- Omega_m today
    a: (...,) scale factor (matching pk_lin.shape[:-1], e.g. (n_z,) broadcast
       to (n_samples, n_z) or already that shape)
    w0, wa: scalar or (...,) -- CPL dark-energy equation of state,
        w(a) = w0 + wa*(1-a). Default (-1, 0) recovers flat LCDM.
    chunk_size: max number of (sample,z) rows processed in one fully-
        vectorized pass. The vectorized root-find/derivative computation
        allocates several (B, n_R) and (B, n_k) arrays -- at full dataset
        scale (e.g. 100k samples x 52 z = 5.2M rows) that's tens of GB if
        done in one shot. Chunking keeps peak memory bounded to
        O(chunk_size * max(n_R, n_k)) regardless of total dataset size,
        while each chunk is still fully vectorized (no per-row Python loop).

    Returns pk_nl_baseline (..., n_k) [Mpc^3] and a dict with ksigma/neff/C/
    hit_edge for diagnostics.
    """
    pk_lin = np.clip(pk_lin, EPS, None)
    batch_shape = pk_lin.shape[:-1]
    n_k = pk_lin.shape[-1]
    B = int(np.prod(batch_shape)) if batch_shape else 1

    pk_flat = pk_lin.reshape(B, n_k)
    Om_flat = np.broadcast_to(Om, batch_shape).reshape(B)
    a_flat = np.broadcast_to(a, batch_shape).reshape(B)
    w0_flat = np.broadcast_to(w0, batch_shape).reshape(B)
    wa_flat = np.broadcast_to(wa, batch_shape).reshape(B)

    if B <= chunk_size:
        ksigma, neff, C, hit_edge = compute_halofit_ingredients(ks, pk_flat, R_grid=R_grid)
        pk_nl_flat = takahashi_pnl(ks, pk_flat, ksigma, neff, C, Om_flat, a_flat, w0_flat, wa_flat)
        pk_nl_flat = np.clip(np.nan_to_num(pk_nl_flat, nan=EPS, posinf=1e30, neginf=EPS), EPS, 1e30)
    else:
        pk_nl_flat = np.empty((B, n_k), dtype=np.float64)
        ksigma = np.empty(B, dtype=np.float64)
        neff = np.empty(B, dtype=np.float64)
        C = np.empty(B, dtype=np.float64)
        hit_edge = np.empty(B, dtype=bool)
        for start in range(0, B, chunk_size):
            stop = min(start + chunk_size, B)
            ks_c, neff_c, C_c, hit_c = compute_halofit_ingredients(ks, pk_flat[start:stop], R_grid=R_grid)
            pnl_c = takahashi_pnl(ks, pk_flat[start:stop], ks_c, neff_c, C_c,
                                   Om_flat[start:stop], a_flat[start:stop],
                                   w0_flat[start:stop], wa_flat[start:stop])
            pnl_c = np.clip(np.nan_to_num(pnl_c, nan=EPS, posinf=1e30, neginf=EPS), EPS, 1e30)
            pk_nl_flat[start:stop] = pnl_c
            ksigma[start:stop], neff[start:stop], C[start:stop], hit_edge[start:stop] = ks_c, neff_c, C_c, hit_c

    pk_nl = pk_nl_flat.reshape(*batch_shape, n_k)

    # Final sanity clip on the BOOST RATIO (pk_nl/pk_lin), not on any
    # intermediate quantity like y or deltaL2 -- those have legitimate wide
    # dynamic ranges even in normal (non-degenerate) deep-nonlinear cases
    # (y ~ hundreds at k~100 Mpc^-1 is completely ordinary), so capping them
    # directly would clip real physics. But a genuinely pathological case
    # (e.g. the k_sigma root-find hitting the R_grid edge, which can happen
    # for ~1/4 of points at very early times / extreme cosmologies) can
    # combine several such quantities into a boost enhancement of many
    # orders of magnitude (1e12+), which no real nonlinear power spectrum
    # exhibits -- clipping the final ratio to a still-extremely-generous
    # window catches that regardless of which internal quantity caused it.
    with np.errstate(invalid="ignore", divide="ignore"):
        boost = pk_nl / pk_lin
    boost = np.nan_to_num(boost, nan=1.0, posinf=1e5, neginf=1e-5)
    np.clip(boost, 1e-5, 1e5, out=boost)
    pk_nl = pk_lin * boost

    diag = {
        "ksigma": ksigma.reshape(batch_shape),
        "neff": neff.reshape(batch_shape),
        "C": C.reshape(batch_shape),
        "hit_edge": hit_edge.reshape(batch_shape),
    }
    return pk_nl, diag