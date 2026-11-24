"""
Fast neural emulator for the matter power spectra P_lin(k, z | cosmology)
and P_nl(k, z | cosmology).

METHOD (v5: numerical baselines for BOTH fields + per-redshift PCA)
-------------------------------------------
v3 emulated log10(P_lin) and log10(P_nl/P_lin) directly (no external
baseline), but compressed the FULL (n_z=52, n_k=500) grid with a single
GLOBAL PCA basis (26000-dim flattened). That turned out to be a poor fit for
this dataset: even with --max_components 400, PCA only reached ~77%
explained variance. A global PCA has to represent, in one shared low-rank
basis, both the nearly-flat boost~1 behavior at high z AND the strongly
nonlinear low-z/high-k structure -- fundamentally different "shapes" of
k-dependence.

v4 fixed this the standard way (matching CosmoPower/EuclidEmulator2
practice): compress EACH REDSHIFT SLICE SEPARATELY.

v5 (this version) ALSO gives P_lin a numerical baseline, exactly mirroring
how P_nl already had one (the Takahashi et al. halofit prediction computed
numerically from the true P_lin). P_lin's baseline is the Eisenstein & Hu
1998 zero-baryon transfer function -- a proper physics approximation, not a
machine-learned fit calibrated to a finite simulation suite, so (like
halofit) it has no narrow calibration range to extrapolate badly outside
of -- normalized DIRECTLY from As via the standard Delta_R^2 primordial
normalization (no As<->sigma8 conversion anywhere), with the redshift
dependence from a linear growth factor obtained by NUMERICALLY INTEGRATING
its governing ODE for the actual w0waCDM expansion history (see
numerical_linear.py), not a symbolic fit or an external package (colossus).
Both correction networks now only have to learn the residual on top of a
numerical baseline that already captures the bulk cosmology-dependent
shape -- the part a linear (PCA) basis struggles to represent efficiently
from scratch.

Everything else from v4 is kept:
    - work in log10 space (P(k,z) spans many decades)
    - per-(z,k)-bin standardization + PCA compression (per-z-slice)
    - a shared-trunk + per-redshift-head MLP: cosmology -> concatenated
      per-z PCA coefficients
    - defensive clipping of extreme per-bin log10(P) values before
      standardizing/PCA-fitting
    - trimmed-loss training (drop the worst loss_trim_frac fraction of
      samples each epoch, so a few extreme/unphysical points can't dominate)
    - an optional triangle cut excluding a disfavored corner of parameter
      space before training
    - a pure-NumPy fused inference path for speed.

USAGE
-----
    python pk_emulator.py --outdir ./emulator_out --n_batches 20 \
        --cosmos "/path/to/input_hs/train_w0wacdm_mps_ml_*_tfree_mnufree.npy" \
        --pk_lin "/path/to/output_hs/train_w0wacdm_expanded_*_mead2020_Tfree_mnufree_pklin.npy" \
        --pk_nl  "/path/to/output_hs/train_w0wacdm_expanded_*_mead2020_Tfree_mnufree_pknonlin.npy"

--pk_nl/--pk_lin/--cosmos each take a glob pattern with a single '*' as a
per-batch numeric id (shared across all three, so batches stay aligned);
--n_batches caps how many of the available batches get loaded (default:
all of them). A plain single-file path with no '*' also still works.

The k/z grid is fixed for this project and no longer loaded from files --
it's computed directly (see load_data).

If no matching files are found, a small synthetic dataset is generated
automatically so the pipeline can be smoke-tested end-to-end.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .numerical.numerical_halofit import halofit_baseline
from .numerical.numerical_linear import linear_baseline_batch

PARAM_NAMES = ["As", "ns", "H0", "Omega_b", "Omega_m", "w", "wa", "Tagn", "mnu"]
AS_IDX = PARAM_NAMES.index("As")
NS_IDX = PARAM_NAMES.index("ns")
H0_IDX = PARAM_NAMES.index("H0")
OB_IDX = PARAM_NAMES.index("Omega_b")
OM_IDX = PARAM_NAMES.index("Omega_m")
W0_IDX = PARAM_NAMES.index("w")
WA_IDX = PARAM_NAMES.index("wa")
EPS = 1e-30  # guards against log10(0) for pathological / masked entries


# =============================================================================
# 1. Data loading
# =============================================================================

def _list_available_batches(pattern):
    """Given a path `pattern`, return the sorted (numerically) list of batch
    ids actually present on disk, as strings (so they can be substituted
    straight back into any of the three patterns via `.replace('*', id)`).
    If `pattern` has no '*' at all, treats it as a single plain file path:
    returns [""] if it exists (pattern.replace('*', '') is a no-op, so this
    reuses the exact same downstream loading code path) or [] if not."""
    if "*" not in pattern:
        return [""] if Path(pattern).exists() else []
    regex = re.compile("^" + re.escape(pattern).replace(re.escape("*"), r"(\d+)") + "$")
    batch_ids = set()
    for f in glob.glob(pattern):
        m = regex.match(f)
        if m:
            batch_ids.add(m.group(1))
    return sorted(batch_ids, key=int)


def load_data(cosmos_pattern, pk_lin_pattern, pk_nl_pattern, n_batches=None, n_z=52, n_k=500, n_params=9):
    """
    Load batched training data from glob-style path patterns, where each
    pattern contains a single '*' standing in for a per-batch numeric id
    shared across all three patterns (e.g.
    ".../train_..._{batch}_..._pklin.npy"). Batch ids are determined from
    whichever batches exist on disk for pk_nl_pattern, then the SAME ids are
    substituted into the other two patterns -- keeps cosmos/pk_lin/pk_nl
    aligned per batch even if directory listings would otherwise come back
    in different orders. `n_batches`, if given, keeps only batches with
    numeric id < n_batches (a threshold on the id itself, NOT a count of how
    many batches to load) -- so a reserved batch number (e.g. a held-out
    test batch numbered 1000) is reliably excluded by passing that same
    number, regardless of how many/which lower-numbered batches exist.

    ks/zs are no longer loaded from files -- every batch in this project
    shares the same fixed (k, z) grid, so they're just computed directly.

    Each batch file is loaded with mmap_mode='r': numpy memory-maps the
    raw file (backed by disk, not a freshly-allocated RAM buffer) instead
    of reading its full contents into memory up front, so the only large
    RAM allocation during loading is the destination float64 array itself
    -- meaningfully lowers peak memory versus loading the raw arrays
    directly and then casting, which briefly needs both the raw and cast
    copies resident at once, especially significant here since each batch
    file is on the order of ~100+MB and there can be many of them.

    Falls back to a small synthetic dataset (matching this project's real
    ks/zs grid) if no batch files are found, so the pipeline can still be
    smoke-tested without real data.
    """
    ks = np.logspace(-5.1, 2, n_k)
    zs = np.concatenate((
        np.linspace(0, 3, 33, endpoint=False),
        np.linspace(3, 10, 7, endpoint=False),
        np.linspace(10, 50, 12),
    ))[:n_z]

    batch_ids = _list_available_batches(pk_nl_pattern)
    if not batch_ids:
        print("!! No batch files found matching the given patterns -- generating a small synthetic dataset for a smoke test.")
        rng = np.random.default_rng(0)
        n = 600
        As9 = rng.uniform(1.0, 4.0, n)
        ns = rng.uniform(0.90, 1.02, n)
        H0 = rng.uniform(60.0, 75.0, n)
        Ob = rng.uniform(0.03, 0.07, n)
        Om = rng.uniform(0.20, 0.40, n)
        w0 = rng.uniform(-1.2, -0.8, n)
        wa = rng.uniform(-0.5, 0.5, n)  # column 6 is wa directly, not w0+wa
        Tagn = rng.uniform(7.2, 8.3, n)
        mnu = rng.uniform(0.0, 0.2, n)
        cosmos = np.stack([As9, ns, H0, Ob, Om, w0, wa, Tagn, mnu], axis=1)

        # A deliberately more elaborate synthetic P(k,z) than v3's, so this
        # smoke test actually exercises "does per-z PCA compress much
        # better than global PCA": several independent, differently-shaped
        # k-dependent response modes, each activated by a different subset
        # of cosmological parameters, plus a BAO-wiggle-like oscillation and
        # a Tagn-driven high-k suppression feature -- this easily needs many
        # global PCA components to capture jointly across 52 z's, while at
        # fixed z it's still just a smooth function of k.
        k_pivot = 0.05
        pk_lin = np.empty((n, n_z, n_k))
        pk_nl = np.empty((n, n_z, n_k))
        for i in range(n):
            growth = np.exp(-0.5 * Om[i] * zs)
            wiggle = 1.0 + 0.05 * np.sin(8 * np.log(ks / k_pivot)) * np.exp(-ks / 2.0)
            shape = (ks / k_pivot) ** (ns[i] - 1) * np.exp(-ks / 5.0) * wiggle
            amp = As9[i] * 1e3
            lin = amp * shape[None, :] * growth[:, None]
            suppress = 1.0 / (1.0 + ((ks[None, :] / (0.5 + 0.3 * (Tagn[i] - 7.6))) ** 2))
            boost = (1.0 + 0.3 * (ks[None, :] / 1.0) ** 1.2 * growth[:, None]) * suppress
            pk_lin[i] = lin
            pk_nl[i] = lin * boost
    else:
        if n_batches is not None and batch_ids != [""]:
            n_before = len(batch_ids)
            batch_ids = [b for b in batch_ids if int(b) < n_batches]
            n_excluded = n_before - len(batch_ids)
            if n_excluded:
                print(f"[batch filter] --n_batches {n_batches}: keeping batch ids < {n_batches} "
                      f"({len(batch_ids)}/{n_before} available batches; excluded {n_excluded} with "
                      f"id >= {n_batches}, e.g. a reserved test batch numbered {n_batches} itself "
                      f"is correctly excluded).")
        if batch_ids == [""]:
            print("Loading 1 file (no batch pattern)")
        else:
            preview = ", ".join(batch_ids[:5]) + (", ..." if len(batch_ids) > 5 else "")
            print(f"Loading {len(batch_ids)} batch(es) (ids: {preview})")

        pk_nl_parts, pk_lin_parts, cosmos_parts = [], [], []
        for b in batch_ids:
            pk_nl_parts.append(np.load(pk_nl_pattern.replace("*", b), mmap_mode="r").astype(np.float64))
            pk_lin_parts.append(np.load(pk_lin_pattern.replace("*", b), mmap_mode="r").astype(np.float64))
            cosmos_parts.append(np.load(cosmos_pattern.replace("*", b), mmap_mode="r").astype(np.float64))
        pk_nl = np.concatenate(pk_nl_parts, axis=0)
        pk_lin = np.concatenate(pk_lin_parts, axis=0)
        cosmos = np.concatenate(cosmos_parts, axis=0)

        print(f"Loaded real data: pk_nl {pk_nl.shape}, pk_lin {pk_lin.shape}, "
              f"cosmos {cosmos.shape}, ks {ks.shape}, zs {zs.shape}")

    assert pk_nl.shape == pk_lin.shape, "pk_nl and pk_lin shapes must match"
    assert pk_nl.shape[0] == cosmos.shape[0], "sample count mismatch between pk arrays and cosmos"
    assert pk_nl.shape[1] == zs.shape[0], "zs length must match pk arrays' z-axis"
    assert pk_nl.shape[2] == ks.shape[0], "ks length must match pk arrays' k-axis"
    return pk_nl, pk_lin, cosmos, ks, zs


# =============================================================================
# 1b. Optional triangle cut: exclude a disfavored corner of parameter space
# =============================================================================

def triangle_cut_mask(cosmos, x_idx, y_idx, anchor1, anchor2):
    """
    Boolean mask, True = KEEP. Cuts everything ABOVE the line through
    `anchor1` and `anchor2` in the (cosmos[:,x_idx], cosmos[:,y_idx]) plane
    -- i.e. points whose y-value exceeds the line's y-value at their x.
    ("Above" = higher y at fixed x, using the ordinary y-up convention,
    regardless of the line's slope sign.)
    """
    x, y = cosmos[:, x_idx], cosmos[:, y_idx]
    x1, y1 = anchor1
    x2, y2 = anchor2
    if x2 == x1:
        raise ValueError("triangle_cut_mask: anchor1 and anchor2 have the same x-value.")
    m = (y2 - y1) / (x2 - x1)
    line_y = y1 + m * (x - x1)
    return y <= line_y  # keep everything on/below the line


def apply_triangle_cut(pk_nl, pk_lin, cosmos, x_idx, y_idx, anchor1, anchor2, label=""):
    mask = triangle_cut_mask(cosmos, x_idx, y_idx, anchor1, anchor2)
    n_cut = int((~mask).sum())
    if n_cut:
        print(f"[triangle cut{f' ({label})' if label else ''}] removing {n_cut}/{len(mask)} "
              f"samples ({100*n_cut/len(mask):.2f}%) above the line through "
              f"({anchor1[0]:g},{anchor1[1]:g}) and ({anchor2[0]:g},{anchor2[1]:g}) in "
              f"({PARAM_NAMES[x_idx]}, {PARAM_NAMES[y_idx]}) space.")
    return pk_nl[mask], pk_lin[mask], cosmos[mask]


def apply_param_range_cut(pk_nl, pk_lin, cosmos, param_idx, min_val=None, max_val=None, label=""):
    """
    Drops samples with cosmos[:,param_idx] outside [min_val, max_val].
    Unlike loss trimming (which only affects the neural network's training
    loss, epoch by epoch), this excludes samples from the ENTIRE pipeline
    -- including PCA fitting. A shared PCA basis fit across both "easy" and
    "hard" cosmologies spends some of its component budget representing
    rare, extreme excursions (e.g. strong Tagn-driven suppression); removing
    those cosmologies before PCA is fit lets the basis (and consequently the
    network's job) concentrate entirely on representing the bulk precisely,
    at the cost of the emulator having no coverage at all for the excluded
    region.
    """
    x = cosmos[:, param_idx]
    mask = np.ones(cosmos.shape[0], dtype=bool)
    if min_val is not None:
        mask &= x >= min_val
    if max_val is not None:
        mask &= x <= max_val
    n_cut = int((~mask).sum())
    if n_cut:
        lo = f"{min_val:g}" if min_val is not None else "-inf"
        hi = f"{max_val:g}" if max_val is not None else "inf"
        print(f"[param range cut{f' ({label})' if label else ''}] removing {n_cut}/{len(mask)} "
              f"samples ({100*n_cut/len(mask):.2f}%) with {PARAM_NAMES[param_idx]} outside [{lo}, {hi}]")
    return pk_nl[mask], pk_lin[mask], cosmos[mask]


def apply_multivariate_outlier_cut(pk_nl, pk_lin, cosmos, k_neighbors=10, cut_percentile=98.0, label=""):
    """
    Drops samples that are ISOLATED in the 9-D parameter space -- i.e. their
    distance to their k-th nearest training neighbor (in min-max-normalized
    space, appropriate for LHC/uniform-sampled priors) is unusually large.
    This directly measures the thing that actually matters for network
    generalization (how much nearby training signal exists to interpolate
    from), rather than inferring it indirectly from distributional
    extremity.

    A z-score/std-based check (the previous version of this function) is
    fundamentally unreliable here for two reasons:
      1. Most LHC designs sample from BOUNDED UNIFORM priors. A uniform
         distribution U(a,b) has std=(b-a)/sqrt(12), so even a point sitting
         EXACTLY at the prior edge can never exceed |z-score| ~ 1.73 -- any
         threshold above that (like the 4.0 default this replaced) is
         mathematically unreachable no matter how cornered a point is.
      2. Per-dimension (or joint-Gaussian/Mahalanobis) checks are blind to
         points that sit near SEVERAL parameter-space edges simultaneously
         without any single dimension being individually extreme -- e.g. a
         point ~1.2-1.6 sigma off in five different parameters at once can
         be a genuinely isolated corner of the sampled hypercube (corners
         are always sparser than the interior) while triggering no
         reasonable per-axis threshold at all.

    k-NN distance in normalized space sidesteps both problems: it directly
    asks "does this point have close training neighbors," which is
    distribution-shape-agnostic and inherently multivariate.
    """
    lo, hi = cosmos.min(axis=0), cosmos.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normed = (cosmos - lo) / span

    from scipy.spatial import cKDTree
    tree = cKDTree(normed)
    k_query = min(k_neighbors + 1, normed.shape[0])  # +1: a point's own nearest "neighbor" is itself (dist 0)
    dists, _ = tree.query(normed, k=k_query)
    knn_dist = dists[:, -1] if k_query > 1 else dists  # distance to the k-th REAL (non-self) neighbor

    threshold = np.percentile(knn_dist, cut_percentile)
    mask = knn_dist <= threshold
    n_cut = int((~mask).sum())
    print(f"[isolation cut{f' ({label})' if label else ''}] removing {n_cut}/{len(mask)} samples "
          f"({100*n_cut/len(mask):.2f}%) whose {k_neighbors}-nearest-neighbor distance in "
          f"min-max-normalized parameter space is in the most-isolated {100-cut_percentile:g}% "
          f"(least nearby training signal) -- distance threshold={threshold:.4f}, "
          f"max observed={knn_dist.max():.4f}.")
    return pk_nl[mask], pk_lin[mask], cosmos[mask]


def diagnose_nan_pattern(pk_nl, pk_lin, zs):
    """Prints where NaNs concentrate -- by redshift, and by how many
    samples are affected vs. how badly -- so it's possible to tell a
    handful of pathological/failed cosmologies apart from a systematic
    region (e.g. an extrapolated k-tail) affecting many samples lightly."""
    for name, arr in [("pk_nl", pk_nl), ("pk_lin", pk_lin)]:
        nan_mask = np.isnan(arr)
        n_total = nan_mask.sum()
        if n_total == 0:
            continue
        frac_by_z = nan_mask.mean(axis=(0, 2))  # (n_z,)
        worst_z = np.argsort(-frac_by_z)[:5]
        per_sample_frac = nan_mask.mean(axis=(1, 2))  # (N,)
        n_samples_any = int((per_sample_frac > 0).sum())
        print(f"    [{name}] {n_total} NaN entries total ({100*n_total/arr.size:.4f}% of grid); "
              f"{n_samples_any}/{arr.shape[0]} samples have at least one NaN "
              f"(worst single sample: {100*per_sample_frac.max():.2f}% of its grid).")
        print(f"        worst redshift indices by NaN fraction: " +
              ", ".join(f"z={zs[zi]:.2f} ({100*frac_by_z[zi]:.3f}%)" for zi in worst_z))


def apply_nan_quality_cut(pk_nl, pk_lin, cosmos, max_nan_frac=0.02, label=""):
    """
    Drops entire samples whose fraction of NaN entries (across pk_nl and
    pk_lin combined) exceeds `max_nan_frac`. These are almost certainly
    cosmologies where the underlying simulation/Boltzmann run failed or
    went numerically unstable -- not just "a bit noisy" but genuinely
    untrustworthy, including at the (z,k) bins that AREN'T NaN (a run
    unstable enough to produce NaNs at some bins has no particular reason
    to be trustworthy at the neighboring ones either). Samples that pass
    this cut may still have a handful of scattered NaN entries; those are
    handled downstream via masked least-squares projection (see
    FieldCompressor.transform_to_coeffs), not imputation, so no fabricated
    cross-sample value ever becomes part of a specific sample's target.
    """
    n = cosmos.shape[0]
    frac = (np.isnan(pk_nl).sum(axis=(1, 2)) + np.isnan(pk_lin).sum(axis=(1, 2))) / (2 * pk_nl.shape[1] * pk_nl.shape[2])
    keep = frac <= max_nan_frac
    n_cut = int((~keep).sum())
    if n_cut:
        print(f"[NaN quality cut{f' ({label})' if label else ''}] removing {n_cut}/{n} samples "
              f"({100*n_cut/n:.2f}%) with >{100*max_nan_frac:.1f}% NaN entries in pk_nl/pk_lin -- "
              f"treated as failed/unstable runs, not trustworthy even at their non-NaN bins.")
    return pk_nl[keep], pk_lin[keep], cosmos[keep]


# =============================================================================
# 2. Targets: log10(P_lin/P_lin_baseline), and log10(P_nl/P_nl_halofit_baseline)
#    -- BOTH fields are corrections relative to a NUMERICAL baseline (never a
#    symbolic-regression fit, so neither has a narrow domain of validity to
#    extrapolate badly outside of)
# =============================================================================

def compute_direct_targets(pk_nl, pk_lin, cosmos, ks, zs, clip=15.0, verbose=True, return_baseline=False):
    """
    Returns (L_lin_corr, L_boost_corr), each (N, n_z, n_k):
        L_lin_corr(k,z)   = log10( P_lin_true(k,z) / P_lin_baseline(k,z) )
        L_boost_corr(k,z) = log10( P_nl_true(k,z) / P_nl_halofit_baseline(k,z) )

    P_lin_baseline is the Eisenstein & Hu 1998 zero-baryon transfer function
    (a proper physics approximation, not a machine-learned fit to a finite
    simulation suite -- no narrow calibration range to extrapolate badly
    outside of) normalized DIRECTLY from As via the standard Delta_R^2
    primordial normalization (no As<->sigma8 conversion needed anywhere),
    with the redshift dependence from a linear growth factor obtained by
    NUMERICALLY INTEGRATING its governing ODE for the actual w0waCDM
    expansion history (see numerical_linear.py) -- not a symbolic fit, not
    an external package, so again no separate domain of validity.

    P_nl_halofit_baseline is the Takahashi et al. halofit prediction
    computed NUMERICALLY from the (here, exact/true) P_lin -- see
    numerical_halofit.py. Same reasoning: k_sigma/n_eff/C are derived
    directly from whatever P_lin is handed in, no narrow-domain fit.

    Both correction networks only have to learn the RESIDUAL on top of
    their respective baselines (mainly the E&H98-vs-truth imperfections/BAO
    wiggles for L_lin_corr, and the AGN-feedback signature for
    L_boost_corr, since vanilla halofit has no Tagn dependence), which
    should be far more PCA-compressible than either raw field -- each
    baseline already captures the bulk cosmology-dependent SHAPE, which is
    the part a linear (PCA) basis struggles to represent efficiently from
    scratch.

    NaNs already present in the raw data are preserved and handled
    downstream by FieldCompressor's per-bin imputation. `clip` bounds each
    field to +-clip before standardizing/PCA-fitting.
    """
    n, nz, nk = pk_lin.shape

    pk_lin_baseline = linear_baseline_batch(
        ks, zs, cosmos, as_idx=AS_IDX, ns_idx=NS_IDX, h_idx=H0_IDX,
        ob_idx=OB_IDX, om_idx=OM_IDX, w0_idx=W0_IDX, wa_idx=WA_IDX,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        L_lin_true = np.log10(np.where(pk_lin > 0, pk_lin, np.nan))
        L_lin_corr = L_lin_true - np.log10(np.clip(pk_lin_baseline, EPS, None))

    Om = cosmos[:, OM_IDX][:, None]                      # (n, 1) -> broadcasts to (n, nz)
    w0 = cosmos[:, W0_IDX][:, None]
    wa = cosmos[:, WA_IDX][:, None]
    a = (1.0 / (1.0 + zs))[None, :]                       # (1, nz) -> broadcasts to (n, nz)
    Om_b = np.broadcast_to(Om, (n, nz))
    w0_b = np.broadcast_to(w0, (n, nz))
    wa_b = np.broadcast_to(wa, (n, nz))
    a_b = np.broadcast_to(a, (n, nz))

    pk_lin_safe = np.clip(np.nan_to_num(pk_lin_baseline, nan=EPS, posinf=1e30, neginf=EPS), EPS, 1e30)
    pk_nl_hf, diag = halofit_baseline(ks, pk_lin_safe, Om_b, a_b, w0_b, wa_b)

    hit_frac = diag["hit_edge"].mean()
    if verbose and hit_frac > 0.001:
        print(f"    [compute_direct_targets] numerical halofit root-find hit the R_grid edge "
              f"for {100*hit_frac:.3f}% of (sample,z) points (likely very early-time/weakly-"
              f"nonlinear cases) -- baseline falls back to the grid edge there; widen R_grid in "
              f"numerical_halofit.py if this fraction is large.")

    with np.errstate(invalid="ignore", divide="ignore"):
        L_nl = np.log10(np.where(pk_nl > 0, pk_nl, np.nan))
        ratio = 10.0 ** (L_nl - np.log10(np.clip(pk_nl_hf, EPS, None)))
        L_boost_corr = np.log10(np.where(ratio > 0, ratio, np.nan))

    n_clipped = 0
    if clip is not None:
        for arr in (L_lin_corr, L_boost_corr):
            bad = np.abs(arr) > clip
            n_clipped += int(np.nansum(bad))
            np.clip(arr, -clip, clip, out=arr)

    if clip is not None and n_clipped and verbose:
        print(f"    [compute_direct_targets] clipped {n_clipped} bins with "
              f"|log10 value| > {clip} before PCA/standardization.")

    if return_baseline:
        return L_lin_corr, L_boost_corr, pk_lin_baseline, pk_nl_hf
    return L_lin_corr, L_boost_corr


# =============================================================================
# 3. Preprocessing: per-(z,k)-bin standardization + PER-REDSHIFT PCA
# =============================================================================

@dataclass
class FieldCompressor:
    """Standardize a flattened field then PCA-compress it, with a fused
    linear map for fast inverse reconstruction. Used once per redshift
    slice (operates on a (N, n_k) array)."""

    var_threshold: float = 0.9999
    max_components: int = 60

    scaler_: StandardScaler = field(default=None, repr=False)
    pca_: PCA = field(default=None, repr=False)
    n_components_: int = 0
    fill_: np.ndarray = field(default=None, repr=False)
    W_out_: np.ndarray = field(default=None, repr=False)
    b_out_: np.ndarray = field(default=None, repr=False)

    def _compute_fill(self, X):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_fill = np.nanmean(X, axis=0)
            global_fill = np.nanmean(X)
        n_fully_nan_cols = int(np.isnan(col_fill).sum())
        if n_fully_nan_cols:
            col_fill = np.where(np.isnan(col_fill), global_fill, col_fill)
        return col_fill, n_fully_nan_cols

    def _impute(self, X):
        mask = np.isnan(X)
        if not mask.any():
            return X
        out = X.copy()
        out[mask] = np.broadcast_to(self.fill_, X.shape)[mask]
        return out

    def fit(self, X, verbose=True):
        n_nan = int(np.isnan(X).sum())
        if n_nan:
            self.fill_, n_fully_nan_cols = self._compute_fill(X)
            X = self._impute(X)
        else:
            self.fill_ = np.zeros(X.shape[1])
            n_fully_nan_cols = 0

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)

        n_cap = min(self.max_components, Z.shape[0] - 1, Z.shape[1])
        pca_full = PCA(n_components=n_cap, svd_solver="randomized", random_state=0)
        pca_full.fit(Z)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, self.var_threshold) + 1)
        n_keep = max(1, min(n_keep, n_cap))
        self.n_components_ = n_keep
        self.pca_ = pca_full
        self.pca_.components_ = pca_full.components_[:n_keep]
        self.pca_.mean_ = pca_full.mean_
        self.pca_.n_components_ = n_keep

        std = self.scaler_.scale_
        mean = self.scaler_.mean_
        self.W_out_ = self.pca_.components_ * std[None, :]
        self.b_out_ = self.pca_.mean_ * std + mean

        self._n_nan = n_nan
        self._n_fully_nan_cols = n_fully_nan_cols
        self._hit_cap = (n_keep == n_cap and n_cap == self.max_components)
        self._var_reached = float(cum_var[-1])
        return self

    def transform_to_coeffs(self, X):
        """For rows with no NaN: standard vectorized PCA transform (fast
        path, the common case). For rows WITH NaN: coefficients are solved
        via masked least-squares using ONLY that row's observed (non-NaN)
        entries -- NOT by imputing a fabricated value and running it through
        the full transform. Imputing with the cross-sample mean and then
        using the resulting coefficient as this sample's training target
        would bake a value from OTHER cosmologies into what the network is
        told is the correct answer for this one -- wrong in a way that's
        correlated with whatever caused the NaN (a failed/unstable run for
        that specific cosmology) in the first place. Rows with too few
        observed entries to constrain a least-squares fit (fewer than
        n_components_+2) fall back to all-zero coefficients, i.e. "predict
        the safe per-bin baseline" rather than an unconstrained/fabricated
        deviation from it.
        """
        n = X.shape[0]
        nan_mask = np.isnan(X)
        has_nan = nan_mask.any(axis=1)
        coeffs = np.empty((n, self.n_components_))

        if (~has_nan).any():
            Z = self.scaler_.transform(X[~has_nan])
            coeffs[~has_nan] = (Z - self.pca_.mean_) @ self.pca_.components_.T

        mean_, scale_ = self.scaler_.mean_, self.scaler_.scale_
        pca_mean, components = self.pca_.mean_, self.pca_.components_
        for i in np.where(has_nan)[0]:
            valid = ~nan_mask[i]
            if valid.sum() < self.n_components_ + 2:
                coeffs[i] = 0.0
                continue
            z_valid = (X[i, valid] - mean_[valid]) / scale_[valid]
            z_centered = z_valid - pca_mean[valid]
            A = components[:, valid].T  # (n_valid, n_components)
            c, *_ = np.linalg.lstsq(A, z_centered, rcond=None)
            coeffs[i] = c
        return coeffs

    def inverse_from_coeffs(self, coeffs):
        return coeffs @ self.W_out_ + self.b_out_

    def explained_variance(self):
        return float(np.cumsum(self.pca_.explained_variance_ratio_[: self.n_components_])[-1])


@dataclass
class PerZFieldCompressor:
    """Wraps one FieldCompressor per redshift slice: compresses each
    (N, n_k) slice of a (N, n_z, n_k) field independently. Much more
    efficient than one global PCA over the flattened (n_z*n_k) field when
    the k-dependence shape changes qualitatively across z (e.g. nearly-flat
    boost~1 at high z vs strongly nonlinear structure at low z)."""

    n_z: int
    var_threshold: float = 0.9999
    max_components_per_z: int = 60

    compressors_: list = field(default_factory=list, repr=False)

    def fit(self, X3, label=""):
        """X3: (N, n_z, n_k)."""
        self.compressors_ = []
        total_nan = 0
        n_hit_cap = 0
        var_reached = []
        for zi in range(self.n_z):
            fc = FieldCompressor(self.var_threshold, self.max_components_per_z).fit(X3[:, zi, :], verbose=False)
            self.compressors_.append(fc)
            total_nan += fc._n_nan
            n_hit_cap += int(fc._hit_cap)
            var_reached.append(fc._var_reached)

        total_components = sum(fc.n_components_ for fc in self.compressors_)
        mean_var = float(np.mean(var_reached))
        min_var = float(np.min(var_reached))
        print(f"[PerZFieldCompressor{f' ({label})' if label else ''}] {self.n_z} redshift slices, "
              f"{total_components} total PCA components ({total_components/self.n_z:.1f}/slice avg), "
              f"mean explained var {100*mean_var:.4f}%, worst slice {100*min_var:.4f}%"
              + (f" -- {n_hit_cap} slice(s) hit the --max_components_per_z cap, consider raising it"
                 if n_hit_cap else ""))
        if total_nan:
            print(f"    ({total_nan} total NaN entries imputed with per-(z,k)-bin training mean across all slices)")
        return self

    def transform_to_coeffs(self, X3):
        return np.concatenate([fc.transform_to_coeffs(X3[:, zi, :]) for zi, fc in enumerate(self.compressors_)], axis=1)

    def inverse_from_coeffs(self, coeffs):
        n = coeffs.shape[0]
        n_k = self.compressors_[0].W_out_.shape[1]
        out = np.empty((n, self.n_z, n_k))
        idx = 0
        for zi, fc in enumerate(self.compressors_):
            nc = fc.n_components_
            out[:, zi, :] = fc.inverse_from_coeffs(coeffs[:, idx:idx + nc])
            idx += nc
        return out

    @property
    def n_components_per_z(self):
        return np.array([fc.n_components_ for fc in self.compressors_], dtype=np.int64)

    @property
    def total_components(self):
        return int(sum(fc.n_components_ for fc in self.compressors_))

    def stacked_W_b(self):
        """Concatenate all per-slice fused inverse-transform matrices for
        saving: W (total_components, n_k), b (n_z, n_k)."""
        W = np.concatenate([fc.W_out_ for fc in self.compressors_], axis=0)
        b = np.stack([fc.b_out_ for fc in self.compressors_], axis=0)
        return W, b


@dataclass
class Preprocessor:
    n_z: int = 52
    n_k: int = 500
    var_threshold: float = 0.9999
    max_components_per_z: int = 60
    target_clip: float = 15.0

    ks_: np.ndarray = field(default=None, repr=False)
    zs_: np.ndarray = field(default=None, repr=False)
    cosmo_scaler_: StandardScaler = field(default=None, repr=False)
    lin_compressor_: PerZFieldCompressor = field(default=None, repr=False)
    boost_compressor_: PerZFieldCompressor = field(default=None, repr=False)
    coeff_scaler_: StandardScaler = field(default=None, repr=False)

    def fit_transform(self, pk_nl, pk_lin, cosmos, ks, zs):
        self.ks_, self.zs_ = np.asarray(ks, dtype=np.float64), np.asarray(zs, dtype=np.float64)
        L_lin, L_boost = compute_direct_targets(pk_nl, pk_lin, cosmos, self.ks_, self.zs_, clip=self.target_clip)

        self.cosmo_scaler_ = StandardScaler()
        X = self.cosmo_scaler_.fit_transform(cosmos)

        self.lin_compressor_ = PerZFieldCompressor(self.n_z, self.var_threshold, self.max_components_per_z).fit(L_lin, label="P_lin")
        self.boost_compressor_ = PerZFieldCompressor(self.n_z, self.var_threshold, self.max_components_per_z).fit(L_boost, label="boost")

        C_lin = self.lin_compressor_.transform_to_coeffs(L_lin)
        C_boost = self.boost_compressor_.transform_to_coeffs(L_boost)
        C = np.concatenate([C_lin, C_boost], axis=1)

        self.coeff_scaler_ = StandardScaler()
        Y = self.coeff_scaler_.fit_transform(C)
        return X.astype(np.float32), Y.astype(np.float32)

    def transform_cosmo(self, cosmos):
        return self.cosmo_scaler_.transform(cosmos).astype(np.float32)

    def transform_targets(self, pk_nl, pk_lin, cosmos):
        L_lin, L_boost = compute_direct_targets(pk_nl, pk_lin, cosmos, self.ks_, self.zs_,
                                                  clip=self.target_clip, verbose=False)
        C_lin = self.lin_compressor_.transform_to_coeffs(L_lin)
        C_boost = self.boost_compressor_.transform_to_coeffs(L_boost)
        C = np.concatenate([C_lin, C_boost], axis=1)
        return self.coeff_scaler_.transform(C).astype(np.float32)

    @property
    def n_lin(self):
        return self.lin_compressor_.total_components

    @property
    def n_boost(self):
        return self.boost_compressor_.total_components


# =============================================================================
# 4. Model
# =============================================================================

class Emulator(nn.Module):
    """Shared trunk, single head predicting concatenated [lin_coeffs, boost_coeffs].
    Kept for reference/backward compatibility; PerZHeadEmulator (below) is the
    default now."""

    def __init__(self, n_in, n_out, hidden=(256, 256, 256)):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PerZHeadEmulator(nn.Module):
    """
    Shared trunk (cosmology -> shared representation) feeding into SEPARATE
    small heads, one per redshift slice per field (n_z lin heads + n_z boost
    heads). Reproduces the exact same output ordering as the flat Emulator
    ([C_lin for all z, concatenated] then [C_boost for all z, concatenated])
    so nothing downstream (Preprocessor, target ordering, PCA reconstruction)
    needs to change -- only how the mapping from cosmology to those
    coefficients is parameterized internally.

    Motivation: different redshift slices have very different difficulty --
    e.g. low-z/high-k bins carry a sharp, Tagn-driven, several-orders-of-
    magnitude AGN-feedback suppression signal that the halofit baseline
    (zero Tagn dependence) knows nothing about, while high-z slices are
    close to trivial. A single shared output layer has to serve both with
    the same final weights; giving each z its own (small) head lets it
    specialize without being diluted by, or fighting for capacity with, the
    easier slices -- while the shared trunk still lets the network learn a
    common cosmology representation efficiently instead of paying a full
    52x parameter cost.
    """

    def __init__(self, n_in, n_comp_lin_per_z, n_comp_boost_per_z,
                 trunk_hidden=(1024, 1024, 1024), head_hidden=64):
        super().__init__()
        trunk_layers = []
        d = n_in
        for h in trunk_hidden:
            trunk_layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = d

        def make_head(n_comp):
            if head_hidden and head_hidden > 0:
                return nn.Sequential(
                    nn.Linear(trunk_out, head_hidden), nn.SiLU(),
                    nn.Linear(head_hidden, int(n_comp)),
                )
            return nn.Sequential(nn.Linear(trunk_out, int(n_comp)))

        self.lin_heads = nn.ModuleList([make_head(nc) for nc in n_comp_lin_per_z])
        self.boost_heads = nn.ModuleList([make_head(nc) for nc in n_comp_boost_per_z])

    def forward(self, x):
        h = self.trunk(x)
        lin_out = torch.cat([head(h) for head in self.lin_heads], dim=1)
        boost_out = torch.cat([head(h) for head in self.boost_heads], dim=1)
        return torch.cat([lin_out, boost_out], dim=1)


def train_emulator(model, X_train, Y_train, X_val, Y_val, epochs=1000, batch_size=512,
                    lr=1e-3, patience=40, device="cpu", loss_trim_frac=0.1, cosmos_train=None):
    """
    cosmos_train (optional, raw (n,9) cosmology array aligned with X_train):
    if given, tracks how often each training sample gets EXCLUDED by the
    loss trim at each epoch, and at the end reports whether frequently-
    excluded samples have systematically different parameter values than
    rarely-excluded ones. This directly tests whether the trimming is
    filtering out genuinely-corrupted/unphysical outliers (its intended
    purpose) versus systematically starving legitimately-hard-but-real
    physics (e.g. strong AGN-feedback suppression) of any gradient signal,
    since those samples would also tend to have the highest per-sample loss
    and would otherwise look identical to "an outlier" to the trim.
    """
    model.to(device)
    n_train = X_train.shape[0]
    idx_arr = torch.arange(n_train)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), idx_arr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)
    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    def per_sample_mse(pred, target):
        return ((pred - target) ** 2).mean(dim=1)

    def trimmed_mean_and_excluded(per_sample_losses, trim_frac=loss_trim_frac):
        n = per_sample_losses.shape[0]
        if not trim_frac:
            return per_sample_losses.mean(), torch.zeros(n, dtype=torch.bool)
        n_keep = max(1, int(np.ceil(n * (1.0 - trim_frac))))
        sorted_vals, sort_idx = torch.sort(per_sample_losses)
        excluded_mask = torch.zeros(n, dtype=torch.bool)
        excluded_mask[sort_idx[n_keep:]] = True
        return sorted_vals[:n_keep].mean(), excluded_mask

    exclude_count = np.zeros(n_train, dtype=np.int64)
    n_epochs_run = 0

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "train_loss_full": [], "val_loss_full": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        for xb, yb, ib in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss, excluded_mask = trimmed_mean_and_excluded(per_sample_mse(pred, yb))
            loss.backward()
            opt.step()
            exclude_count[ib.numpy()[excluded_mask.numpy()]] += 1
        n_epochs_run += 1

        model.eval()
        with torch.no_grad():
            per_sample_val = per_sample_mse(model(Xv), Yv)
            per_sample_train = per_sample_mse(model(Xt), Yt)
            val_loss, _ = trimmed_mean_and_excluded(per_sample_val)
            train_loss_full, _ = trimmed_mean_and_excluded(per_sample_train)
            val_loss, train_loss_full = val_loss.item(), train_loss_full.item()
            val_loss_full_untrimmed = per_sample_val.mean().item()
            train_loss_full_untrimmed = per_sample_train.mean().item()
        sched.step(val_loss)
        history["train_loss"].append(train_loss_full)
        history["val_loss"].append(val_loss)
        history["train_loss_full"].append(train_loss_full_untrimmed)
        history["val_loss_full"].append(val_loss_full_untrimmed)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or patience_ctr == 0:
            print(f"epoch {epoch:4d}  train_mse(trimmed)={train_loss_full:.6f}  "
                  f"val_mse(trimmed)={val_loss:.6f}  best={best_val:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch} (best val_mse={best_val:.6f})")
            break

    model.load_state_dict(best_state)
    history = {k: np.array(v) for k, v in history.items()}

    if loss_trim_frac and cosmos_train is not None and n_epochs_run > 0:
        _report_trim_bias(exclude_count, n_epochs_run, cosmos_train)

    return model, history


def _report_trim_bias(exclude_count, n_epochs_run, cosmos_train):
    """Compares parameter distributions between the most-frequently-trimmed
    and least-frequently-trimmed training samples. A large, consistent gap
    (especially in Tagn) means the loss trim is systematically excluding a
    specific physical regime from ever contributing gradient signal, not
    just randomly dropping noisy samples epoch to epoch.

    Also suggests a concrete cutoff: if you'd rather EXCLUDE the hard
    region entirely (from the whole pipeline, including PCA fitting -- see
    apply_param_range_cut) than rely on per-epoch trimming to route around
    it, the printed percentile of the worst-parameter among frequently-
    excluded samples is a reasonable starting point for --cut_max_val (or
    --cut_min_val)."""
    frac_excluded = exclude_count / n_epochs_run
    n = len(frac_excluded)
    n_tail = max(1, int(0.1 * n))
    most_idx = np.argsort(-frac_excluded)[:n_tail]
    least_idx = np.argsort(frac_excluded)[:n_tail]

    print(f"\n[loss trim bias check] fraction of epochs each sample was excluded: "
          f"mean={frac_excluded.mean():.3f}, samples excluded >50% of epochs: "
          f"{(frac_excluded > 0.5).sum()}/{n}")
    print(f"    parameter means, most-excluded {n_tail} samples  vs  least-excluded {n_tail} samples:")
    worst_gap, worst_name, worst_idx = 0.0, None, None
    for j, name in enumerate(PARAM_NAMES):
        most_mean, least_mean = cosmos_train[most_idx, j].mean(), cosmos_train[least_idx, j].mean()
        std = cosmos_train[:, j].std()
        gap_in_std = (most_mean - least_mean) / std if std > 0 else 0.0
        flag = "  <-- large systematic gap" if abs(gap_in_std) > 1.0 else ""
        print(f"      {name:>10}: most-excluded={most_mean:10.4g}   least-excluded={least_mean:10.4g}   "
              f"gap={gap_in_std:+.2f} std{flag}")
        if abs(gap_in_std) > abs(worst_gap):
            worst_gap, worst_name, worst_idx = gap_in_std, name, j

    print("    A large gap means the trim is systematically starving a specific physical regime of "
          "gradient signal every epoch, not filtering random/occasional outliers.")
    if worst_name is not None and abs(worst_gap) > 1.0:
        vals = cosmos_train[:, worst_idx]
        if worst_gap > 0:
            suggested = float(np.percentile(vals, 85))
            print(f"    If you'd rather EXCLUDE this region from the whole pipeline (not just NN "
                  f"training) to sharpen bulk precision: try --cut_param {worst_name} "
                  f"--cut_max_val {suggested:.4g} (~85th percentile of {worst_name} in the training set).")
        else:
            suggested = float(np.percentile(vals, 15))
            print(f"    If you'd rather EXCLUDE this region from the whole pipeline (not just NN "
                  f"training) to sharpen bulk precision: try --cut_param {worst_name} "
                  f"--cut_min_val {suggested:.4g} (~15th percentile of {worst_name} in the training set).")
    print()


# =============================================================================
# 5. Fast inference path
# =============================================================================

class NumpyEmulator:
    """Extracts trained torch weights + fused preprocessing maps into plain
    NumPy arrays. No torch/sklearn dependency at deploy time -- just a chain
    of small matrix multiplies (including the per-z-block PCA reconstruction).

    predict() now returns BOTH P_lin and P_nl: P_lin is no longer emulated
    directly, it's the numerical E&H98+growth-ODE baseline (see
    numerical_linear.py) times the NN-predicted correction, exactly
    mirroring how P_nl was already built from the halofit baseline times a
    correction -- both fields now follow the same "physics baseline plus
    small, PCA-compressible residual" pattern."""

    def __init__(self, model, prep: Preprocessor):
        self.is_per_z_head = isinstance(model, PerZHeadEmulator)

        if self.is_per_z_head:
            self.trunk_weights = self._extract_seq_weights(model.trunk)
            self.lin_head_weights = [self._extract_seq_weights(h) for h in model.lin_heads]
            self.boost_head_weights = [self._extract_seq_weights(h) for h in model.boost_heads]
        else:
            self.weights = self._extract_seq_weights(model.net)

        self.cosmo_mean = prep.cosmo_scaler_.mean_.astype(np.float32)
        self.cosmo_std = prep.cosmo_scaler_.scale_.astype(np.float32)
        self.coeff_mean = prep.coeff_scaler_.mean_.astype(np.float32)
        self.coeff_std = prep.coeff_scaler_.scale_.astype(np.float32)

        self.n_lin = prep.n_lin
        self.n_boost = prep.n_boost
        self.n_z = prep.n_z
        self.n_k = prep.n_k
        self.ks = prep.ks_.astype(np.float64)
        self.zs = prep.zs_.astype(np.float64)

        self.n_comp_lin_per_z = prep.lin_compressor_.n_components_per_z
        self.n_comp_boost_per_z = prep.boost_compressor_.n_components_per_z
        self.W_lin, self.b_lin = prep.lin_compressor_.stacked_W_b()      # W: (n_lin, n_k), b: (n_z, n_k)
        self.W_boost, self.b_boost = prep.boost_compressor_.stacked_W_b()
        self.W_lin, self.b_lin = self.W_lin.astype(np.float32), self.b_lin.astype(np.float32)
        self.W_boost, self.b_boost = self.W_boost.astype(np.float32), self.b_boost.astype(np.float32)

        # Must match what the network was actually trained on (Preprocessor's
        # target_clip): the training targets themselves were clipped to this
        # range before the network ever saw them, so it was never trained to
        # produce anything beyond it. A much looser inference-time clip (e.g.
        # a generic +-30) lets an ordinary, moderate misprediction for a
        # single bin turn into a multi-order-of-magnitude blowup once
        # exponentiated (10**L) -- exactly the mechanism behind isolated
        # catastrophic-outlier bins coexisting with an otherwise reasonable
        # median error. Clip to what training actually bounded instead.
        self.target_clip = float(prep.target_clip)

    @staticmethod
    def _extract_seq_weights(seq_module):
        weights = []
        for layer in seq_module:
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy().T.astype(np.float32)
                b = layer.bias.detach().cpu().numpy().astype(np.float32)
                weights.append(("linear", W, b))
            elif isinstance(layer, nn.SiLU):
                weights.append(("silu", None, None))
        return weights

    @staticmethod
    def _silu(x):
        x = np.clip(x, -60.0, 60.0)
        return x / (1.0 + np.exp(-x))

    @classmethod
    def _apply_seq(cls, X, weights):
        h = X
        for kind, W, b in weights:
            if kind == "linear":
                h = h @ W + b
            else:
                h = cls._silu(h)
        return h

    def _mlp_forward(self, X):
        if not self.is_per_z_head:
            return self._apply_seq(X, self.weights)
        trunk_out = self._apply_seq(X, self.trunk_weights)
        lin_out = np.concatenate([self._apply_seq(trunk_out, w) for w in self.lin_head_weights], axis=1)
        boost_out = np.concatenate([self._apply_seq(trunk_out, w) for w in self.boost_head_weights], axis=1)
        return np.concatenate([lin_out, boost_out], axis=1)

    @staticmethod
    def _reconstruct_per_z(coeffs, n_comp_per_z, W_stacked, b_stacked, n_z, n_k):
        """coeffs: (n, total_components) -> (n, n_z, n_k), using each
        z-slice's own block of PCA components (W_stacked/b_stacked are the
        vertical/row stack of all per-slice fused inverse-transforms)."""
        n = coeffs.shape[0]
        out = np.empty((n, n_z, n_k), dtype=np.float32)
        idx = 0
        for zi in range(n_z):
            nc = int(n_comp_per_z[zi])
            out[:, zi, :] = coeffs[:, idx:idx + nc] @ W_stacked[idx:idx + nc, :] + b_stacked[zi, :]
            idx += nc
        return out

    def predict(self, cosmos: np.ndarray, warn_extrapolation: bool = True, return_lin: bool = False):
        """cosmos: (n, 9) raw (unstandardized) cosmological parameters ->
        returns P_nl of shape (n, n_z, n_k) [Mpc^3] (default, backward
        compatible), or (P_lin, P_nl) if return_lin=True.

        Pipeline: emulate log10(P_lin/P_lin_baseline) -> combine with the
        numerical E&H98+growth-ODE linear baseline (pure NumPy, no external
        model) to get P_lin -> run the numerical halofit baseline on THAT
        P_lin -> apply the NN-predicted boost correction on top to get
        P_nl. Both baselines are pure-physics, dependency-free NumPy
        calculations -- no separate domain of validity for either."""
        cosmos = np.atleast_2d(cosmos).astype(np.float64)
        n = cosmos.shape[0]

        if warn_extrapolation:
            z = (cosmos - self.cosmo_mean) / self.cosmo_std
            max_abs_z = np.abs(z).max(axis=1)
            for i in np.where(max_abs_z > 5)[0]:
                j = int(np.argmax(np.abs(z[i])))
                print(f"[NumpyEmulator] warning: sample {i} is {abs(z[i, j]):.1f} std devs "
                      f"from the training mean in param '{PARAM_NAMES[j]}' -- prediction is an "
                      f"extrapolation and may be unreliable.")

        Xs = (cosmos.astype(np.float32) - self.cosmo_mean) / self.cosmo_std
        np.clip(Xs, -50.0, 50.0, out=Xs)
        Y = self._mlp_forward(Xs)
        C = Y * self.coeff_std + self.coeff_mean
        C_lin, C_boost = C[:, : self.n_lin], C[:, self.n_lin :]

        L_lin_corr = self._reconstruct_per_z(C_lin, self.n_comp_lin_per_z, self.W_lin, self.b_lin, self.n_z, self.n_k)
        L_lin_corr = np.nan_to_num(L_lin_corr, nan=0.0, posinf=self.target_clip, neginf=-self.target_clip)
        np.clip(L_lin_corr, -self.target_clip, self.target_clip, out=L_lin_corr)

        pk_lin_baseline = linear_baseline_batch(
            self.ks, self.zs, cosmos, as_idx=AS_IDX, ns_idx=NS_IDX, h_idx=H0_IDX,
            ob_idx=OB_IDX, om_idx=OM_IDX, w0_idx=W0_IDX, wa_idx=WA_IDX,
        )
        pk_lin_pred = pk_lin_baseline * (10.0 ** L_lin_corr)
        pk_lin_pred = np.nan_to_num(np.clip(pk_lin_pred, EPS, 1e30), nan=EPS, posinf=1e30, neginf=EPS)

        Om = cosmos[:, OM_IDX][:, None]                      # (n, 1)
        w0 = cosmos[:, W0_IDX][:, None]
        wa = cosmos[:, WA_IDX][:, None]
        a = (1.0 / (1.0 + self.zs))[None, :]                  # (1, n_z)
        Om_b = np.broadcast_to(Om, (n, self.n_z))
        w0_b = np.broadcast_to(w0, (n, self.n_z))
        wa_b = np.broadcast_to(wa, (n, self.n_z))
        a_b = np.broadcast_to(a, (n, self.n_z))
        pk_nl_baseline, _ = halofit_baseline(self.ks, pk_lin_pred, Om_b, a_b, w0_b, wa_b)

        L_boost_corr = self._reconstruct_per_z(C_boost, self.n_comp_boost_per_z, self.W_boost, self.b_boost, self.n_z, self.n_k)
        L_boost_corr = np.nan_to_num(L_boost_corr, nan=0.0, posinf=self.target_clip, neginf=-self.target_clip)
        np.clip(L_boost_corr, -self.target_clip, self.target_clip, out=L_boost_corr)

        pk_nl_pred = pk_nl_baseline * (10.0 ** L_boost_corr)
        pk_nl_pred = np.nan_to_num(np.clip(pk_nl_pred, EPS, 1e30), nan=EPS, posinf=1e30, neginf=EPS)

        if return_lin:
            return pk_lin_pred, pk_nl_pred
        return pk_nl_pred

    def save(self, path):
        payload = dict(
            is_per_z_head=self.is_per_z_head,
            cosmo_mean=self.cosmo_mean, cosmo_std=self.cosmo_std,
            coeff_mean=self.coeff_mean, coeff_std=self.coeff_std,
            n_lin=self.n_lin, n_boost=self.n_boost, n_z=self.n_z, n_k=self.n_k,
            ks=self.ks, zs=self.zs,
            n_comp_lin_per_z=self.n_comp_lin_per_z, n_comp_boost_per_z=self.n_comp_boost_per_z,
            W_lin=self.W_lin, b_lin=self.b_lin, W_boost=self.W_boost, b_boost=self.b_boost,
            target_clip=self.target_clip,
        )
        if self.is_per_z_head:
            payload["trunk_weights"] = np.array(self.trunk_weights, dtype=object)
            payload["lin_head_weights"] = np.array(self.lin_head_weights, dtype=object)
            payload["boost_head_weights"] = np.array(self.boost_head_weights, dtype=object)
        else:
            payload["weights"] = np.array(self.weights, dtype=object)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.is_per_z_head = bool(d["is_per_z_head"]) if "is_per_z_head" in d else False
        if obj.is_per_z_head:
            obj.trunk_weights = list(d["trunk_weights"])
            obj.lin_head_weights = [list(w) for w in d["lin_head_weights"]]
            obj.boost_head_weights = [list(w) for w in d["boost_head_weights"]]
        else:
            obj.weights = list(d["weights"])
        obj.cosmo_mean, obj.cosmo_std = d["cosmo_mean"], d["cosmo_std"]
        obj.coeff_mean, obj.coeff_std = d["coeff_mean"], d["coeff_std"]
        obj.n_lin, obj.n_boost = int(d["n_lin"]), int(d["n_boost"])
        obj.n_z, obj.n_k = int(d["n_z"]), int(d["n_k"])
        obj.ks, obj.zs = d["ks"], d["zs"]
        obj.target_clip = float(d["target_clip"]) if "target_clip" in d else 15.0
        obj.n_comp_lin_per_z, obj.n_comp_boost_per_z = d["n_comp_lin_per_z"], d["n_comp_boost_per_z"]
        obj.W_lin, obj.b_lin = d["W_lin"], d["b_lin"]
        obj.W_boost, obj.b_boost = d["W_boost"], d["b_boost"]
        return obj


# =============================================================================
# 6. Evaluation
# =============================================================================

def evaluate(pk_true, pk_pred, label=""):
    valid = np.isfinite(pk_true) & (pk_true != 0) & np.isfinite(pk_pred)
    n_skipped = pk_true.size - valid.sum()
    with np.errstate(over="ignore", invalid="ignore"):
        rel_err = np.abs(pk_pred[valid] - pk_true[valid]) / np.abs(pk_true[valid])
    print(f"[{label}] relative error over full (k,z) grid"
          f"{f' (skipping {n_skipped} NaN/zero/inf bins)' if n_skipped else ''}:")
    print(f"    median = {np.median(rel_err)*100:.4f}%   <- bulk precision, this is the one that matters most")
    print(f"    68th   = {np.percentile(rel_err, 68)*100:.4f}%   (~1-sigma-equivalent, still 'typical')")
    print(f"    80th   = {np.percentile(rel_err, 80)*100:.4f}%")
    print(f"    mean   = {rel_err.mean()*100:.4f}%   (tail-dominated, not representative of bulk)")
    print(f"    95th   = {np.percentile(rel_err, 95)*100:.4f}%")
    print(f"    99th   = {np.percentile(rel_err, 99)*100:.4f}%")
    print(f"    max    = {rel_err.max()*100:.4f}%")
    return rel_err


def evaluate_by_z(pk_true, pk_pred, zs, label=""):
    """Median relative error broken out by redshift -- catches problems
    concentrated in specific z ranges that an aggregate metric would hide."""
    print(f"[{label}] median relative error by redshift:")
    for zi, zv in enumerate(zs):
        t, p = pk_true[:, zi, :], pk_pred[:, zi, :]
        valid = np.isfinite(t) & (t != 0) & np.isfinite(p)
        if not valid.any():
            continue
        err = np.abs(p[valid] - t[valid]) / np.abs(t[valid])
        print(f"    z={zv:7.2f}  median={100*np.median(err):7.3f}%  95th={100*np.percentile(err,95):8.3f}%")


def diagnose_worst_errors(pk_true, pk_pred, ks, zs, cosmos=None, n_worst=15, label=""):
    """
    Locates the individual (sample, z, k) bins with the largest relative
    error and prints their true/predicted values alongside the overall
    scale of pk_true, so it's possible to tell apart two very different
    failure modes that both show up as "huge relative error":
        (a) a genuine large ABSOLUTE misprediction, or
        (b) pk_true being extremely close to zero (e.g. strong AGN-feedback
            suppression at high k), where relative error is fundamentally
            unstable -- a tiny, unremarkable absolute miss becomes an
            astronomical percentage purely from dividing by a near-zero
            denominator, with no actual modeling problem involved.
    If the worst offenders all have |pk_true| many orders of magnitude
    below the median |pk_true| (see the printed percentile-rank column),
    that's case (b): the fix is to stop trusting/reporting relative error
    on those bins, not to keep chasing model accuracy that was never really
    achievable there.

    If cosmos is given, also prints each worst bin's own cosmology (not an
    aggregate-loss-based inference like the loss-trim-bias check, which
    averages over all ~2000+ output coefficients per sample and can dilute
    a failure concentrated in a handful of bins into invisibility) --
    a direct, per-bin look at which parameter values actually produced the
    worst offenders.
    """
    n = pk_true.shape[0]
    valid = np.isfinite(pk_true) & (pk_true != 0) & np.isfinite(pk_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = np.where(valid, np.abs(pk_pred - pk_true) / np.abs(pk_true), -1.0)

    flat_idx = np.argsort(rel_err, axis=None)[::-1][:n_worst]
    idx_sample, idx_z, idx_k = np.unravel_index(flat_idx, pk_true.shape)

    all_true_abs = np.abs(pk_true[valid])
    scale_pctile = (np.searchsorted(np.sort(all_true_abs), np.abs(pk_true[idx_sample, idx_z, idx_k]))
                     / all_true_abs.size * 100.0)

    print(f"[{label}] worst {n_worst} individual (sample,z,k) errors:")
    if cosmos is None:
        print(f"    {'rel err':>12}  {'pk_true':>12}  {'pk_pred':>12}  {'z':>7}  {'k [1/Mpc]':>10}  "
              f"{'|pk_true| percentile rank':>26}")
        for si, zi, ki, pr in zip(idx_sample, idx_z, idx_k, scale_pctile):
            t, p = pk_true[si, zi, ki], pk_pred[si, zi, ki]
            print(f"    {100*rel_err[si,zi,ki]:11.2f}%  {t:12.4e}  {p:12.4e}  {zs[zi]:7.2f}  "
                  f"{ks[ki]:10.4g}  {pr:25.4f}%")
    else:
        param_hdr = "  ".join(f"{name:>8}" for name in PARAM_NAMES)
        print(f"    {'rel err':>12}  {'pk_true':>12}  {'pk_pred':>12}  {'z':>7}  {'k [1/Mpc]':>10}  {param_hdr}")
        for si, zi, ki, pr in zip(idx_sample, idx_z, idx_k, scale_pctile):
            t, p = pk_true[si, zi, ki], pk_pred[si, zi, ki]
            param_str = "  ".join(f"{v:8.4g}" for v in cosmos[si])
            print(f"    {100*rel_err[si,zi,ki]:11.2f}%  {t:12.4e}  {p:12.4e}  {zs[zi]:7.2f}  "
                  f"{ks[ki]:10.4g}  {param_str}")

        # Direct comparison: mean parameter values among the SAMPLES that
        # produced these worst individual bins, vs the mean over ALL test
        # samples -- a much sharper signal than the aggregate per-sample
        # loss used by the loss-trim-bias check, since it's restricted to
        # exactly the samples implicated in the worst bins rather than
        # diluted across everything else that sample also predicts fine.
        worst_sample_idx = np.unique(idx_sample)
        print(f"    ({len(worst_sample_idx)} distinct samples produced these {n_worst} worst bins) "
              f"parameter means, worst-bin samples vs all test samples:")
        for j, name in enumerate(PARAM_NAMES):
            worst_mean = cosmos[worst_sample_idx, j].mean()
            all_mean = cosmos[:, j].mean()
            std = cosmos[:, j].std()
            gap = (worst_mean - all_mean) / std if std > 0 else 0.0
            flag = "  <-- likely driver" if abs(gap) > 1.0 else ""
            print(f"      {name:>10}: worst-bin samples={worst_mean:10.4g}   "
                  f"all test samples={all_mean:10.4g}   gap={gap:+.2f} std{flag}")

    low_pctile = scale_pctile[scale_pctile < 1.0]
    if len(low_pctile) >= n_worst * 0.6:
        print(f"    -> {len(low_pctile)}/{n_worst} of the worst offenders have |pk_true| in the "
              f"bottom 1% of the whole dataset's scale: this looks like the near-zero-denominator "
              f"pathology (relative error is unstable there regardless of model quality), not a "
              f"modeling problem specifically.")


def evaluate_pca_reconstruction(pk_nl, pk_lin, cosmos, ks, zs, prep, label=""):
    """
    Isolates PCA-compression error from prediction error: takes the TRUE
    data, encodes it through the ALREADY-FITTED PCA compressors, decodes it
    straight back (no neural network, no PCE, no prediction of any kind
    involved), and compares the round-tripped reconstruction to the
    original. This answers "how much accuracy is lost purely to choosing
    --max_components_per_z, independent of how well anything learns to
    predict those coefficients from cosmology" -- i.e. a ceiling on
    achievable accuracy from this compression scheme alone.

    Reports mean percent error (MPE, signed -- shows systematic bias) and
    mean absolute percent error (MAPE, shows overall magnitude) for both
    P_lin (baseline from the numerical E&H98+growth-ODE calculation, so
    this isolates PCA error on the L_lin_corr residual only) and P_nl
    (baseline from the TRUE P_lin via numerical halofit).
    """
    L_lin_corr, L_boost_corr, pk_lin_baseline_true, pk_nl_hf_true = compute_direct_targets(
        pk_nl, pk_lin, cosmos, ks, zs, clip=prep.target_clip, verbose=False, return_baseline=True
    )

    C_lin = prep.lin_compressor_.transform_to_coeffs(L_lin_corr)
    C_boost = prep.boost_compressor_.transform_to_coeffs(L_boost_corr)
    L_lin_corr_recon = prep.lin_compressor_.inverse_from_coeffs(C_lin)
    L_boost_recon = prep.boost_compressor_.inverse_from_coeffs(C_boost)

    pk_lin_recon = pk_lin_baseline_true * (10.0 ** L_lin_corr_recon)
    pk_nl_recon = pk_nl_hf_true * (10.0 ** L_boost_recon)

    def _mpe_mape(true, recon, name):
        valid = np.isfinite(true) & (true != 0) & np.isfinite(recon)
        pe = (recon[valid] - true[valid]) / true[valid] * 100.0
        print(f"    [{name}] MPE (signed, mean) = {pe.mean():+.4f}%   "
              f"MAPE (mean |error|) = {np.abs(pe).mean():.4f}%   "
              f"({int((~valid).sum())} bins skipped as NaN/zero)")

    print(f"[{label}] PCA round-trip reconstruction error (no prediction, "
          f"true data encoded+decoded through the fitted compressors only):")
    _mpe_mape(pk_lin, pk_lin_recon, "P_lin (baseline from E&H98+growth-ODE, isolates PCA error only)")
    _mpe_mape(pk_nl, pk_nl_recon, "P_nl (baseline from TRUE P_lin, so this isolates PCA error only)")

    # Aggregate MAPE averages over all 500 k x 52 z bins, which can hide a
    # severe failure concentrated in a small corner (diluted by ~26000
    # mostly-fine bins). Report the same breakdown restricted to the last
    # few k-bins (highest k, where AGN-feedback suppression is strongest and
    # the halofit baseline -- which has zero Tagn dependence -- is most
    # wrong) at low z (where that suppression is typically strongest) --
    # this isolates whether the PCA basis ITSELF already loses fidelity
    # there (a compression problem, fixable by more components in those
    # slices) versus reconstructing fine (meaning any failure downstream is
    # the network failing to predict the right coefficients, not a ceiling
    # PCA imposed).
    n_z, n_k = pk_nl.shape[1], pk_nl.shape[2]
    k_tail = slice(max(0, n_k - 10), n_k)
    z_low = slice(0, min(10, n_z))
    print(f"[{label}] SAME check, restricted to the highest-k bins (k={ks[k_tail][0]:.3g}-{ks[-1]:.3g}) "
          f"at low z (z={zs[z_low][0]:.2f}-{zs[z_low][-1]:.2f}) -- the corner where AGN feedback "
          f"suppression is strongest and the halofit baseline has zero Tagn dependence:")
    _mpe_mape(pk_lin[:, z_low, k_tail], pk_lin_recon[:, z_low, k_tail], "P_lin (high-k, low-z corner)")
    _mpe_mape(pk_nl[:, z_low, k_tail], pk_nl_recon[:, z_low, k_tail], "P_nl (high-k, low-z corner)")


def benchmark(np_emu: NumpyEmulator, X_test_raw):
    x1 = X_test_raw[:1]
    t0 = time.perf_counter()
    for _ in range(200):
        np_emu.predict(x1, warn_extrapolation=False)
    t_single = (time.perf_counter() - t0) / 200

    t0 = time.perf_counter()
    _ = np_emu.predict(X_test_raw, warn_extrapolation=False)
    t_batch = time.perf_counter() - t0

    print("\n[Speed benchmark]")
    print(f"  1 sample            : {t_single*1e6:.2f} us/call")
    print(f"  {X_test_raw.shape[0]} samples batch : "
          f"{t_batch*1e3:.3f} ms total ({t_batch/X_test_raw.shape[0]*1e6:.2f} us/sample)")


# =============================================================================
# 7. main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk_nl", default="pk_nls.npy",
                     help="path OR glob pattern with a single '*' for the batch id, e.g. "
                          "'.../train_..._*_..._pknonlin.npy'")
    ap.add_argument("--pk_lin", default="pk_lins.npy",
                     help="path OR glob pattern with a single '*' for the batch id, e.g. "
                          "'.../train_..._*_..._pklin.npy'")
    ap.add_argument("--cosmos", default="cosmos.npy",
                     help="path OR glob pattern with a single '*' for the batch id, e.g. "
                          "'.../train_..._*_..._tfree_mnufree.npy'")
    ap.add_argument("--n_batches", type=int, default=None,
                     help="keep only batches with numeric id < this value (NOT a count of how many "
                          "batches to load, and not position in the sorted list) -- e.g. "
                          "--n_batches 1000 loads every available batch numbered 0-999 and reliably "
                          "excludes a batch literally numbered 1000 (e.g. a reserved test batch), "
                          "regardless of how many/which lower-numbered batches actually exist. "
                          "Default: load every available batch.")
    ap.add_argument("--outdir", default="./emulator_out")
    ap.add_argument("--var_threshold", type=float, default=0.9999)
    ap.add_argument("--max_components_per_z", type=int, default=60,
                     help="PCA component cap PER REDSHIFT SLICE (not global). Each of the 52 "
                          "slices is compressed separately -- a single slice's k-dependence "
                          "should need far fewer components than the old global (n_z*n_k) PCA.")
    ap.add_argument("--target_clip", type=float, default=15.0,
                     help="clip |log10(P_lin/baseline)| and |log10(P_nl/baseline)| to this value "
                          "before standardizing/PCA, guarding against a few pathological data "
                          "points skewing the fit for the rest of the dataset.")
    ap.add_argument("--max_nan_frac", type=float, default=0.02,
                     help="drop samples whose fraction of NaN entries in pk_nl/pk_lin exceeds "
                          "this (default 2%%) -- treated as failed/unstable simulation runs, "
                          "not trustworthy even at their non-NaN bins. Set to 1.0 or higher to "
                          "keep all samples (still handled via masked projection downstream, "
                          "no matter how NaN-heavy).")
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024],
                     help="shared-trunk hidden layer sizes (used as the trunk in the default "
                          "per-z-head architecture, or as the sole network if --no_per_z_heads).")
    ap.add_argument("--head_hidden", type=int, default=64,
                     help="hidden layer width of each per-redshift-per-field head (0 = linear "
                          "head, no hidden layer). Only used with the default per-z-head "
                          "architecture.")
    ap.add_argument("--per_z_heads", dest="per_z_heads", action="store_true", default=True,
                     help="use a shared trunk + separate per-redshift heads (default: on) -- "
                          "gives each redshift dedicated final-layer capacity instead of one "
                          "shared output layer serving all 52 slices.")
    ap.add_argument("--no_per_z_heads", dest="per_z_heads", action="store_false",
                     help="use a single flat shared network instead (the older architecture).")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--loss_trim_frac", type=float, default=0.1,
                     help="fraction of highest-loss samples to drop from the loss at every "
                          "epoch, so a few extreme/unphysical outlier samples don't dominate "
                          "training at the expense of the bulk. Set to 0 to disable.")
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--triangle_cut", dest="triangle_cut", action="store_true", default=True,
                     help="exclude a disfavored corner before splitting into train/val/test "
                          "(default: on; corner set by --cut_ob_anchor*/--cut_h_anchor*).")
    ap.add_argument("--no_triangle_cut", dest="triangle_cut", action="store_false",
                     help="disable the triangle cut.")
    ap.add_argument("--cut_ob_anchor1", type=float, default=0.05, help="Omega_b at anchor 1")
    ap.add_argument("--cut_h_anchor1", type=float, default=88, help="H0 at anchor 1")
    ap.add_argument("--cut_ob_anchor2", type=float, default=0.07, help="Omega_b at anchor 2")
    ap.add_argument("--cut_h_anchor2", type=float, default=75, help="H0 at anchor 2")
    ap.add_argument("--cut_param", type=str, default=None, choices=PARAM_NAMES,
                     help="optionally exclude a range of one parameter from the ENTIRE pipeline "
                          "(including PCA fitting, not just NN training loss like --loss_trim_frac "
                          "does) -- e.g. --cut_param Tagn --cut_max_val 8.0 to drop the hardest, "
                          "most-AGN-suppressed cosmologies entirely and let PCA/the network "
                          "concentrate fully on bulk precision for the rest. The trim-bias report "
                          "printed after training suggests a concrete threshold if this looks useful.")
    ap.add_argument("--cut_min_val", type=float, default=None, help="keep samples with --cut_param >= this")
    ap.add_argument("--cut_max_val", type=float, default=None, help="keep samples with --cut_param <= this")
    ap.add_argument("--outlier_cut", dest="outlier_cut", action="store_true", default=True,
                     help="drop samples that are ISOLATED in the 9-D parameter space (few nearby "
                          "training neighbors), via k-nearest-neighbor distance in normalized "
                          "parameter space -- default: on. See --outlier_k_neighbors / "
                          "--outlier_cut_percentile.")
    ap.add_argument("--no_outlier_cut", dest="outlier_cut", action="store_false",
                     help="disable the isolation cut.")
    ap.add_argument("--outlier_k_neighbors", type=int, default=10,
                     help="how many nearest neighbors define 'local density' for the isolation cut.")
    ap.add_argument("--outlier_cut_percentile", type=float, default=98.0,
                     help="drop the most-isolated (100 - this) percent of samples by k-NN distance "
                          "(default 98 -> drops the most-isolated 2%%). Lower this (e.g. 90-95) for "
                          "a much more aggressive bulk-precision focus, given rare/isolated "
                          "cosmologies are explicitly OK to sacrifice.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pk_nl, pk_lin, cosmos, ks, zs = load_data(args.cosmos, args.pk_lin, args.pk_nl, n_batches=args.n_batches)
    n_z, n_k = pk_nl.shape[1], pk_nl.shape[2]

    print("NaN pattern before any cuts:")
    diagnose_nan_pattern(pk_nl, pk_lin, zs)
    pk_nl, pk_lin, cosmos = apply_nan_quality_cut(pk_nl, pk_lin, cosmos, max_nan_frac=args.max_nan_frac)

    if args.triangle_cut:
        pk_nl, pk_lin, cosmos = apply_triangle_cut(
            pk_nl, pk_lin, cosmos,
            x_idx=PARAM_NAMES.index("Omega_b"), y_idx=PARAM_NAMES.index("H0"),
            anchor1=(args.cut_ob_anchor1, args.cut_h_anchor1),
            anchor2=(args.cut_ob_anchor2, args.cut_h_anchor2),
            label="high Omega_b + high H0",
        )

    if args.cut_param is not None and (args.cut_min_val is not None or args.cut_max_val is not None):
        pk_nl, pk_lin, cosmos = apply_param_range_cut(
            pk_nl, pk_lin, cosmos, param_idx=PARAM_NAMES.index(args.cut_param),
            min_val=args.cut_min_val, max_val=args.cut_max_val, label="bulk-precision focus",
        )

    if args.outlier_cut:
        pk_nl, pk_lin, cosmos = apply_multivariate_outlier_cut(
            pk_nl, pk_lin, cosmos, k_neighbors=args.outlier_k_neighbors,
            cut_percentile=args.outlier_cut_percentile, label="bulk-precision focus",
        )

    idx = np.arange(cosmos.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=args.val_frac / (1 - args.test_frac), random_state=args.seed
    )
    print(f"Split sizes -> train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

    prep = Preprocessor(n_z=n_z, n_k=n_k, var_threshold=args.var_threshold,
                         max_components_per_z=args.max_components_per_z, target_clip=args.target_clip)
    X_train, Y_train = prep.fit_transform(pk_nl[idx_train], pk_lin[idx_train], cosmos[idx_train], ks, zs)
    X_val = prep.transform_cosmo(cosmos[idx_val])
    Y_val = prep.transform_targets(pk_nl[idx_val], pk_lin[idx_val], cosmos[idx_val])
    X_test = prep.transform_cosmo(cosmos[idx_test])

    evaluate_pca_reconstruction(pk_nl[idx_test], pk_lin[idx_test], cosmos[idx_test], ks, zs, prep,
                                 label="test set, PCA-only (pre-training)")

    n_out = prep.n_lin + prep.n_boost
    print(f"Total network output size: {n_out} ({prep.n_lin} lin + {prep.n_boost} boost, "
          f"vs {n_z*n_k} raw grid points per field)")

    if args.per_z_heads:
        model = PerZHeadEmulator(n_in=cosmos.shape[1],
                                  n_comp_lin_per_z=prep.lin_compressor_.n_components_per_z,
                                  n_comp_boost_per_z=prep.boost_compressor_.n_components_per_z,
                                  trunk_hidden=tuple(args.hidden), head_hidden=args.head_hidden)
    else:
        model = Emulator(n_in=cosmos.shape[1], n_out=n_out, hidden=tuple(args.hidden))
    print(model)

    model, history = train_emulator(model, X_train, Y_train, X_val, Y_val,
                                     epochs=args.epochs, batch_size=args.batch_size,
                                     lr=args.lr, patience=args.patience, device=device,
                                     loss_trim_frac=args.loss_trim_frac,
                                     cosmos_train=cosmos[idx_train])
    np.savez(outdir / "history.npz", **history)

    np_emu = NumpyEmulator(model, prep)
    pk_lin_pred_test, pk_nl_pred_test = np_emu.predict(cosmos[idx_test], warn_extrapolation=False, return_lin=True)

    evaluate(pk_lin[idx_test], pk_lin_pred_test, label="test set, P_lin")
    evaluate_by_z(pk_lin[idx_test], pk_lin_pred_test, zs, label="test set, P_lin")

    evaluate(pk_nl[idx_test], pk_nl_pred_test, label="test set, P_nl")
    evaluate_by_z(pk_nl[idx_test], pk_nl_pred_test, zs, label="test set, P_nl")
    diagnose_worst_errors(pk_nl[idx_test], pk_nl_pred_test, ks, zs, cosmos=cosmos[idx_test], label="test set, P_nl")

    np_emu.save(outdir / "numpy_emulator.npz")
    torch.save(model.state_dict(), outdir / "torch_model.pt")
    with open(outdir / "config.json", "w") as f:
        json.dump({"hidden": args.hidden, "n_lin": prep.n_lin, "n_boost": prep.n_boost,
                    "n_z": n_z, "n_k": n_k, "param_names": PARAM_NAMES}, f, indent=2)
    print(f"\nSaved deployable NumPy emulator to {outdir/'numpy_emulator.npz'}")

    benchmark(np_emu, cosmos[idx_test].astype(np.float32))

    print("\nExample usage after loading:")
    print("    from pk_emulator import NumpyEmulator")
    print("    emu = NumpyEmulator.load('numpy_emulator.npz')")
    print("    pk_nl_pred = emu.predict(cosmo_params)                        # P_nl only")
    print("    pk_lin_pred, pk_nl_pred = emu.predict(cosmo_params, return_lin=True)  # both")


if __name__ == "__main__":
    main()