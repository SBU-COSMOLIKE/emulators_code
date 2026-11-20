"""
Fast Matter Power Spectrum Emulator for Cobaya

This module provides a neural network-based emulator for cosmological matter
power spectra within the Cobaya framework. It uses a local emulmps module
that handles all model loading and prediction internally.

The emulator predicts matter power spectra P(k,z) from cosmological parameters.

    P_lin = exp(NN_lin)   * P_lin_syren          (linear emulator)
    B     = exp(NN_boost) * B_syren              (boost emulator, dimensionless)
    P_nl  = B * P_lin                            (P_lin from the LINEAR EMULATOR)

B_syren is the syren-halofit boost built from the syren linear spectrum, exactly
as in the boost training target log(B_CAMB / B_syren).  The nonlinear spectrum
is always assembled from the emulated linear spectrum, never from syren's.

Currently supports LCDM, wCDM, and w0waCDM cosmologies.

Author: Victoria Lloyd & V. Miranda
Date: 2025
"""

import os, sys, contextlib

# Must be set before TensorFlow is imported anywhere; assign, don't setdefault
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hides I/W/E C++ logs (cpu_feature_guard, TF-TRT)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # hides the oneDNN notice

@contextlib.contextmanager
def _mute_stderr_fd():
    """Silence C/C++ writes to stderr (fd 2), e.g. XLA's cuFFT/cuDNN/cuBLAS spam."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)

with _mute_stderr_fd():
    import tensorflow as tf
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass
        
import numpy as np
from scipy.interpolate import RectBivariateSpline
from cobaya.theory import Theory
from typing import Mapping, Tuple, Optional
from cobaya.typing import empty_dict, InfoDict
from cobaya.log import LoggedError, get_logger
from pathlib import Path
import logging

import tensorflow as tf

#### vic add

def _get_nl_pcs(emu, params_9):
    """
    Replicates PkEmulator._predict_nl_fracs_all_z up to the point where it
    has pcs_z, but returns the raw per-z PCA coefficients instead of the
    fully reconstructed log(boost) grid.

    Shape (emu.N_ZS, emu._nl_n_pcs).
    """
    buf = emu._params_buf
    buf[0, :] = emu._fill_params(params_9)

    x_norm    = emu._norm_params_for_scaler(buf, emu._nl_param_scaler)
    net_input = emu._make_nl_network_input(x_norm)
    t_comps_norm = emu._nl_compiled(
        tf.constant(net_input, dtype=tf.float32)
    ).numpy()

    if emu._nl_t_comp_scaler is not None:
        t_comps_raw = emu._nl_t_comp_scaler.inverse_transform(t_comps_norm)
    else:
        t_comps_raw = t_comps_norm

    pcs_flat = (emu._nl_t_comp_pca.inverse_transform(t_comps_raw)
                if emu._nl_t_comp_pca is not None else t_comps_raw)
    return pcs_flat.reshape(emu.N_ZS, emu._nl_n_pcs)

#### vic add end

# --- Path Management ---
def _get_project_root() -> Path:
    """Returns the root directory Path object, relative to this file."""
    return Path(__file__).resolve().parent

# --- Dependency Guards ---
ROOT = _get_project_root()
# Optional imports for nonlinear corrections
try:
    import sys; sys.path.insert(0, f"{ROOT}/emulmps_emul/symbolic_pofk")
    from symbolic_pofk.linear import As_to_sigma8, plin_emulated
    from symbolic_pofk.syrenhalofit import run_halofit, run_halofit_vec
    class symbolic_linear:
        As_to_sigma8 = As_to_sigma8
        plin_emulated = plin_emulated
    class syrenhalofit:
        run_halofit = run_halofit
        run_halofit_vec = run_halofit_vec
    SYMBOLIC_POFK_AVAILABLE = True
except ImportError as e:
    logging.error(f"ERROR: A required dependency could not be imported.")
    logging.error(f"Missing component: {e.name if hasattr(e, 'name') else 'symbolic_pofk'}")
    SYMBOLIC_POFK_AVAILABLE = False
    symbolic_linear = None
    syrenhalofit = None


# Import the local emulmps module
try:
    from .emulmps_emul import emulmps_w0wa as emulmps_emul
    from emulmps_emul import get_pks
except ImportError:
    try:
        from .emulmps_emul.emulmps_w0wa import get_pks
    except ImportError as e:
        raise ImportError(
            "Could not import emulmps module. Ensure the emulmps directory "
            "is present in the same directory as this theory file. "
            f"Error: {e}"
        )

########## Taken from cobaya/theories/cosmo/boltzmannbase.py ############
class PowerSpectrumInterpolator(RectBivariateSpline):
    r"""
    2D spline interpolation object (scipy.interpolate.RectBivariateSpline)
    to evaluate matter power spectrum as function of z and k.

    *This class is adapted from CAMB's own P(k) interpolator, by Antony Lewis;
    it's mostly interface-compatible with the original.*

    :param z: values of z for which the power spectrum was evaluated.
    :param k: values of k for which the power spectrum was evaluated.
    :param P_or_logP: Values of the power spectrum (or log-values, if logP=True).
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param logsign: if logP is True, P_or_logP is log(logsign*Pk)
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(self, z, k, P_or_logP, extrap_kmin=None, extrap_kmax=None, logP=False,
                 logsign=1):
        self.islog = logP
        z, k = (np.atleast_1d(x) for x in [z, k])
        if len(z) < 4:
            raise ValueError('Require at least four redshifts for Pk interpolation.'
                             'Consider using Pk_grid if you just need a small number'
                             'of specific redshifts (doing 1D splines in k yourself).')
        z, k, P_or_logP = np.array(z), np.array(k), np.array(P_or_logP)
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.logsign = logsign
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        logk = np.log(self.k)
        if extrap_kmin and extrap_kmin < self.input_kmin:
            if not logP:
                raise ValueError('extrap_kmin must use logP')
            logk = np.hstack(
                [np.log(extrap_kmin),
                 np.log(self.input_kmin) * 0.1 + np.log(extrap_kmin) * 0.9, logk])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, 2:] = P_or_logP
            diff = (logPnew[:, 3] - logPnew[:, 2]) / (logk[3] - logk[2])
            delta = diff * (logk[2] - logk[0])
            logPnew[:, 0] = logPnew[:, 2] - delta
            logPnew[:, 1] = logPnew[:, 2] - delta * 0.9
            P_or_logP = logPnew
        if extrap_kmax and extrap_kmax > self.input_kmax:
            if not logP:
                raise ValueError('extrap_kmax must use logP')
            logk = np.hstack(
                [logk, np.log(self.input_kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                 np.log(extrap_kmax)])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logk[-3] - logk[-4])
            delta = diff * (logk[-1] - logk[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            P_or_logP = logPnew
        super().__init__(self.z, logk, P_or_logP)

    @property
    def input_kmin(self):
        return self.k[0]

    @property
    def input_kmax(self):
        return self.k[-1]

    @property
    def kmin(self):
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
        z = np.atleast_1d(z).flatten()
        min_z, max_z = min(z), max(z)
        if min_z < self.zmin and not np.allclose(min_z, self.zmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={min(z)} "
                              f"(minimum z computed is {self.zmin}).")
        if max_z > self.zmax and not np.allclose(max_z, self.zmax):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={max(z)} "
                              f"(maximum z computed is {self.zmax}).")
        k = np.atleast_1d(k).flatten()
        min_k, max_k = min(k), max(k)
        if min_k < self.kmin and not np.allclose(min_k, self.kmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={min(k)} 1/Mpc "
                              f"(minimum k possible is {self.kmin} 1/Mpc).")
        if max_k > self.kmax and not np.allclose(max_k, self.kmax):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={max(k)} 1/Mpc "
                              f"(maximum k possible is {self.kmax} 1/Mpc).")

    def P(self, z, k, grid=None):
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid, warn=False)
        else:
            return np.log(self(z, np.log(k), grid=grid, warn=False))

    def __call__(self, *args, warn=True, **kwargs):
        if warn:
            get_logger(self.__class__.__name__).warning(
                "Do not call the instance directly. Use instead methods P(z, k) or "
                "logP(z, k) to get the (log)power spectrum. (If you know what you are "
                "doing, pass warn=False)")
        return super().__call__(*args, **kwargs)


class emulmps(Theory):
    """
    Fast Matter Power Spectrum Emulator Theory Code.

    Linear P(k) always comes from the linear emulator (unless use_syren=True).

    Nonlinear P(k):

    1. **Emulated boost**: if nl_model_file and nl_metadata_file are supplied
       in extra_args, the boost network predicts the dimensionless
       B(k,z) = P_nl/P_lin = exp(NN) * B_syren, and

           P_nl,emul = B_emul * P_lin,emul

       with P_lin,emul the output of the linear emulator.

    2. **Linear only**: if no boost model is loaded, only linear P(k) is
       stored and requesting nonlinear P(k) raises an error.  (The symbolic
       fallback selected by nonlinear_method is not implemented.)

    Extra args
    ----------
    model_file : str
        Path to the linear Keras model.
    metadata_file : str
        Path to the linear metadata bundle.
    model_type : str
        'mlp' or 'npce' (default: 'mlp').
    nl_model_file : str, optional
        Path to the nonlinear boost Keras model.
    nl_metadata_file : str, optional
        Path to the nonlinear boost metadata bundle.
    nl_model_type : str
        'mlp' or 'npce' for the boost model (default: 'mlp').
    nonlinear_method : str or None
        Reserved for a symbolic fallback ('syrenhalofit', 'halofit+' or None).
        Currently has no effect on the computation.
    use_syren : bool
        DEBUG ONLY.  If True, bypass BOTH networks and return the pure syren
        approximations (syren P_lin and syren-halofit boost).  Must be False
        (the default) to use the emulators.
    param_order : list of str
        Cosmological parameters the emulator reads from Cobaya (default: the
        full w0waCDM + HMCode_logT_AGN + mnu list).  Optional entries:

        * w / w0, wa       -> omitted: w0 = -1, wa = 0 (LCDM / wCDM)
        * HMCode_logT_AGN  -> omitted: fixed at fixed_params['HMCode_logT_AGN']
        * mnu              -> omitted: fixed at fixed_params['mnu']

        At start-up param_order is checked against the number of inputs each
        loaded network was trained on: listing HMCode_logT_AGN for a model
        without it is an error; omitting a parameter a model does take gives
        a warning and uses the fixed value.
    fixed_params : dict, optional
        Values used for HMCode_logT_AGN and mnu when they are not in
        param_order (default {HMCode_logT_AGN: 7.8, mnu: 0.06}).  For a model
        trained at fixed mnu this must be the training value, since mnu
        always enters the syren baseline.
    kmax_cut : float or None
        Drop emulator k modes above this value, in 1/Mpc (default 50; the same
        units as CAMB's kmax in a Cobaya yaml).  Applied to the k grid, the
        linear spectrum and the boost before anything else uses them.  Set to
        null to keep the full emulator grid.

    Hybrid CAMB / emulator test mode
    --------------------------------
    lin_source : 'emul' (default) or 'camb'
        Where the linear P(k) comes from.
    boost_source : 'emul' (default) or 'camb'
        Where the boost B = P_nl / P_lin comes from.

        P_nl = B[boost_source] * P_lin[lin_source], and the linear grid handed
        to the likelihood is P_lin[lin_source].  With both set to 'emul' the
        code path is identical to the plain emulator.  If either is 'camb', a
        nested Cobaya model holding only CAMB is evaluated at every point and
        interpolated onto the emulator's (z, k) grid.
    camb : dict
        Required if either source is 'camb'.  The CAMB theory block copied
        verbatim from the CAMB yaml (path, use_renames, extra_args, ...), so
        the nested CAMB has the same settings as the reference run.
    camb_params : dict, optional
        Extra fixed parameters for the nested CAMB (e.g. {tau: 0.0697186}, or
        mnu if it is not in param_order).
    camb_want_cl : bool
        Also request Cl tt from the nested CAMB (default True), as the
        roman_real likelihood does from CAMB in use_emulator 0 mode; this
        changes CAMB's k sampling, so it is needed for an exact match.
    camb_max_nz : int
        CAMB is run at the emulator's own redshifts when there are at most
        this many (default 256, CAMB's own limit), so nothing is interpolated
        in z.  With more, CAMB runs at a subsample and is splined in z; avoid
        that with halofit_version mead2020_feedback, whose boost has a kink
        in z at z = 2 that a spline smooths over (percent-level at k > 5/Mpc).
    pk_compare_file : str, optional
        If set, save k, z and both versions of P_lin and of the boost to this
        .npz at each evaluation ('{i}' in the name is replaced by the call
        number).

    In hybrid mode omegabh2 and omegach2 are added to the requirements so the
    nested CAMB receives exactly the values defined in the yaml.
    """

    renames: Mapping[str, str]
    extra_args: InfoDict
    path: str

    def initialize(self):
        super().initialize()

        self.renames    = empty_dict
        self.extra_args = getattr(self, 'extra_args', {})

        # False = use the emulators.  True bypasses both networks (debug only).
        self.use_syren         = bool(self.extra_args.get('use_syren', False))
        self.nonlinear_method  = self.extra_args.get('nonlinear_method', 'halofit+')

        # Validate symbolic fallback method
        if self.nonlinear_method is not None:
            if not SYMBOLIC_POFK_AVAILABLE:
                raise LoggedError(
                    self.log,
                    "nonlinear_method specified but symbolic_pofk is not installed."
                )
            valid_methods = ['syrenhalofit', 'halofit+']
            if self.nonlinear_method not in valid_methods:
                raise LoggedError(
                    self.log,
                    f"Invalid nonlinear_method '{self.nonlinear_method}'. "
                    f"Valid options: {valid_methods}"
                )

        self.param_order = self.extra_args.get(
            'param_order',
            ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa", "HMCode_logT_AGN", "mnu"]
        )

        valid_params = {"As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa", "w", "HMCode_logT_AGN", "mnu"}
        for param in self.param_order:
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{param}' in param_order. "
                    f"Valid parameters: {valid_params}"
                )

        required_base = {"As_1e9", "ns", "H0", "omegab", "omegam"}
        provided = set(self.param_order)
        if not required_base.issubset(provided):
            missing = required_base - provided
            raise ValueError(f"Missing required parameters: {missing}.")

        if "w" in self.param_order and "w0" in self.param_order:
            raise ValueError(
                "Cannot specify both 'w' and 'w0' in param_order."
            )

        # Values for the optional inputs when they are not sampled/passed.
        self.fixed_params = {'HMCode_logT_AGN': 7.8, 'mnu': 0.06}
        user_fixed = dict(self.extra_args.get('fixed_params', None) or {})
        bad = set(user_fixed) - set(self.fixed_params)
        if bad:
            raise LoggedError(
                self.log,
                f"fixed_params only accepts {sorted(self.fixed_params)}, got {sorted(bad)}."
            )
        self.fixed_params.update({k: float(v) for k, v in user_fixed.items()})
        clash = set(user_fixed) & set(self.param_order)
        if clash:
            self.log.warning(
                f"fixed_params {sorted(clash)} are also in param_order; the value "
                "from Cobaya is used and fixed_params is ignored for them."
            )

        self.req = {param: None for param in self.param_order}

        # --- Hybrid CAMB / emulator test mode ---------------------------------
        self.lin_source   = str(self.extra_args.get('lin_source', 'emul')).lower()
        self.boost_source = str(self.extra_args.get('boost_source', 'emul')).lower()
        for name, src in (('lin_source', self.lin_source),
                          ('boost_source', self.boost_source)):
            if src not in ('emul', 'camb'):
                raise LoggedError(
                    self.log, f"{name} must be 'emul' or 'camb', got '{src}'."
                )
        self._use_camb    = 'camb' in (self.lin_source, self.boost_source)
        self._camb_model  = None
        self._camb_z      = None
        self._n_calls     = 0
        self._warned_lowk = False

        # --- k cut: keep only emulator modes with k <= kmax_cut (1/Mpc) -------
        kmax_cut = self.extra_args.get('kmax_cut', 50.0)
        self.kmax_cut = float(kmax_cut) if kmax_cut else None
        if self.kmax_cut is not None and not self.kmax_cut > 0:
            raise LoggedError(self.log, f"kmax_cut must be positive, got {kmax_cut}.")
        self._logged_kcut = False

        self.pk_compare_file = self.extra_args.get('pk_compare_file', None)
        if self._use_camb:
            self.camb_info = self.extra_args.get('camb', None)
            if not isinstance(self.camb_info, Mapping):
                raise LoggedError(
                    self.log,
                    "lin_source/boost_source = 'camb' needs a 'camb' block in "
                    "extra_args (copy the camb theory block of the CAMB yaml)."
                )
            self.camb_params = dict(self.extra_args.get('camb_params', None) or {})
            self.camb_max_nz = int(self.extra_args.get('camb_max_nz', 256))
            self.camb_want_cl = bool(self.extra_args.get('camb_want_cl', True))
            # Take the physical densities from the yaml so the nested CAMB sees
            # the same omegabh2 / omegach2 (neutrino convention included) as
            # the reference CAMB run.
            for p in ('omegabh2', 'omegach2'):
                self.req[p] = None

        model_file       = self.extra_args.get('model_file', None)
        metadata_file    = self.extra_args.get('metadata_file', None)
        model_type       = self.extra_args.get('model_type', 'mlp')
        nl_model_file    = self.extra_args.get('nl_model_file', None)
        nl_metadata_file = self.extra_args.get('nl_metadata_file', None)
        nl_model_type    = self.extra_args.get('nl_model_type', 'mlp')

        self.emul_folder = self.extra_args.get('emul_folder', None)

        if (nl_model_file is None) != (nl_metadata_file is None):
            raise LoggedError(
                self.log,
                "nl_model_file and nl_metadata_file must be given together."
            )

        from .emulmps_emul.emulmps_w0wa import PkEmulator
        self._emulator = PkEmulator(
            model_file=model_file,
            metadata_file=metadata_file,
            model_type=model_type,
            nl_model_file=nl_model_file,
            nl_metadata_file=nl_metadata_file,
            nl_model_type=nl_model_type,
        )
        self._emulator.DEFAULT_TAGN = self.fixed_params['HMCode_logT_AGN']
        self._emulator.DEFAULT_MNU  = self.fixed_params['mnu']
        self._check_param_coverage()

        ##### vic add
        # --- PC diagnostics (off by default; only for single-eval T_AGN sweeps) ---
        self.save_pc_diagnostics    = self.extra_args.get('save_pc_diagnostics', False)
        self.pc_responsibility_path = self.extra_args.get(
            'pc_responsibility_path', 'pc_k20_30_responsibility.txt')
        self.pc_diagnostics_path    = self.extra_args.get(
            'pc_diagnostics_path', 'pcs_coeffs_vs_tagn.txt')

        if self.save_pc_diagnostics and self._emulator.has_nl_model():
            K_LO, K_HI = 20.0, 30.0
            emu = self._emulator
            k_mask = (emu.K_MODES >= K_LO) & (emu.K_MODES <= K_HI)
            mats = emu.NL_INVERSE_TRANSFORM_MATRICES        # (N_ZS, n_pcs, N_K)

            window_energy = np.sum(mats[:, :, k_mask] ** 2, axis=2)
            total_energy  = np.sum(mats ** 2, axis=2)
            frac_energy_in_window = window_energy / total_energy

            rank_rows = []
            for iz, z_val in enumerate(emu.Z_MODES):
                for p in range(mats.shape[1]):
                    rank_rows.append((iz, z_val, p, frac_energy_in_window[iz, p]))

            np.savetxt(
                self.pc_responsibility_path,
                np.array(rank_rows),
                header="z_index  z_value  pc_index  frac_energy_in_k20_30_window",
                fmt="%d %.6f %d %.6e",
            )
            self.log.info(f"Saved PC responsibility ranking to {self.pc_responsibility_path}")

        ##### vic add end

        if self._use_camb:
            if self.boost_source == 'emul' and not self._emulator.has_nl_model():
                raise LoggedError(
                    self.log,
                    "boost_source='emul' needs nl_model_file and nl_metadata_file."
                )
            self.log.warning(
                "HYBRID TEST MODE: P_lin from %s, boost from %s "
                "(P_nl = B_%s * P_lin,%s).  CAMB runs at every evaluation.",
                self.lin_source.upper(), self.boost_source.upper(),
                self.boost_source, self.lin_source,
            )
        elif self._emulator.has_nl_model():
            self.log.info("Nonlinear boost: emulated, P_nl = B_emul * P_lin,emul.")
        else:
            self.log.info("Nonlinear boost: disabled (linear P(k) only).")
            if self.nonlinear_method is not None:
                self.log.warning(
                    f"nonlinear_method='{self.nonlinear_method}' was requested but no boost "
                    "model is loaded and the symbolic fallback is not implemented: only "
                    "linear P(k) will be available."
                )

        if self.use_syren:
            self.log.warning(
                "use_syren=True: BOTH networks are bypassed and the pure syren "
                "approximations are returned.  Set use_syren: False to use the emulators."
            )

        self.log.info(
            f"PkEmulator loaded (model_file={model_file}, "
            f"metadata_file={metadata_file})"
        )

        if ("w0" in self.param_order or "w" in self.param_order) and "wa" in self.param_order:
            cosmo_type = "w0waCDM"
        elif "w" in self.param_order:
            cosmo_type = "wCDM (constant w)"
        elif "w0" in self.param_order:
            cosmo_type = "wCDM (w0 only, wa=0)"
        else:
            cosmo_type = "LCDM"

        self.log.info(f"emulmps emulator initialized with {cosmo_type} cosmology")
        self.log.info(f"Parameter order: {self.param_order}")

    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=True, **params):
        """
        Calculate P(k,z) and store linear (and optionally nonlinear) grids in
        state.

        P_lin comes from the linear emulator.  If a boost model is loaded,
        P_nl = B_emul * P_lin with B_emul the dimensionless emulated boost;
        otherwise only the linear grid is stored.
        """
        try:
            emul_params = self._emul_param_vector(params)

            # ------------------------------------------------------------------
            # Linear P(k)
            # ------------------------------------------------------------------
            k_mpc, z_array, Pk_lin_mpc = self._emulator.get_pks(
                emul_params, use_syren=self.use_syren
            )

            # Drop emulator k modes above kmax_cut (1/Mpc).  Everything below
            # (validity checks, boost, CAMB hybrid, stored grids) sees only
            # the cut grid.
            k_keep = self._k_keep(k_mpc)
            if k_keep is not None:
                k_mpc      = np.asarray(k_mpc)[k_keep]
                Pk_lin_mpc = np.asarray(Pk_lin_mpc)[:, k_keep]

            if not np.all(np.isfinite(Pk_lin_mpc)) or np.any(~(Pk_lin_mpc > 0)):
                self.log.debug(f"Non-finite or non-positive Pk_lin at params={params} — rejecting point.")
                return False

            # ------------------------------------------------------------------
            # Nonlinear P(k) = B_emul * P_lin,emul
            #
            # get_boost returns the dimensionless B = exp(NN) * B_syren on the
            # full (z, k) grid.  It is applied to the EMULATED linear spectrum
            # above, so P_nl never inherits syren's linear P(k).
            # ------------------------------------------------------------------
            Pk_nl_mpc = None
            if self._emulator.has_nl_model():
                _, _, boost = self._emulator.get_boost(
                    emul_params, use_syren=self.use_syren
                )
                if k_keep is not None:
                    boost = np.asarray(boost)[:, k_keep]

                #### vic add

                # --- PC diagnostics: capture the NL PCA coefficients for this sample ---
                if self.save_pc_diagnostics:
                    pcs_z = _get_nl_pcs(self._emulator, emul_params)   # (N_ZS, n_pcs)
                    tagn_val = emul_params[7]
                    rows = [
                        (tagn_val, iz, z_val, p, pcs_z[iz, p])
                        for iz, z_val in enumerate(self._emulator.Z_MODES)
                        for p in range(pcs_z.shape[1])
                    ]
                    with open(self.pc_diagnostics_path, "a") as f:
                        if f.tell() == 0:
                            f.write("# tagn  z_index  z_value  pc_index  coefficient\n")
                        np.savetxt(f, rows, fmt="%.4f %d %.6f %d %.8e")

                #### vic add end

                if not np.all(np.isfinite(boost)) or np.any(~(boost > 0)):
                    # Only fatal if the emulated boost is actually used.  With
                    # boost_source='camb' it is kept for the comparison only.
                    if not (self._use_camb and self.boost_source == 'camb'):
                        self.log.debug(
                            f"Non-finite or non-positive boost at params={params} — "
                            "rejecting point."
                        )
                        return False
                    self.log.warning(
                        "Emulated boost is non-finite or non-positive at this point "
                        "(not used: boost_source='camb'); the saved comparison will "
                        "contain NaN/inf for boost_emul."
                    )

            # ------------------------------------------------------------------
            # Hybrid test mode: swap the linear spectrum and/or the boost for
            # CAMB's, on the emulator's own (z, k) grid.
            # ------------------------------------------------------------------
            if self._use_camb:
                boost_emul = boost if self._emulator.has_nl_model() else None
                camb_out = self._camb_lin_and_boost(params, k_mpc, z_array)
                if camb_out is None:
                    self.log.debug(f"Nested CAMB failed at params={params} — rejecting point.")
                    return False
                Pk_lin_camb, boost_camb = camb_out
                self._compare_with_camb(
                    k_mpc, z_array, Pk_lin_mpc, Pk_lin_camb, boost_emul, boost_camb
                )
                if self.lin_source == 'camb':
                    Pk_lin_mpc = Pk_lin_camb.astype(Pk_lin_mpc.dtype)
                boost = boost_camb if self.boost_source == 'camb' else boost_emul

            if self._nl_available():
                Pk_nl_mpc = (boost * Pk_lin_mpc).astype(np.float32)

                if not np.all(np.isfinite(Pk_nl_mpc)) or np.any(~(Pk_nl_mpc > 0)):
                    self.log.debug(
                        f"Non-finite or non-positive Pk_nl at params={params} — "
                        "rejecting point."
                    )
                    return False

            # ------------------------------------------------------------------
            # Store results
            # ------------------------------------------------------------------
            if Pk_nl_mpc is not None:
                state[("Pk_grid", True, "delta_tot", "delta_tot")] = (
                    k_mpc, z_array, Pk_nl_mpc
                )

            state[("Pk_grid", False, "delta_tot", "delta_tot")] = (
                k_mpc, z_array, Pk_lin_mpc
            )
            state["Pk_grid"] = {
                'k': k_mpc,
                'z': z_array,
                'Pk': Pk_lin_mpc,
            }

            if want_derived:
                derived = {}
                if 'sigma8' in self.output_params:
                    derived['sigma8'] = self._compute_sigma8(
                        Pk_lin_mpc, k_mpc, z_array, z=0.0
                    )
                state["derived"] = derived

            return True

        except Exception as e:
            if self.stop_at_error:
                self.log.error(
                    f"emulmps emulator evaluation failed: {e}\n"
                    f"Parameters: {params}"
                )
                raise
            else:
                self.log.debug(
                    f"emulmps emulator evaluation failed: {e}\n"
                    "Returning False (likelihood=0)."
                )
                return False

    # ----------------------------------------------------------------------
    # Parameter handling
    # ----------------------------------------------------------------------
    def _emul_param_vector(self, params):
        """
        Build the emulator's full input [As_1e9, ns, H0, Ob, Om, w0, wa,
        T_AGN, mnu] by NAME, so param_order may omit (or reorder) the
        optional entries.  Omitted: w0 = -1, wa = 0, T_AGN and mnu from
        fixed_params.
        """
        po = self.param_order
        if 'w' in po:
            w0 = params['w']
        elif 'w0' in po:
            w0 = params['w0']
        else:
            w0 = -1.0
        return [
            params['As_1e9'], params['ns'], params['H0'],
            params['omegab'], params['omegam'],
            w0,
            params['wa'] if 'wa' in po else 0.0,
            params['HMCode_logT_AGN'] if 'HMCode_logT_AGN' in po
            else self.fixed_params['HMCode_logT_AGN'],
            params['mnu'] if 'mnu' in po else self.fixed_params['mnu'],
        ]

    def _check_param_coverage(self):
        """
        Compare param_order with the inputs the loaded networks actually read
        (each reads the first n_features_in_ columns of
        [As, ns, H0, Ob, Om, w0+wa, ..., T_AGN, mnu]).
        """
        if self.use_syren:
            return
        emu = self._emulator
        n_in = {'linear': emu.n_lin_inputs}
        if emu.has_nl_model():
            n_in['boost'] = emu.n_nl_inputs
        self.log.info(
            "Network inputs: " + ", ".join(f"{m} = {n}" for m, n in n_in.items())
        )
        for col, name in ((7, 'HMCode_logT_AGN'), (8, 'mnu')):
            readers = [m for m, n in n_in.items() if n > col]
            listed = name in self.param_order
            if readers and not listed:
                self.log.warning(
                    f"The {' and '.join(readers)} model(s) take {name} as an input but "
                    f"it is not in param_order: fixed at {self.fixed_params[name]} "
                    "(set fixed_params, or add it to param_order to vary it)."
                )
            elif listed and not readers:
                if name == 'HMCode_logT_AGN':
                    raise LoggedError(
                        self.log,
                        "HMCode_logT_AGN is in param_order but none of the loaded "
                        f"models use it (inputs: {n_in}).  Remove it from param_order "
                        "and from the yaml params block."
                    )
                self.log.warning(
                    "mnu is in param_order but none of the loaded networks take it "
                    f"as an input (inputs: {n_in}).  It only changes the syren "
                    "baseline; keep it fixed at the models' training value."
                )
        if 'HMCode_logT_AGN' not in self.param_order and not any(
                n > 7 for n in n_in.values()):
            self.log.info("No T_AGN input (e.g. mead2020 without feedback).")

    # ----------------------------------------------------------------------
    # Hybrid CAMB / emulator test mode
    # ----------------------------------------------------------------------
    def _k_keep(self, k_mpc):
        """
        Boolean mask of the emulator k modes kept by kmax_cut (k <= kmax_cut,
        both in 1/Mpc), or None if nothing is cut.  Modes are dropped, never
        interpolated, so the last k kept is the largest grid point <= the cut.
        """
        if self.kmax_cut is None:
            return None
        k_mpc = np.asarray(k_mpc)
        keep = k_mpc <= self.kmax_cut * (1.0 + 1e-10)
        if keep.all():
            return None
        if keep.sum() < 4:
            raise LoggedError(
                self.log,
                f"kmax_cut = {self.kmax_cut} 1/Mpc leaves only {keep.sum()} of "
                f"{keep.size} emulator k modes (grid starts at {k_mpc.min():.3g} 1/Mpc)."
            )
        if not self._logged_kcut:
            self._logged_kcut = True
            self.log.info(
                "kmax_cut = %g 1/Mpc: keeping %d of %d emulator k modes "
                "(last k kept = %.4g 1/Mpc, emulator grid went to %.4g 1/Mpc).",
                self.kmax_cut, keep.sum(), keep.size, k_mpc[keep].max(), k_mpc.max()
            )
        return keep

    def _nl_available(self):
        """Nonlinear P(k) exists if a boost network is loaded or CAMB supplies the boost."""
        return self._emulator.has_nl_model() or (
            self._use_camb and self.boost_source == 'camb'
        )

    def _camb_point(self, params):
        """Parameters handed to the nested CAMB, mirroring the CAMB yaml."""
        point = {
            'As':       1e-9 * params['As_1e9'],
            'ns':       params['ns'],
            'H0':       params['H0'],
            'omegabh2': params['omegabh2'],
            'omegach2': params['omegach2'],
        }
        if 'w' in self.param_order:
            point['w'] = params['w']
        elif 'w0' in self.param_order:
            point['w'] = params['w0']
        for p in ('wa', 'mnu', 'HMCode_logT_AGN'):
            if p in self.param_order:
                point[p] = params[p]
        # Same fixed mnu as the emulator (T_AGN is not passed: plain mead2020
        # in CAMB does not accept it).
        if 'mnu' not in point and 'mnu' not in self.camb_params:
            point['mnu'] = self.fixed_params['mnu']
        return {k: float(v) for k, v in point.items()}

    def _init_camb_model(self, point, k_mpc, z_array):
        """Build the nested Cobaya model that holds only CAMB (done once)."""
        from copy import deepcopy
        from cobaya.model import get_model

        z_req = np.unique(np.asarray(z_array, dtype=np.float64))
        if len(z_req) > self.camb_max_nz:
            idx = np.unique(
                np.round(np.linspace(0, len(z_req) - 1, self.camb_max_nz)).astype(int)
            )
            z_req = z_req[idx]
            self.log.warning(
                "Nested CAMB: emulator has %d redshifts, running CAMB at %d of "
                "them and splining in z.  The mead2020_feedback boost has a kink "
                "at z = 2, so expect percent-level interpolation error at high k "
                "near there.", len(np.unique(z_array)), len(z_req)
            )

        # Wide flat priors: the nested model is only ever evaluated at the
        # point given by the outer run, so these never matter.
        wide = {
            'As': (1e-12, 1e-6), 'ns': (0.1, 2.0), 'H0': (10.0, 200.0),
            'omegabh2': (1e-4, 1.0), 'omegach2': (1e-4, 5.0), 'mnu': (0.0, 10.0),
            'w': (-20.0, 5.0), 'wa': (-20.0, 20.0), 'HMCode_logT_AGN': (3.0, 12.0),
        }
        params_info = {p: {'prior': {'min': wide[p][0], 'max': wide[p][1]}}
                       for p in point}
        clash = set(self.camb_params) & set(params_info)
        if clash:
            raise LoggedError(
                self.log,
                f"camb_params {sorted(clash)} are already passed from param_order."
            )
        params_info.update({p: {'value': v} for p, v in self.camb_params.items()})

        info = {
            # keep the outer run's log level (get_model resets the logger)
            'debug': self.log.getEffectiveLevel() <= logging.DEBUG,
            'params': params_info,
            'likelihood': {'one': None},
            'theory': {'camb': deepcopy(dict(self.camb_info))},
        }
        self._camb_model = get_model(info)
        # With kmax_cut the grid never exceeds the cut, so ask for exactly that
        # and let the camb block's own kmax decide (same kmax as the reference
        # CAMB run).  Without a cut, leave 20% head-room in case the emulator
        # k grid scales with h.
        headroom = 1.0 if self.kmax_cut else 1.2
        reqs = {
            'Pk_grid': {
                'z': z_req,
                'k_max': headroom * float(np.max(k_mpc)),
                'nonlinear': (False, True),
            }
        }
        # The roman_real likelihood in CAMB mode requests Cl tt from CAMB, which
        # changes CAMB's k sampling (P(k) moves by up to ~0.1% near the BAO
        # scales).  Ask for the same here so the nested CAMB matches it.
        if self.camb_want_cl:
            reqs['Cl'] = {'tt': 0}
        self._camb_model.add_requirements(reqs)
        self._camb_z = z_req

    def _camb_to_emul_grid(self, k_c, z_c, vals_c, k_mpc, z_array, lowk):
        """
        Interpolate a positive CAMB quantity from (z_c, k_c) onto the emulator
        (z_array, k_mpc) grid: cubic spline of the log in log k, then (only if
        the redshifts differ) cubic spline in z.

        Below CAMB's smallest k: lowk='powerlaw' continues the log-log slope
        (linear spectrum), lowk='flat' holds the edge value (boost).
        """
        from scipy.interpolate import CubicSpline

        logk_c = np.log(np.asarray(k_c, dtype=np.float64))
        logk   = np.log(np.asarray(k_mpc, dtype=np.float64))
        logv_c = np.log(np.asarray(vals_c, dtype=np.float64))

        inside = np.clip(logk, logk_c[0], logk_c[-1])
        out = CubicSpline(logk_c, logv_c, axis=1)(inside)          # (nz_c, nk)

        low = logk < logk_c[0]
        if np.any(low) and lowk == 'powerlaw':
            slope = (logv_c[:, 1] - logv_c[:, 0]) / (logk_c[1] - logk_c[0])
            out[:, low] = logv_c[:, [0]] + slope[:, None] * (logk[low] - logk_c[0])[None, :]

        z_c     = np.asarray(z_c, dtype=np.float64)
        z_array = np.asarray(z_array, dtype=np.float64)
        idx = np.clip(np.searchsorted(z_c, z_array), 0, len(z_c) - 1)
        idx = np.where(
            np.abs(z_c[np.maximum(idx - 1, 0)] - z_array) < np.abs(z_c[idx] - z_array),
            np.maximum(idx - 1, 0), idx
        )
        if np.allclose(z_c[idx], z_array, rtol=0.0, atol=1e-8):
            out = out[idx]
        else:
            out = CubicSpline(z_c, out, axis=0)(z_array)
        return np.exp(out)

    def _camb_lin_and_boost(self, params, k_mpc, z_array):
        """
        Run the nested CAMB at this point and return (P_lin, boost) on the
        emulator grid, in 1/Mpc and Mpc^3.  Returns None if CAMB fails.
        """
        point = self._camb_point(params)
        if self._camb_model is None:
            self._init_camb_model(point, k_mpc, z_array)

        loglike = self._camb_model.loglikes(point, return_derived=False)
        if not np.all(np.isfinite(np.atleast_1d(loglike))):
            if self.stop_at_error:
                raise LoggedError(self.log, f"Nested CAMB failed at {point}.")
            return None

        provider = self._camb_model.provider
        k_c, z_c, pk_lin = provider.get_Pk_grid(nonlinear=False)
        k_n, z_n, pk_nl  = provider.get_Pk_grid(nonlinear=True)
        if not (np.array_equal(k_c, k_n) and np.array_equal(z_c, z_n)):
            raise LoggedError(self.log, "Nested CAMB linear and nonlinear grids differ.")

        if np.max(k_mpc) > k_c[-1] * (1.0 + 1e-6):
            raise LoggedError(
                self.log,
                f"Emulator k_max = {np.max(k_mpc):.4g} 1/Mpc is above the nested "
                f"CAMB k_max = {k_c[-1]:.4g} 1/Mpc; raise kmax in the camb block."
            )
        if np.min(k_mpc) < k_c[0] and not self._warned_lowk:
            self._warned_lowk = True
            self.log.warning(
                "Emulator k_min = %.3g 1/Mpc is below CAMB's k_min = %.3g 1/Mpc: "
                "the CAMB linear spectrum is continued as a power law and the "
                "CAMB boost held constant below that.", np.min(k_mpc), k_c[0]
            )

        order = np.argsort(z_c)
        z_c, pk_lin, pk_nl = z_c[order], pk_lin[order], pk_nl[order]

        Pk_lin_camb = self._camb_to_emul_grid(k_c, z_c, pk_lin, k_mpc, z_array, 'powerlaw')
        boost_camb  = self._camb_to_emul_grid(k_c, z_c, pk_nl / pk_lin, k_mpc, z_array, 'flat')
        return Pk_lin_camb, boost_camb

    def _compare_with_camb(self, k_mpc, z_array, Pk_lin_emul, Pk_lin_camb,
                           boost_emul, boost_camb):
        """Log (and optionally save) emulator vs CAMB at the P(k) level."""
        self._n_calls += 1
        np_err = np.seterr(all="ignore")          # boost_emul may be NaN/inf (see calculate)
        d_lin = np.asarray(Pk_lin_emul, dtype=np.float64) / Pk_lin_camb - 1.0
        msg = (f"emulator vs CAMB on the emulator grid: P_lin max |frac err| = "
               f"{np.max(np.abs(d_lin)):.3e} (rms {np.sqrt(np.mean(d_lin**2)):.3e})")
        if boost_emul is not None:
            d_b = np.asarray(boost_emul, dtype=np.float64) / boost_camb - 1.0
            msg += (f"; boost max |frac err| = {np.max(np.abs(d_b)):.3e} "
                    f"(rms {np.sqrt(np.mean(d_b**2)):.3e})")
        self.log.info(msg)
        np.seterr(**np_err)

        if self.pk_compare_file:
            out = {
                'k': np.asarray(k_mpc), 'z': np.asarray(z_array),
                'Pk_lin_emul': np.asarray(Pk_lin_emul), 'Pk_lin_camb': Pk_lin_camb,
                'boost_camb': boost_camb,
            }
            if boost_emul is not None:
                out['boost_emul'] = np.asarray(boost_emul)
            np.savez(str(self.pk_compare_file).replace('{i}', str(self._n_calls)), **out)

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"), nonlinear=False):
        r"""
        Get matter power spectrum grid.

        Returns P(k,z) in standard Cobaya units: k in 1/Mpc, Pk in Mpc^3.
        """
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )

        nl_available = self._nl_available()
        if nonlinear and not nl_available:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but no boost model is loaded "
                "(set nl_model_file and nl_metadata_file in extra_args)."
            )

        key = ("Pk_grid", nonlinear) + tuple(sorted(var_pair))
        if key in self.current_state:
            return self.current_state[key]

        if not nonlinear and "Pk_grid" in self.current_state:
            pk_dict = self.current_state["Pk_grid"]
            return pk_dict['k'], pk_dict['z'], pk_dict['Pk']

        raise LoggedError(
            self.log,
            f"Matter power spectrum (nonlinear={nonlinear}) not computed."
        )

    def get_Pk_interpolator(
        self,
        var_pair=("delta_tot", "delta_tot"),
        nonlinear=False,
        extrap_kmin=None,
        extrap_kmax=None,
    ) -> PowerSpectrumInterpolator:
        r"""Get a P(z,k) bicubic interpolation object."""
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )

        nl_available = self._nl_available()
        if nonlinear and not nl_available:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but no boost model is loaded "
                "(set nl_model_file and nl_metadata_file in extra_args)."
            )

        key = (
            ("Pk_interpolator", nonlinear, extrap_kmin, extrap_kmax) +
            tuple(sorted(var_pair))
        )

        if key in self.current_state:
            return self.current_state[key]

        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)

        log_p = True
        sign  = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
                self.log.debug(
                    "Power spectrum has both positive and negative values; "
                    "using linear interpolation."
                )

        extrapolating = (
            (extrap_kmax and extrap_kmax > k[-1]) or
            (extrap_kmin and extrap_kmin < k[0])
        )

        if log_p:
            pk_for_interp = np.log(sign * pk)
        elif extrapolating:
            raise LoggedError(
                self.log,
                f'Cannot do log extrapolation with zero-crossing Pk for {var_pair}'
            )
        else:
            pk_for_interp = pk

        result = PowerSpectrumInterpolator(
            z, k, pk_for_interp,
            logP=log_p,
            logsign=sign,
            extrap_kmin=extrap_kmin,
            extrap_kmax=extrap_kmax,
        )

        self.current_state[key] = result
        return result

    def _compute_sigma8(self, Pk_2d, k_array, z_array, z=0.0):
        """Compute sigma8 from linear P(k) at redshift z."""
        z_idx = np.argmin(np.abs(z_array - z))
        if np.abs(z_array[z_idx] - z) > 0.01:
            from scipy.interpolate import interp1d
            Pk_interp = interp1d(z_array, Pk_2d, axis=0, kind='cubic')
            Pk_z = Pk_interp(z)
        else:
            Pk_z = Pk_2d[z_idx, :]

        R  = 8.0
        x  = k_array * R
        W  = np.zeros_like(x)
        mask = x > 1e-6
        W[mask]  = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / x[mask]**3
        W[~mask] = 1.0

        integrand = k_array**2 * Pk_z * W**2 / (2.0 * np.pi**2)
        log_k     = np.log(k_array)
        return np.sqrt(np.trapz(integrand * k_array, log_k))

    def get_can_support_params(self):
        return ['Pk_grid', 'Pk_interpolator', 'sigma8']

    def get_sigma8(self):
        return self.current_state.get('derived', {}).get('sigma8')

    def get_param(self, param_name):
        return self.current_state.get('derived', {}).get(param_name)

    def get_version(self):
        return '1.0.0'

    def get_speed(self):
        return 20.0