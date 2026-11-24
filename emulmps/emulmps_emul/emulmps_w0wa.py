#  Author: Victoria Lloyd (2025) & V. Miranda
 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
 
import warnings
warnings.filterwarnings("ignore", message=".*TF-TRT.*")
 
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
 
# Clean warning format: category name + message only, no file/line noise
warnings.formatwarning = lambda msg, cat, *a, **kw: f"{cat.__name__}: {msg}\n"
 
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
from . import train_utils_pk_emulator as utils
from . import train_utils_pk_emulator as utils
from . train_utils_pk_emulator import CustomActivationLayer, TComponentScaler
import tensorflow as tf

class _AliasLoader:
    """Redirect any train_utils_pk_emulator_vN import to the canonical module."""
    def find_module(self, name, path=None):
        if name.startswith('train_utils_pk_emulator'):
            return self

    def load_module(self, name):
        if name not in sys.modules:
            sys.modules[name] = utils
        return sys.modules[name]

sys.meta_path.append(_AliasLoader())

from .numerical.numerical_halofit import *
from .numerical.numerical_linear import *

PARAM_NAMES = ["As", "ns", "H0", "Omega_b", "Omega_m", "w", "wa", "Tagn", "mnu"]
AS_IDX = PARAM_NAMES.index("As")
NS_IDX = PARAM_NAMES.index("ns")
H0_IDX = PARAM_NAMES.index("H0")
OB_IDX = PARAM_NAMES.index("Omega_b")
OM_IDX = PARAM_NAMES.index("Omega_m")
W0_IDX = PARAM_NAMES.index("w")
WA_IDX = PARAM_NAMES.index("wa")
EPS = 1e-30

class EmulatorWarning(UserWarning):
    """Warnings emitted by the emulmps PkEmulator."""
    pass
 
 
def _warn(msg: str) -> None:
    """Emit a clean emulator warning that bypasses cobaya's logger."""
    warnings.warn(f"[emulmps] {msg}", EmulatorWarning, stacklevel=3)
 
 
def _is_npce(model_type: Optional[str]) -> bool:
    """Bundle model_type can be 'mlp', 'npce', 'npce_single_z', ... — match on substring."""
    return model_type is not None and "npce" in model_type
 
 
# --- Path Management ---
def _get_project_root() -> Path:
    return Path(__file__).resolve().parent
 
 
ROOT = _get_project_root()
 
try:
    from tensorflow import keras
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
    from symbolic_pofk.linear import plin_emulated, get_approximate_D, growth_correction_R, As_to_sigma8
    from symbolic_pofk.syrenhalofit import run_halofit_vec
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    _warn(f"A required dependency could not be imported: {e.name}. "
          f"If running locally, ensure symbolic_pofk is accessible.")
    _DEPENDENCIES_LOADED = False
 
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass
 
 
# --- CORE EMULATOR CLASS ---
class PkEmulator:
    """
    Cosmology Emulator for the Matter Power Spectrum P(k, z).
 
    Input convention
    ----------------
    All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, wa, T_AGN, mnu].
    The conversion wa -> w0+wa is done internally before every param_scaler call,
    matching the training convention.
 
    Linear P(k)
    -----------
    get_pks() returns P_lin = exp(NN) * P_lin_syren on the full grid
    (K_MODES x Z_MODES).  The network was trained on log(P_lin_CAMB / P_lin_syren).
 
    Nonlinear boost
    ---------------
    The boost network was trained on
 
        log frac = log( B_CAMB / B_syren ),   B_CAMB = P_nl_CAMB / P_lin_CAMB,
 
    (train_utils COLASet: boost = pks_target / pks_lin, both from CAMB), with
    B_syren the syren-halofit boost computed from the *syren* linear spectrum.
    get_boost() therefore returns
 
        B_emu = exp(NN) * B_syren          (dimensionless)
 
    and the nonlinear spectrum is P_nl = B_emu * P_lin, with P_lin taken from
    the LINEAR EMULATOR (get_pks).  Never multiply by an absolute P_nl_syren:
    that would replace the emulated P_lin by syren's.
 
    Grid
    ----
    The k grid is read from the metadata bundles ("ks", written by the
    training code since the k <= 50 runs); bundles without it use the original
    500-point grid.  If the linear and boost models were trained on different
    k ranges of the same grid, both are used on their common k range (the
    shorter one).  Z_MODES is fixed.
    """
 
    N_PCS     = 30          # default
    N_K_MODES = 500
 
    K_MODES = DEFAULT_K_MODES = np.logspace(-5.1, 2, N_K_MODES)
    Z_MODES = np.concatenate((
        np.linspace(0, 3,  33, endpoint=False),
        np.linspace(3, 10,  7, endpoint=False),
        np.linspace(10, 50, 12)
    ))
    N_ZS = len(Z_MODES)

    # Values used for optional trailing inputs when the caller omits them
    # (e.g. a mead2020 model that has no T_AGN and was trained at fixed mnu).
    DEFAULT_W0   = -1.0
    DEFAULT_WA   = 0.0
    DEFAULT_TAGN = 7.8
    DEFAULT_MNU  = 0.06
    N_FULL_PARAMS = 9

    def __init__(self,
                 model_file: str = None,
                 metadata_file: str = None,
                 model_type: str = "mlp",
                 base_model_path: str = "models",
                 metadata_path: str = "metadata_w0wacdm",
                 num_batches: int = 10,
                 nl_model_file: str = None,
                 nl_metadata_file: str = None,
                 nl_model_type: str = "mlp"):
        """
        Parameters
        ----------
        model_file, metadata_file : linear Keras model and metadata bundle.
        model_type : 'mlp' or 'npce' (overridden by the bundle's model_type if present).
        nl_model_file, nl_metadata_file : boost Keras model and metadata bundle.
        nl_model_type : 'mlp' or 'npce' (overridden by the bundle if present).
        """
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")
 
        self.MODEL_DIR    = ROOT / base_model_path
        self.METADATA_DIR = ROOT / metadata_path
        self.NUM_BATCHES  = num_batches
 
        resolved_model_path    = Path(model_file)    if model_file    is not None else self.MODEL_DIR / "emulator_w0wacdm.h5"
        resolved_metadata_path = Path(metadata_file) if metadata_file is not None else self.METADATA_DIR / "metadata.joblib"
 
        try:
            # -----------------------------------------------------------------
            # Load linear model + metadata
            # -----------------------------------------------------------------
            _bundle           = joblib.load(resolved_metadata_path)
            self._set_k_grid(_bundle.get("ks", None))
            self.param_scaler = _bundle["param_scaler"]
            self.t_comp_pca   = _bundle["t_comp_pca"]
 
            self._params_buf = np.empty((1, 9), dtype=np.float32)
            self._a_array    = (1.0 / (self.Z_MODES + 1)).astype(np.float32)
            self._a_array64  = 1.0 / (self.Z_MODES + 1.0)   # for the syren boost
 
            self.model = keras.models.load_model(
                resolved_model_path,
                custom_objects={"CustomActivationLayer": CustomActivationLayer},
                compile=False,
            )
 
            # Model type: bundle > input-shape inference > argument
            bundle_model_type = _bundle.get("model_type", None)
            if bundle_model_type is not None:
                self.model_type = bundle_model_type
            else:
                n_inputs = self.model.input_shape[-1]
                n_params = self.param_scaler.n_features_in_
                self.model_type = "mlp" if n_inputs == n_params else "npce"
                if self.model_type != model_type:
                    _warn(f"'model_type' not in bundle; inferred '{self.model_type}' from the "
                          f"model input shape ({n_inputs} inputs), overriding '{model_type}'.")
 
            self.t_comp_scaler = _bundle.get("t_comp_scaler", None)
 
            self._pce_indices = _bundle.get("npce_indices", None)
            if _is_npce(self.model_type) and self._pce_indices is None:
                raise ValueError(
                    f"Linear model_type='{self.model_type}' but bundle['npce_indices'] is None. "
                    "Please (re)package your metadata."
                )
 
            self._metadata_bundle = _bundle
 
            @tf.function(jit_compile=False)
            def compiled_inference(x):
                return self.model(x, training=False)
            self._compiled_inference = compiled_inference
 
            self.PCAS:    Dict[float, PCA]            = {}
            self.SCALERS: Dict[float, StandardScaler] = {}
            self._pcas_loaded               = False
            self.INVERSE_TRANSFORM_MATRICES = None
            self.INVERSE_TRANSFORM_OFFSETS  = None
            self._load_pcas_and_scalers()
 
            # -----------------------------------------------------------------
            # Load nonlinear boost model + metadata (optional)
            # -----------------------------------------------------------------
            self._nl_model          = None
            self._nl_compiled       = None
            self._nl_param_scaler   = None
            self._nl_t_comp_pca     = None
            self._nl_t_comp_scaler  = None
            self._nl_pce_indices    = None
            self._nl_model_type     = None
            self._nl_n_pcs          = None
            self.NL_INVERSE_TRANSFORM_MATRICES = None
            self.NL_INVERSE_TRANSFORM_OFFSETS  = None
 
            if nl_model_file is not None and nl_metadata_file is not None:
                self._load_nl_model(nl_model_file, nl_metadata_file, nl_model_type)
 
            # -----------------------------------------------------------------
            # Baseline model setup (disabled)
            # -----------------------------------------------------------------
            self._baseline = None
 
            # -----------------------------------------------------------------
            # Warm up linear model
            # -----------------------------------------------------------------
            n_feat = self.param_scaler.n_features_in_
            dummy_params = self.param_scaler.transform(
                np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0, 7.8, 0.06]],
                         dtype=np.float32)[:, :n_feat])
            dummy_input = self._make_network_input(dummy_params)
            dummy_tf    = tf.constant(dummy_input, dtype=tf.float32)
            _ = self._compiled_inference(dummy_tf)
            _ = self._compiled_inference(dummy_tf)
 
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[emulmps] Required model or metadata file not found: {e.filename}\n"
                "Ensure 'models' and 'metadata' directories are correctly placed."
            ) from e
 
    # -------------------------------------------------------------------------
    # Nonlinear model loading
    # -------------------------------------------------------------------------
 
    def _load_nl_model(
        self,
        nl_model_file: str,
        nl_metadata_file: str,
        nl_model_type: str,
    ) -> None:
        """
        Load the nonlinear boost model and its metadata bundle (same full
        K_MODES x Z_MODES grid as the linear model).
        """
        nl_model_path    = Path(nl_model_file)
        nl_metadata_path = Path(nl_metadata_file)
 
        nl_bundle = joblib.load(nl_metadata_path)
 
        self._nl_param_scaler  = nl_bundle["param_scaler"]
        self._nl_t_comp_pca    = nl_bundle.get("t_comp_pca", None)
        self._nl_t_comp_scaler = nl_bundle.get("t_comp_scaler", None)
        if self._nl_t_comp_scaler is None:
            _warn("nl_metadata: t_comp_scaler not found — assuming raw t-components.")
 
        self._nl_model = keras.models.load_model(
            nl_model_path,
            custom_objects={"CustomActivationLayer": CustomActivationLayer},
            compile=False,
        )
 
        # Model type: bundle > input-shape inference (of the NL model) > argument
        nl_bundle_model_type = nl_bundle.get("model_type", None)
        if nl_bundle_model_type is not None:
            self._nl_model_type = nl_bundle_model_type
        else:
            n_inputs = self._nl_model.input_shape[-1]
            n_params = self._nl_param_scaler.n_features_in_
            self._nl_model_type = "mlp" if n_inputs == n_params else "npce"
            if self._nl_model_type != nl_model_type:
                _warn(f"'model_type' not in nonlinear bundle; inferred '{self._nl_model_type}' "
                      f"from the NL model input shape ({n_inputs} inputs), "
                      f"overriding '{nl_model_type}'.")
 
        self._nl_pce_indices = nl_bundle.get("npce_indices", None)
        if _is_npce(self._nl_model_type) and self._nl_pce_indices is None:
            raise ValueError(
                f"NL model_type='{self._nl_model_type}' but nl_bundle['npce_indices'] is None."
            )
 
        @tf.function(jit_compile=False)
        def nl_compiled_inference(x):
            return self._nl_model(x, training=False)
        self._nl_compiled = nl_compiled_inference
 
        # ---------------------------------------------------------------
        # Per-z PCAs / scalers -> fused inverse transforms (full grid)
        # ---------------------------------------------------------------
        nl_pcas    = nl_bundle["pcas"]
        nl_scalers = nl_bundle["scalers"]
 
        inverse_matrices, inverse_offsets = [], []
        for z in self.Z_MODES:
            z_key = float(f"{z:.3f}")
            if z_key not in nl_pcas:
                raise ValueError(
                    f"NL metadata has no PCA for z={z_key}: the boost model must cover all "
                    f"{self.N_ZS} Z_MODES (bundle has {len(nl_pcas)} redshifts)."
                )
            pca, scaler = nl_pcas[z_key], nl_scalers[z_key]
            if hasattr(scaler, "scale_"):
                scale, mean = scaler.scale_, scaler.mean_
            elif hasattr(scaler, "std"):
                scale, mean = scaler.std, scaler.mean
            else:
                raise AttributeError(f"NL scaler for z={z:.3f} has neither 'scale_' nor 'std'.")
            inverse_matrices.append(pca.components_ * scale[None, :])
            inverse_offsets.append(pca.mean_ * scale + mean)
 
        self._nl_n_pcs = next(iter(nl_pcas.values())).n_components_
 
        self.NL_INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
        self.NL_INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)
 
        nl_ks = nl_bundle.get("ks", None)
        nl_ks = self.DEFAULT_K_MODES if nl_ks is None else np.asarray(nl_ks, dtype=np.float64)
        if self.NL_INVERSE_TRANSFORM_MATRICES.shape[-1] != len(nl_ks):
            raise ValueError(
                f"NL PCA has {self.NL_INVERSE_TRANSFORM_MATRICES.shape[-1]} k-points but its "
                f"bundle's k grid has {len(nl_ks)}."
            )
        n = min(len(nl_ks), self.N_K_MODES)
        if not np.allclose(nl_ks[:n], self.K_MODES[:n], rtol=1e-10, atol=0):
            raise ValueError("The linear and boost models were trained on different k grids.")
        if n < self.N_K_MODES:
            _warn(f"Boost model covers k <= {nl_ks[n - 1]:.4g} 1/Mpc, linear model "
                  f"k <= {self.K_MODES[-1]:.4g}: using both on the common range.")
            self._truncate_k(n)
        if n < len(nl_ks):
            if n == self.N_K_MODES and n < len(nl_ks):
                _warn(f"Linear model covers k <= {self.K_MODES[-1]:.4g} 1/Mpc, boost model "
                      f"k <= {nl_ks[-1]:.4g}: using both on the common range.")
            self.NL_INVERSE_TRANSFORM_MATRICES = np.ascontiguousarray(self.NL_INVERSE_TRANSFORM_MATRICES[..., :n])
            self.NL_INVERSE_TRANSFORM_OFFSETS  = np.ascontiguousarray(self.NL_INVERSE_TRANSFORM_OFFSETS[..., :n])
        if self._nl_t_comp_pca is not None:
            n_feat = getattr(self._nl_t_comp_pca, "n_features_in_", None)
            if n_feat is not None and n_feat != self.N_ZS * self._nl_n_pcs:
                raise ValueError(
                    f"NL tPCA expects {n_feat} features but the bundle has "
                    f"{self.N_ZS} z x {self._nl_n_pcs} PCs."
                )
 
        n_feat = self._nl_param_scaler.n_features_in_
        dummy_norm = self._nl_param_scaler.transform(
            np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0, 7.8, 0.06]],
                     dtype=np.float32)[:, :n_feat]
        )
        dummy_tf = tf.constant(self._make_nl_network_input(dummy_norm), dtype=tf.float32)
        _ = self._nl_compiled(dummy_tf)
        _ = self._nl_compiled(dummy_tf)
 
        _warn(f"Nonlinear boost model loaded from {nl_model_path}")
 
    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------
 
    def _fill_params(self, params) -> np.ndarray:
        """
        Pad a parameter vector to the full 9-column layout
        [As, ns, H0, Ob, Om, w0, wa, T_AGN, mnu].

        Accepts 5 (LCDM) to 9 entries; missing trailing entries take the class
        defaults (w0=-1, wa=0, T_AGN=DEFAULT_TAGN, mnu=DEFAULT_MNU).  Each
        network only reads its first n_features_in_ columns, so the padded
        T_AGN / mnu never reach a model that was not trained on them; mnu
        does always enter the syren baseline.
        """
        p = np.asarray(params, dtype=np.float64).ravel()
        if not 5 <= p.size <= self.N_FULL_PARAMS:
            raise ValueError(
                f"[emulmps] expected 5-{self.N_FULL_PARAMS} parameters "
                f"[As, ns, H0, Ob, Om, (w0, wa, T_AGN, mnu)], got {p.size}."
            )
        defaults = [self.DEFAULT_W0, self.DEFAULT_WA, self.DEFAULT_TAGN, self.DEFAULT_MNU]
        return np.concatenate([p, defaults[p.size - 5:]])

    @property
    def n_lin_inputs(self) -> int:
        """Number of leading parameter columns the linear network reads."""
        return int(self.param_scaler.n_features_in_)

    @property
    def n_nl_inputs(self) -> int:
        """Number of leading parameter columns the boost network reads (0 if none)."""
        if self._nl_param_scaler is None:
            return 0
        return int(self._nl_param_scaler.n_features_in_)

    def _set_k_grid(self, ks) -> None:
        """k grid from the linear bundle (instance attribute; the class default
        is the original 500-point grid)."""
        self.K_MODES   = self.DEFAULT_K_MODES if ks is None else np.asarray(ks, dtype=np.float64)
        self.N_K_MODES = len(self.K_MODES)
        if ks is not None and self.N_K_MODES != len(self.DEFAULT_K_MODES):
            _warn(f"k grid from metadata: {self.N_K_MODES} modes, "
                  f"{self.K_MODES[0]:.3g} <= k <= {self.K_MODES[-1]:.4g} 1/Mpc.")
 
    def _truncate_k(self, n: int) -> None:
        """Keep the first n k modes for the linear model (common range with the boost)."""
        self.K_MODES   = self.K_MODES[:n]
        self.N_K_MODES = n
        self.INVERSE_TRANSFORM_MATRICES = np.ascontiguousarray(self.INVERSE_TRANSFORM_MATRICES[..., :n])
        self.INVERSE_TRANSFORM_OFFSETS  = np.ascontiguousarray(self.INVERSE_TRANSFORM_OFFSETS[..., :n])
 
    def _load_pcas_and_scalers(self):
        if self._pcas_loaded:
            return
 
        try:
            self.PCAS    = self._metadata_bundle["pcas"]
            self.SCALERS = self._metadata_bundle["scalers"]
 
            # N_PCS from the stored PCAs rather than the class default
            self.N_PCS = next(iter(self.PCAS.values())).n_components_
 
            inverse_matrices, inverse_offsets = [], []
            for z in self.Z_MODES:
                z_key  = float(f"{z:.3f}")
                pca    = self.PCAS[z_key]
                scaler = self.SCALERS[z_key]
 
                if hasattr(scaler, 'scale_'):
                    scale, mean = scaler.scale_, scaler.mean_
                elif hasattr(scaler, 'std'):
                    scale, mean = scaler.std, scaler.mean
                else:
                    raise AttributeError(
                        f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
                    )
 
                inverse_matrices.append(pca.components_ * scale[None, :])
                inverse_offsets.append(pca.mean_ * scale + mean)
 
            self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
            self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)
 
            if self.INVERSE_TRANSFORM_MATRICES.shape[-1] != self.N_K_MODES:
                raise ValueError(
                    f"Linear PCA has {self.INVERSE_TRANSFORM_MATRICES.shape[-1]} k-points; "
                    f"its bundle's k grid has {self.N_K_MODES}."
                )
            self._pcas_loaded = True
 
        except (KeyError, FileNotFoundError) as e:
            raise RuntimeError(
                f"[emulmps] Could not load linear metadata from bundle (the linear model must "
                f"cover all {self.N_ZS} Z_MODES): {e}"
            ) from e
 
    def _evaluate_pce_basis(self, X_norm: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Evaluate a PCE multi-index basis at X_norm."""
        X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
        N, D      = X_clipped.shape
        max_deg   = int(indices.max()) if indices.size > 0 else 0
 
        pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
        pow_table[:, 0, :] = 1.0
        if max_deg >= 1:
            pow_table[:, 1, :] = X_clipped.T
        for p in range(2, max_deg + 1):
            pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T
 
        d_idx   = np.arange(D)[np.newaxis, :]
        powered = pow_table[d_idx, indices, :]
        return powered.prod(axis=1).T.astype(np.float32)
 
    def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
        """Build linear-model network input (MLP: pass-through; NPCE: PCE basis)."""
        if not _is_npce(self.model_type):
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm, self._pce_indices)
 
    def _make_nl_network_input(self, params_norm: np.ndarray) -> np.ndarray:
        """Build NL-model network input."""
        if not _is_npce(self._nl_model_type):
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm, self._nl_pce_indices)
 
    def _norm_params_for_scaler(self, params_wa: np.ndarray, scaler) -> np.ndarray:
        """
        Convert [As, ns, H0, Ob, Om, w0, wa, ...] -> [..., w0+wa, ...] then apply
        scaler.transform().  Both models were trained with w0+wa in column 6.
        """
        buf = params_wa.copy().reshape(1, -1)
        buf[0, 6] = buf[0, 5] + buf[0, 6]   # wa -> w0+wa
        return scaler.transform(buf[:, :scaler.n_features_in_])
 
    # -------------------------------------------------------------------------
    # Linear P(k) inference
    # -------------------------------------------------------------------------
 
    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
        """Symbolic P_lin(k, z) approximation (syren) on the full grid, Mpc^3. wa in col 6."""
        p = np.asarray(params, dtype=np.float64)
        As, ns, H0_in, Ob, Om, w0, wa = p[:7]
        h   = H0_in / 100.0
        mnu = p[8] if len(p) > 8 else self.DEFAULT_MNU
        k_h = self.K_MODES / h
        a   = self._a_array64
 
        pk_fid = plin_emulated(k_h, Om, Ob, h, ns, As=As, w0=w0, wa=wa, mnu=mnu)
 
        kref = 1e-4
        D0 = get_approximate_D(k=kref, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=kref, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a)
 
        growth = (Dz / D0) ** 2 * (Rz / R0)
        return ((pk_fid * growth[:, None]) / h**3).astype(np.float32)
 
    def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
        """Linear model: normalised params -> log-frac for all z (full grid)."""
        net_input    = self._make_network_input(params_norm)
        t_comps_norm = self._compiled_inference(
            tf.constant(net_input, dtype=tf.float32)).numpy()
 
        if self.t_comp_scaler is not None:
            t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm
 
        pcs_flat = (self.t_comp_pca.inverse_transform(t_comps_raw)
                    if self.t_comp_pca is not None else t_comps_raw)
        pcs_z = pcs_flat.reshape(self.N_ZS, self.N_PCS)
 
        return (np.einsum('zp,zpk->zk',
                          pcs_z.astype(np.float32),
                          self.INVERSE_TRANSFORM_MATRICES,
                          optimize=True)
                + self.INVERSE_TRANSFORM_OFFSETS)
 
    def get_pks(
        self,
        params: List[float],
        use_syren: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return linear P(k, z) for a single cosmology on the full grid.
 
        params    : [10^9 A_s, ns, H0, Ob, Om, w0, wa, T_AGN, mnu] — wa in col 6.
                    Trailing entries may be omitted (see _fill_params).
        use_syren : if True, skip the NN and return the syren approximation only.
 
        Returns (K_MODES, Z_MODES, P_lin) with P_lin in Mpc^3, shape (N_ZS, N_K_MODES).
        """
        buf = self._params_buf
        buf[0, :] = self._fill_params(params)   # wa stays in col 6 for the approximation
 
        if self._baseline is not None:
            _, _, pk_mps = self._baseline.get_pks(buf[0, :7])
        else:
            pk_mps = self._compute_mps_approximation(buf[0])
 
        if use_syren is True:
            return self.K_MODES, self.Z_MODES, pk_mps
 
        x_norm = self._norm_params_for_scaler(buf, self.param_scaler)
        pks = (np.exp(self._predict_fracs_all_z(x_norm)) * pk_mps).astype(np.float32)
        return self.K_MODES, self.Z_MODES, pks
 
    # -------------------------------------------------------------------------
    # Nonlinear boost inference
    # -------------------------------------------------------------------------
 
    def _compute_syren_boost_nl(self, params: np.ndarray) -> np.ndarray:
        """
        B_syren = P_nl_syren / P_lin_syren on the full grid (Z_MODES x K_MODES).
 
        Mirrors the training denominator (train_utils _compute_mps_nl_approximation_parallel):
        syren linear P(k) on the same k grid, evolved with the syren growth
        approximations, then syren halofit with return_boost=True.  The input
        P_lin is syren's — NOT the emulated P_lin — because that is how the
        training target was defined.  wa in column 6.
        """
        p = np.asarray(params, dtype=np.float64)
        As, ns, H0_in, Ob, Om, w0, wa = p[:7]
        h   = H0_in / 100.0
        mnu = p[8] if len(p) > 8 else self.DEFAULT_MNU
        k_h = self.K_MODES / h
        a   = self._a_array64
 
        pk_lin_h = plin_emulated(k_h, Om, Ob, h, ns, As=As, mnu=mnu, w0=w0, wa=wa)   # (Mpc/h)^3
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a)
        pk_lin_h_z = pk_lin_h[None, :] * ((Dz / D0) ** 2 * (Rz / R0))[:, None]   # (N_ZS, N_K_MODES)
 
        sigma8 = As_to_sigma8(As, Om, Ob, h, ns, mnu=mnu, w0=w0, wa=wa)
        boost = run_halofit_vec(k_h, sigma8, Om, Ob, h, ns, a,
                                return_boost=True, Plin_in=pk_lin_h_z)
        return np.asarray(boost, dtype=np.float64).reshape(self.N_ZS, self.N_K_MODES)
 
    def _predict_nl_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
        """NL network: normalised params -> log(B_CAMB / B_syren) for all z (full grid)."""
        net_input    = self._make_nl_network_input(params_norm)
        t_comps_norm = self._nl_compiled(
            tf.constant(net_input, dtype=tf.float32)).numpy()
 
        if self._nl_t_comp_scaler is not None:
            t_comps_raw = self._nl_t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm
 
        pcs_flat = (self._nl_t_comp_pca.inverse_transform(t_comps_raw)
                    if self._nl_t_comp_pca is not None else t_comps_raw)
        pcs_z = pcs_flat.reshape(self.N_ZS, self._nl_n_pcs)
 
        return (np.einsum('zp,zpk->zk',
                          pcs_z.astype(np.float32),
                          self.NL_INVERSE_TRANSFORM_MATRICES,
                          optimize=True)
                + self.NL_INVERSE_TRANSFORM_OFFSETS)
 
    def get_boost(self, params, pk_lin=None, use_syren=None):
        """
        Return the dimensionless nonlinear boost B(k, z) = P_nl / P_lin on the
        full grid.  Multiply by the LINEAR EMULATOR output (get_pks) to get P_nl.
 
        params    : [10^9 A_s, ns, H0, Ob, Om, w0, wa, T_AGN, mnu] — wa in col 6.
                    Trailing entries may be omitted (see _fill_params).
        pk_lin    : unused (kept for API compatibility); the boost is
                    dimensionless and independent of the P_lin it is applied to.
        use_syren : if True, return the syren-halofit boost only (no network).
        """
        if self._nl_model is None:
            raise RuntimeError("No nonlinear boost model loaded.")
 
        buf = self._params_buf
        buf[0, :] = self._fill_params(params)
 
        b_syren = self._compute_syren_boost_nl(buf[0])
        log_b   = np.log(b_syren)
 
        if use_syren is not True:
            x_norm = self._norm_params_for_scaler(buf, self._nl_param_scaler)
            log_b  = log_b + self._predict_nl_fracs_all_z(x_norm)   # log(B_CAMB) estimate
 
        return self.K_MODES, self.Z_MODES, np.exp(log_b).astype(np.float32)
 
    def has_nl_model(self) -> bool:
        """Return True if a nonlinear boost model has been loaded."""
        return self._nl_model is not None
 
 
# --- Public Module-Level Interface ---
 
_pk_emulator_instance = None
if _DEPENDENCIES_LOADED:
    try:
        _pk_emulator_instance = PkEmulator()
    except Exception as e:
        _warn(f"Module-level singleton not loaded "
              f"(expected if using explicit model_file/metadata_file): {e}")
 
 
def get_pks(params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level convenience function for linear P(k, z) using the default
    model files.  Prefer constructing PkEmulator(...) with explicit paths.
    """
    if _pk_emulator_instance is None:
        raise RuntimeError("Default PkEmulator is not available; construct PkEmulator(...) "
                           "with explicit model_file / metadata_file instead.")
    return _pk_emulator_instance.get_pks(params, use_syren=use_syren)