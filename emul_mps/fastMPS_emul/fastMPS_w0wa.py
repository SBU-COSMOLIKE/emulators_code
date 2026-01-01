# Author: Victoria Lloyd (2025)
import os
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
import sys
from . import train_utils_pk_emulator as utils
sys.modules['train_utils_pk_emulator'] = utils
# import train_utils_pk_emulator_w0wa_weighted as utils
from . train_utils_pk_emulator import CustomActivationLayer
from keras.losses import MeanSquaredError


# Set up logging for the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Path Management ---
def _get_project_root() -> Path:
    """Returns the root directory Path object, relative to this file."""
    return Path(__file__).resolve().parent

# --- Dependency Guards ---
ROOT = _get_project_root()
try:

    from tensorflow import keras
    from colossus.cosmology import cosmology as Cosmo
    # Issues installing symbolic_pofk as a package, including a local copy in the emulator
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk"); from symbolic_pofk.linear import As_to_sigma8, plin_emulated
    
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    logging.error(f"FATAL ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
    logging.error(f"Missing component: {e.name}")
    logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
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
    Encapsulates models and handles I/O robustly.
    """

    # --- Configuration Constants (Class Attributes) ---
    N_PCS = 22          # Number of k-space PCs per redshift
    N_K_MODES = 2400    # Number of k-modes in the output spectrum
    K_MODES = np.logspace(-5, 2, N_K_MODES)

    Z_MODES = np.concatenate((
        np.linspace(0, 2, 100, endpoint=False),
        np.linspace(2, 10, 10, endpoint=False),
        np.linspace(10, 50, 12)
    ))
    N_ZS = len(Z_MODES)
    
    def __init__(self, base_model_path: str = "models", metadata_path: str = "metadata", num_batches: int = 10):
        """Initializes the emulator by loading all necessary models and metadata."""
        
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

        logging.info("[PkEmulator] Initializing and loading models...")
        
        # Paths are relative to the package root
        self.MODEL_DIR = ROOT / base_model_path
        self.METADATA_DIR = ROOT / metadata_path
        self.NUM_BATCHES = num_batches
        
        # Load the core components using Path objects
        try:
            self.param_scaler = joblib.load(self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches")
            self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")
            self.model = keras.models.load_model(
                self.MODEL_DIR / f"emulator.h5",
                custom_objects={
                    "CustomActivationLayer": CustomActivationLayer,
                    "mse": MeanSquaredError()
                },

            )

            # Load the 122 PCA and Scaler objects
            self.PCAS: Dict[float, PCA] = {}
            self.SCALERS: Dict[float, StandardScaler] = {}
            for z in self.Z_MODES:
                z_key = float(f"{z:.3f}")
                self.PCAS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
                self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

            logging.info("[PkEmulator] All models and metadata loaded successfully.")

        except FileNotFoundError as e:
            logging.error(f"CRITICAL: Required model or metadata file not found: {e.filename}")
            logging.warning("Please ensure the 'models' and 'metadata' directories are correctly placed relative to pk_emulator.py.")
            raise
    
    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
        """
        Computes the analytical P_lin(k, z) approximation.
        
        This function uses Colossus to calculate the growth factors.
        
        Args:
            params: 1D array of cosmological parameters [10^9 A_s, ns, H0, Ob, Om].
            
        Returns:
            np.ndarray: P_lin(k, z) array of shape (N_ZS, N_K_MODES).
        """
        # Destructure parameters (clearer than indexing)
        As, ns, H0_in, Ob, Om, w0, wa = params
        h = H0_in / 100.0 # Convert H0 to h

        # Compute dependent parameter (sigma8)
        sigma8 = As_to_sigma8(As, Om, Ob, h, ns)
        
        # Compute fiducial P(k) at z=0
        pk_fid = plin_emulated(
            self.K_MODES, sigma8, Om, Ob, h, ns,
            emulator='fiducial', extrapolate=False,
            kmin=self.K_MODES.min(), kmax=self.K_MODES.max()
        )
        
        # Compute growth factor using Colossus
        cosmo = Cosmo.setCosmology('tmp_cosmo', {
            'flat': True, 'H0': H0_in, 'Om0': Om,
            'Ob0': Ob, 'sigma8': sigma8, 'ns': ns},
            persistence=''
        )

        D0 = cosmo.growthFactor(0.0)
        Dz = cosmo.growthFactor(self.Z_MODES)
        growth_factors = (Dz / D0) ** 2
        
        # Apply growth factors and return
        # pk_fid[None, :] ensures broadcasting works cleanly: (1, K) * (Z, 1) = (Z, K)
        return pk_fid[None, :] * growth_factors[:, None]


    def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
        """
        Performs the full NN prediction and PCA reconstructions.
        
        Args:
            params: Normalized 2D array of parameters (1, N_params).
            
        Returns:
            np.ndarray: Final predicted log fractional differences, shape (N_ZS, N_K_MODES).
        """
        # Predict T-components
        t_comps_pred = self.model.predict(params, verbose=0)
        
        # Inverse T-components to flat k-PCs (shape 1, N_ZS * N_PCS)
        pcs_flat = self.t_comp_pca.inverse_transform(t_comps_pred)
        
        # Reshape to individual redshifts (shape N_ZS, N_PCS)
        # We drop the single batch dimension (axis 0) since we only have one cosmology
        pcs_pred_z_stack = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        
        # Inverse K-PCA and Scaler Loop
        reconstructed_fracs = [
            self.SCALERS[float(f"{z:.3f}")].inverse_transform(
                self.PCAS[float(f"{z:.3f}")].inverse_transform(
                    pcs_z.reshape(1, -1)
                )
            )[0] # Extract the 1D result from the (1, 2000) array
            for z, pcs_z in zip(self.Z_MODES, pcs_pred_z_stack)
        ]

        # Concatenate and return
        # Resulting shape: (N_ZS, N_K_MODES)
        return np.stack(reconstructed_fracs)


    def get_pks(self, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns P(k, z) for all emulator redshifts for a given cosmology.
        
        Args:
            params: List or 1D array of 5 cosmological parameters.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
        """
        # Normalize cosmological parameters
        # Ensure input is a 2D array (1, N_params) for the scaler/model
        x_norm = self.param_scaler.transform(np.array(params).reshape(1, -1))
        
        # Generate predicted fractional differences (shape N_ZS, N_K_MODES)
        frac = self._predict_fracs_all_z(x_norm)

        # Compute MPS approximation (shape N_ZS, N_K_MODES)
        pk_mps = self._compute_mps_approximation(np.array(params))

        # Full emulated P(k, z)
        pks = frac * pk_mps

        # Return the k-modes, z-modes, and the P(k,z) array
        return self.K_MODES, self.Z_MODES, pks


# --- Public Module-Level Interface ---

# Instantiate the emulator once when the module is imported.
_pk_emulator_instance = None
if _DEPENDENCIES_LOADED:
    try:
        _pk_emulator_instance = PkEmulator()
    except Exception as e:
        logging.warning(f"PkEmulator instance failed during initialization: {e}")

def get_pks(params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level function to get P(k, z). This is the streamlined public interface.
    
    It calls the get_pks method of the globally-instantiated PkEmulator object.
    
    Args:
        params: List or 1D array of 5 cosmological parameters, [10^9 A_s, ns, H0, Ob, Om].
            
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
    """
    if _pk_emulator_instance is None:
        raise RuntimeError("PkEmulator is not loaded. Check prior error messages regarding dependencies or files.")
        
    return _pk_emulator_instance.get_pks(params)



# Example Usage:
# import fastMPS
# k_modes, z_modes, pks_pred = pk_emulator.get_pks([3.0e-9, 0.965, 67.5, 0.048, 0.31])

# # Author: Victoria Lloyd (2025)
# import os
# import numpy as np
# import joblib
# from typing import Dict, Any, List, Tuple
# import logging
# from pathlib import Path
# import sys
# from train_utils_pk_emulator_w0wa_weighted import CustomActivationLayer
# from keras.losses import MeanSquaredError

# # Set up logging for the module
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # --- Path Management ---
# def _get_project_root() -> Path:
#     """Returns the root directory Path object, relative to this file."""
#     return Path(__file__).resolve().parent

# # --- Dependency Guards ---
# ROOT = _get_project_root()
# try:
#     from tensorflow import keras
#     from colossus.cosmology import cosmology as Cosmo
#     # Issues installing symbolic_pofk as a package, including a local copy in the emulator
#     sys.path.insert(0, f"{ROOT}/symbolic_pofk")
#     from symbolic_pofk.linear import As_to_sigma8, plin_emulated

#     _DEPENDENCIES_LOADED = True
# except ImportError as e:
#     logging.error(f"FATAL ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
#     logging.error(f"Missing component: {e.name if hasattr(e, 'name') else e}")
#     logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
#     _DEPENDENCIES_LOADED = False

# try:
#     from sklearn.decomposition import PCA 
#     from sklearn.preprocessing import StandardScaler
#     import tensorflow as tf
# except ImportError:
#     pass

# # Import utils for custom layer
# try:
#     import train_utils_pk_emulator_w0wa_weighted as utils
# except ImportError:
#     logging.warning("Could not import train_utils_pk_emulator_w0wa_weighted")
#     utils = None

# # Load eigvals for loss function
# t_eigvals = np.load(
#     ROOT / "metadata" / "t_components_eigvals.npy"
# ).astype(np.float32)
# weights = tf.constant(1.0 / t_eigvals, dtype=tf.float32)
# weights = weights / tf.reduce_mean(weights)

# def weighted_t_mse(weights):
#     def loss(y_true, y_pred):
#         diff = y_pred - y_true
#         return tf.reduce_mean(diff**2 * weights)
#     return loss

# loss_fn = weighted_t_mse(weights)

# # --- CORE EMULATOR CLASS ---
# class PkEmulator:
#     """
#     Cosmology Emulator for the Matter Power Spectrum P(k, z).
#     Encapsulates models and handles I/O robustly.
#     """

#     # --- Configuration Constants (Class Attributes) ---
#     N_PCS = 22          # Number of k-space PCs per redshift
#     N_K_MODES = 2400    # Number of k-modes in the output spectrum
#     K_MODES = np.logspace(-5, 2, N_K_MODES)

#     Z_MODES = np.concatenate((
#         np.linspace(0, 2, 100, endpoint=False),
#         np.linspace(2, 10, 10, endpoint=False),
#         np.linspace(10, 50, 12)
#     ))
#     N_ZS = len(Z_MODES)
    
#     def __init__(self, base_model_path: str = "models", metadata_path: str = "metadata", num_batches: int = 10):
#         """Initializes the emulator by loading all necessary models and metadata."""
        
#         if not _DEPENDENCIES_LOADED:
#             raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

#         logging.info("[PkEmulator] Initializing and loading models...")
        
#         # Paths are relative to the package root
#         self.MODEL_DIR = ROOT / base_model_path
#         self.METADATA_DIR = ROOT / metadata_path
#         self.NUM_BATCHES = num_batches
        
#         # Load the core components using Path objects
#         try:
#             self.param_scaler = joblib.load(self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches")
#             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")
#             self.t_scaler = joblib.load(
#                 self.METADATA_DIR / "t_components_scaler_lowk"
#             )
            
#             # Build model architecture (compatible with Keras 2.x)
#             logging.info("[PkEmulator] Building model architecture...")
#             # self.model = self._build_model_architecture()
            
#             # # Load weights
#             # weights_path = self.MODEL_DIR / "model_weights.weights.h5"
#             # logging.info(f"[PkEmulator] Loading weights from: {weights_path}")
#             # self.model.load_weights(str(weights_path))
#             # logging.info("[PkEmulator] ✓ Model weights loaded successfully")
#             self.model = keras.models.load_model(
#                 self.MODEL_DIR / f"emulator.h5",
#                 custom_objects={
#                     "CustomActivationLayer": CustomActivationLayer,
#                     "mse": MeanSquaredError()
#                 },

#             )
            
#             # Load the 122 PCA and Scaler objects
#             self.PCAS: Dict[float, PCA] = {}
#             self.SCALERS: Dict[float, StandardScaler] = {}
#             for z in self.Z_MODES:
#                 z_key = float(f"{z:.3f}")
#                 self.PCAS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
#                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")
                
#             logging.info("[PkEmulator] All models and metadata loaded successfully.")

#         except FileNotFoundError as e:
#             logging.error(f"CRITICAL: Required model or metadata file not found: {e.filename}")
#             logging.warning("Please ensure the 'models' and 'metadata' directories are correctly placed relative to fastMPS_w0wa.py.")
#             raise
    
#     def _build_model_architecture(self):
#         """
#         Build the neural network architecture.
        
#         This must exactly match your training architecture!
#         Adjust layers/units if your model is different.
        
#         Returns:
#             keras.Model: Compiled model with random weights
#         """
#         # Input layer - 7 parameters: [As_1e9, ns, H0, Ob, Om, w0, wa]
#         inputs = keras.Input(shape=(7,), name='input_layer')
        
#         # Hidden layers - ADJUST THESE TO MATCH YOUR TRAINING!
#         # Check your training script for the exact architecture
#         x = keras.layers.Dense(256, activation='relu', name='dense_1')(inputs)
#         x = keras.layers.Dropout(0.1, name='dropout_1')(x)
#         x = keras.layers.Dense(512, activation='relu', name='dense_2')(x)
#         x = keras.layers.Dropout(0.1, name='dropout_2')(x)
#         x = keras.layers.Dense(512, activation='relu', name='dense_3')(x)
#         x = keras.layers.Dropout(0.1, name='dropout_3')(x)
#         x = keras.layers.Dense(256, activation='relu', name='dense_4')(x)
#         x = keras.layers.Dropout(0.1, name='dropout_4')(x)
        
#         # Custom activation layer (if you use one)
#         if utils and hasattr(utils, 'CustomActivationLayer'):
#             x = utils.CustomActivationLayer(name='custom_activation')(x)
        
#         # Output layer - 10 t-components (adjust if different)
#         outputs = keras.layers.Dense(10, activation='linear', name='output_layer')(x)
        
#         # Build model
#         model = keras.Model(inputs=inputs, outputs=outputs, name='t_components_model')
        
#         # Compile with loss function (optional but good practice)
#         model.compile(optimizer='adam', loss=loss_fn)
        
#         return model
    
#     def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
#         """
#         Computes the analytical P_lin(k, z) approximation.
        
#         This function uses Colossus to calculate the growth factors.
        
#         Args:
#             params: 1D array of cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
            
#         Returns:
#             np.ndarray: P_lin(k, z) array of shape (N_ZS, N_K_MODES).
#         """
#         # Destructure parameters (clearer than indexing)
#         As, ns, H0_in, Ob, Om, w0, wa = params
#         h = H0_in / 100.0 # Convert H0 to h

#         # Compute dependent parameter (sigma8)
#         sigma8 = As_to_sigma8(As, Om, Ob, h, ns)
        
#         # Compute fiducial P(k) at z=0
#         pk_fid = plin_emulated(
#             self.K_MODES, sigma8, Om, Ob, h, ns,
#             emulator='fiducial', extrapolate=False,
#             kmin=self.K_MODES.min(), kmax=self.K_MODES.max()
#         )
        
#         # Compute growth factor using Colossus
#         cosmo = Cosmo.setCosmology('tmp_cosmo', {
#             'flat': True, 'H0': H0_in, 'Om0': Om,
#             'Ob0': Ob, 'sigma8': sigma8, 'ns': ns,
#             'w0': w0, 'wa': wa},
#             persistence='')

#         D0 = cosmo.growthFactor(0.0)
#         Dz = cosmo.growthFactor(self.Z_MODES)
#         growth_factors = (Dz / D0) ** 2
        
#         # Apply growth factors and return
#         # pk_fid[None, :] ensures broadcasting works cleanly: (1, K) * (Z, 1) = (Z, K)
#         return pk_fid[None, :] * growth_factors[:, None]

#     def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
#         """
#         Performs the full NN prediction and PCA reconstructions.
        
#         Args:
#             params: Normalized 2D array of parameters (1, N_params).
            
#         Returns:
#             np.ndarray: Final predicted log fractional differences, shape (N_ZS, N_K_MODES).
#         """
#         # Predict T-components
#         t_comps_pred_scaled = self.model.predict(params, verbose=0)

#         # Invert NN scaling
#         t_comps_pred = self.t_scaler.inverse_transform(t_comps_pred_scaled)
        
#         # Inverse T-components to flat k-PCs (shape 1, N_ZS * N_PCS)
#         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_pred)
        
#         # Reshape to individual redshifts (shape N_ZS, N_PCS)
#         # We drop the single batch dimension (axis 0) since we only have one cosmology
#         pcs_pred_z_stack = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        
#         # Inverse K-PCA and Scaler Loop
#         reconstructed_fracs = [
#             self.SCALERS[float(f"{z:.3f}")].inverse_transform(
#                 self.PCAS[float(f"{z:.3f}")].inverse_transform(
#                     pcs_z.reshape(1, -1)
#                 )
#             )[0] # Extract the 1D result from the (1, 2400) array
#             for z, pcs_z in zip(self.Z_MODES, pcs_pred_z_stack)
#         ]

#         # Concatenate and return
#         # Resulting shape: (N_ZS, N_K_MODES)
#         return np.stack(reconstructed_fracs)

#     def get_pks(self, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Returns P(k, z) for all emulator redshifts for a given cosmology.
        
#         Args:
#             params: List or 1D array of 7 cosmological parameters [As_1e9, ns, H0, Ob, Om, w0, wa].
            
#         Returns:
#             Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
#         """
#         # Normalize cosmological parameters
#         # Ensure input is a 2D array (1, N_params) for the scaler/model
#         x_norm = self.param_scaler.transform(np.array(params).reshape(1, -1))
        
#         # Generate predicted fractional differences (shape N_ZS, N_K_MODES)
#         frac = self._predict_fracs_all_z(x_norm)

#         # Compute MPS approximation (shape N_ZS, N_K_MODES)
#         pk_mps = self._compute_mps_approximation(np.array(params))

#         # Full emulated P(k, z)
#         pks = frac * pk_mps

#         # Return the k-modes, z-modes, and the P(k,z) array
#         return self.K_MODES, self.Z_MODES, pks


# # --- Public Module-Level Interface ---

# # Instantiate the emulator once when the module is imported.
# _pk_emulator_instance = None
# if _DEPENDENCIES_LOADED:
#     try:
#         _pk_emulator_instance = PkEmulator()
#     except Exception as e:
#         logging.warning(f"PkEmulator instance failed during initialization: {e}")

# def get_pks(params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Module-level function to get P(k, z). This is the streamlined public interface.
    
#     It calls the get_pks method of the globally-instantiated PkEmulator object.
    
#     Args:
#         params: List or 1D array of 7 cosmological parameters:
#                 [As_1e9, ns, H0, Omega_b, Omega_m, w0, wa]
            
#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z))
#             - k_modes: array of k values in h/Mpc, shape (2400,)
#             - z_modes: array of redshifts, shape (122,)
#             - P(k,z): power spectrum in (Mpc/h)^3, shape (122, 2400)
#     """
#     if _pk_emulator_instance is None:
#         raise RuntimeError("PkEmulator is not loaded. Check prior error messages regarding dependencies or files.")
        
#     return _pk_emulator_instance.get_pks(params)


# Example Usage:
# import fastMPS.fastMPS_w0wa as fastMPS_emul
# k_modes, z_modes, pks_pred = fastMPS_emul.get_pks([2.1, 0.965, 67.5, 0.048, 0.31, -1.0, 0.0])







# # Author: Victoria Lloyd (2025)
# import os
# import numpy as np
# import joblib
# from typing import Dict, Any, List, Tuple
# import logging
# from pathlib import Path
# import keras
# import sys

# ## Alias keras.src → keras for deserialization compatibility
# #if not hasattr(keras, "src"):
# #    sys.modules["keras.src"] = keras
# #    sys.modules["keras.src.models"] = keras.models
# #    sys.modules["keras.src.models.functional"] = keras.models
# from . import train_utils_pk_emulator_w0wa_weighted as utils


# # Set up logging for the module
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # --- Path Management ---
# def _get_project_root() -> Path:
#     """Returns the root directory Path object, relative to this file."""
#     return Path(__file__).resolve().parent

# # --- Dependency Guards ---
# ROOT = _get_project_root()
# try:

#     from tensorflow import keras
#     from colossus.cosmology import cosmology as Cosmo
#     # Issues installing symbolic_pofk as a package, including a local copy in the emulator
#     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk"); from symbolic_pofk.linear import As_to_sigma8, plin_emulated

#     _DEPENDENCIES_LOADED = True
# except ImportError as e:
#     logging.error(f"FATAL ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
#     logging.error(f"Missing component: {e.name}")
#     logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
#     _DEPENDENCIES_LOADED = False

# try:
#     from sklearn.decomposition import PCA 
#     from sklearn.preprocessing import StandardScaler
#     import tensorflow as tf
# except ImportError:
#     pass

# t_eigvals = np.load(
#     ROOT / "metadata" / "t_components_eigvals.npy"
# ).astype(np.float32)
# weights = tf.constant(1.0 / t_eigvals, dtype=tf.float32)
# weights = weights / tf.reduce_mean(weights)

# def weighted_t_mse(weights):
#     def loss(y_true, y_pred):
#         diff = y_pred - y_true
#         return tf.reduce_mean(diff**2 * weights)
#     return loss

# loss_fn = weighted_t_mse(weights)

# # --- CORE EMULATOR CLASS ---
# class PkEmulator:
#     """
#     Cosmology Emulator for the Matter Power Spectrum P(k, z).
#     Encapsulates models and handles I/O robustly.
#     """

#     # --- Configuration Constants (Class Attributes) ---
#     N_PCS = 22          # Number of k-space PCs per redshift
#     N_K_MODES = 2400    # Number of k-modes in the output spectrum
#     K_MODES = np.logspace(-5, 2, N_K_MODES)

#     Z_MODES = np.concatenate((
#         np.linspace(0, 2, 100, endpoint=False),
#         np.linspace(2, 10, 10, endpoint=False),
#         np.linspace(10, 50, 12)
#     ))
#     N_ZS = len(Z_MODES)
    
#     def __init__(self, base_model_path: str = "models", metadata_path: str = "metadata", num_batches: int = 10):
#         """Initializes the emulator by loading all necessary models and metadata."""
        
#         if not _DEPENDENCIES_LOADED:
#             raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

#         logging.info("[PkEmulator] Initializing and loading models...")
        
#         # Paths are relative to the package root
#         self.MODEL_DIR = ROOT / base_model_path
#         self.METADATA_DIR = ROOT / metadata_path
#         self.NUM_BATCHES = num_batches
        
#         # Load the core components using Path objects
#         try:
#             self.param_scaler = joblib.load(self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches")
#             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")
#             self.t_scaler = joblib.load(
#                 self.METADATA_DIR / "t_components_scaler_lowk"
#             )
#             print("NN_t_components_w0wa_weighted_fixed.h5")
#             self.model = keras.models.load_model(
#                 self.MODEL_DIR / f"NN_t_components_w0wa_weighted_fixed.h5",
#                 custom_objects={
#                     "CustomActivationLayer": utils.CustomActivationLayer,
#                     "loss": loss_fn,
#                 },

#             )
            
#             # Load the 122 PCA and Scaler objects
#             self.PCAS: Dict[float, PCA] = {}
#             self.SCALERS: Dict[float, StandardScaler] = {}
#             for z in self.Z_MODES:
#                 z_key = float(f"{z:.3f}")
#                 self.PCAS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
#                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")
                
#             logging.info("[PkEmulator] All models and metadata loaded successfully.")

#         except FileNotFoundError as e:
#             logging.error(f"CRITICAL: Required model or metadata file not found: {e.filename}")
#             logging.warning("Please ensure the 'models' and 'metadata' directories are correctly placed relative to pk_emulator.py.")
#             raise
    
#     def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
#         """
#         Computes the analytical P_lin(k, z) approximation.
        
#         This function uses Colossus to calculate the growth factors.
        
#         Args:
#             params: 1D array of cosmological parameters [10^9 A_s, ns, H0, Ob, Om].
            
#         Returns:
#             np.ndarray: P_lin(k, z) array of shape (N_ZS, N_K_MODES).
#         """
#         # Destructure parameters (clearer than indexing)
#         As, ns, H0_in, Ob, Om, w0, wa = params
#         h = H0_in / 100.0 # Convert H0 to h

#         # Compute dependent parameter (sigma8)
#         sigma8 = As_to_sigma8(As, Om, Ob, h, ns)
        
#         # Compute fiducial P(k) at z=0
#         pk_fid = plin_emulated(
#             self.K_MODES, sigma8, Om, Ob, h, ns,
#             emulator='fiducial', extrapolate=False,
#             kmin=self.K_MODES.min(), kmax=self.K_MODES.max()
#         )
        
#         # Compute growth factor using Colossus
#         cosmo = Cosmo.setCosmology('tmp_cosmo', {
#             'flat': True, 'H0': H0_in, 'Om0': Om,
#             'Ob0': Ob, 'sigma8': sigma8, 'ns': ns,
#             'w0': w0, 'wa':wa},
#             persistence='')

#         D0 = cosmo.growthFactor(0.0)
#         Dz = cosmo.growthFactor(self.Z_MODES)
#         growth_factors = (Dz / D0) ** 2
        
#         # Apply growth factors and return
#         # pk_fid[None, :] ensures broadcasting works cleanly: (1, K) * (Z, 1) = (Z, K)
#         return pk_fid[None, :] * growth_factors[:, None]


#     def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
#         """
#         Performs the full NN prediction and PCA reconstructions.
        
#         Args:
#             params: Normalized 2D array of parameters (1, N_params).
            
#         Returns:
#             np.ndarray: Final predicted log fractional differences, shape (N_ZS, N_K_MODES).
#         """
#         # Predict T-components
#         t_comps_pred_scaled = self.model.predict(params, verbose=0)

#         # Invert NN scaling
#         t_comps_pred = self.t_scaler.inverse_transform(t_comps_pred_scaled)
        
#         # Inverse T-components to flat k-PCs (shape 1, N_ZS * N_PCS)
#         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_pred)
        
#         # Reshape to individual redshifts (shape N_ZS, N_PCS)
#         # We drop the single batch dimension (axis 0) since we only have one cosmology
#         pcs_pred_z_stack = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        
#         # Inverse K-PCA and Scaler Loop
#         reconstructed_fracs = [
#             self.SCALERS[float(f"{z:.3f}")].inverse_transform(
#                 self.PCAS[float(f"{z:.3f}")].inverse_transform(
#                     pcs_z.reshape(1, -1)
#                 )
#             )[0] # Extract the 1D result from the (1, 2000) array
#             for z, pcs_z in zip(self.Z_MODES, pcs_pred_z_stack)
#         ]

#         # Concatenate and return
#         # Resulting shape: (N_ZS, N_K_MODES)
#         return np.stack(reconstructed_fracs)


#     def get_pks(self, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Returns P(k, z) for all emulator redshifts for a given cosmology.
        
#         Args:
#             params: List or 1D array of 5 cosmological parameters.
            
#         Returns:
#             Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
#         """
#         # Normalize cosmological parameters
#         # Ensure input is a 2D array (1, N_params) for the scaler/model
#         x_norm = self.param_scaler.transform(np.array(params).reshape(1, -1))
        
#         # Generate predicted fractional differences (shape N_ZS, N_K_MODES)
#         frac = self._predict_fracs_all_z(x_norm)

#         # Compute MPS approximation (shape N_ZS, N_K_MODES)
#         pk_mps = self._compute_mps_approximation(np.array(params))

#         # Full emulated P(k, z)
#         pks = frac * pk_mps

#         # Return the k-modes, z-modes, and the P(k,z) array
#         return self.K_MODES, self.Z_MODES, pks

#     def weighted_t_mse(self, weights):
#         def loss(y_true, y_pred):
#             diff = y_pred - y_true
#             return tf.reduce_mean(diff**2 * weights)
#         return loss


# # --- Public Module-Level Interface ---

# # Instantiate the emulator once when the module is imported.
# _pk_emulator_instance = None
# if _DEPENDENCIES_LOADED:
#     try:
#         _pk_emulator_instance = PkEmulator()
#     except Exception as e:
#         logging.warning(f"PkEmulator instance failed during initialization: {e}")

# def get_pks(params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Module-level function to get P(k, z). This is the streamlined public interface.
    
#     It calls the get_pks method of the globally-instantiated PkEmulator object.
    
#     Args:
#         params: List or 1D array of 5 cosmological parameters, [10^9 A_s, ns, H0, Ob, Om].
            
#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
#     """
#     if _pk_emulator_instance is None:
#         raise RuntimeError("PkEmulator is not loaded. Check prior error messages regarding dependencies or files.")
        
#     return _pk_emulator_instance.get_pks(params)



# # Example Usage:
# # import fastMPS
# # k_modes, z_modes, pks_pred = fastMPS.get_pks([3.0e-9, 0.965, 67.5, 0.048, 0.31])
