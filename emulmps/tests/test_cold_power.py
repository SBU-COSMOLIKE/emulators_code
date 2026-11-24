"""Check the Eisenstein-Hu cold-field approximation and Cobaya state changes.

Run with Cocoa's Python and ROOTDIR set. The network test uses the ordinary
installed linear and Mead2020 files; it does not run CAMB or a survey.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from cobaya.log import LoggedError
from emulmps.emulmps import emulmps


def test_ratio_limits_and_species():
    theory = emulmps.__new__(emulmps)
    theory.log = logging.getLogger("cold_power_test")
    theory.cold_neutrino_species = 1
    parameters = [2.1, 0.96, 69.0, 0.048, 0.3, -1.0, 0.0, 7.8, 0.06]
    k = np.array([1e-10, 0.01, 1e5])
    z = np.array([0.0, 1.0, 2.0])
    fraction = 0.06 * (3.046 / 3)**0.75 / 94.0708 / (0.3 * 0.69**2)
    first = theory._cold_power_ratio(k, z, parameters)
    # The finite-time growth fit has a small residual at k=0 (3.9e-5
    # here); it does not impose the exact infinite-growth limit.
    np.testing.assert_allclose(first[:, 0], 1.0, atol=5e-5, rtol=0)
    np.testing.assert_allclose(first[:, -1], (1 - fraction)**-2, rtol=1e-10)
    assert np.all((first >= 1.0) & (first <= (1 - fraction)**-2 * (1 + 1e-12)))
    theory.cold_neutrino_species = 3
    third = theory._cold_power_ratio(k, z, parameters)
    assert np.all(third[:, 1] > first[:, 1])
    parameters[-1] = 0.0
    np.testing.assert_array_equal(theory._cold_power_ratio(k, z, parameters), 1.0)
    parameters[5] = -0.9
    with pytest.raises(LoggedError, match="flat LCDM"):
        theory._cold_power_ratio(k, z, parameters)


def test_cobaya_cold_grid_and_revisited_cosmology():
    """Requests, interpolation and A-to-B-to-A must preserve field identity."""
    from cobaya.model import get_model

    root = Path(os.environ["ROOTDIR"])
    data = root / "external_modules/data/emultrf/emulmps"
    info = {
        "params": {
            "As_1e9": 2.1, "ns": 0.96, "H0": 69.0, "omegab": 0.048,
            "omegam": {"prior": {"min": 0.2, "max": 0.5}},
            "w": -1.0, "wa": 0.0,
        },
        "likelihood": {"one": None},
        "theory": {"emulmps": {
            "external": emulmps,
            "stop_at_error": True,
            "extra_args": {
                "model_file": str(data / "npce_emul/emulator_npce.keras"),
                "metadata_file": str(data / "npce_emul/metadata.joblib"),
                "nl_model_file": str(data / "w0wa_mead2020/emulator_mead2020.keras"),
                "nl_metadata_file": str(data / "w0wa_mead2020/metadata.joblib"),
                "param_order": ["As_1e9", "ns", "H0", "omegab", "omegam", "w", "wa"],
                "fixed_params": {"mnu": 0.06},
                "cold_power": "eisenstein_hu",
                "cold_neutrino_species": 1,
            },
        }},
    }
    model = get_model(info)
    try:
        model.add_requirements({"Pk_interpolator": {
            "z": [0.0, 1.0], "k_max": 5.0, "nonlinear": False,
            "vars_pairs": [["delta_tot", "delta_tot"]],
        }})
        model.logposterior({"omegam": 0.3}, cached=False)
        theory = model.theory["emulmps"]
        total_before = theory.get_Pk_grid()[2].copy()
        with pytest.raises(LoggedError, match="not computed"):
            theory.get_Pk_grid(var_pair=("delta_nonu", "delta_nonu"))

        # Adding a field invalidates the earlier total-only cached state.
        model.add_requirements({"Pk_interpolator": {
            "z": [0.0, 1.0], "k_max": 5.0, "nonlinear": False,
            "vars_pairs": [["delta_nonu", "delta_nonu"]],
        }})
        predictions = []
        for omega in [0.3, 0.4, 0.3]:
            assert np.isfinite(model.logposterior({"omegam": omega}, cached=True).loglike)
            k, z, cold = theory.get_Pk_grid(var_pair=("delta_nonu", "delta_nonu"))
            total = theory.get_Pk_grid()[2]
            parameters = [2.1, 0.96, 69.0, 0.048, omega, -1.0, 0.0, 7.8, 0.06]
            np.testing.assert_allclose(cold / total, theory._cold_power_ratio(k, z, parameters), rtol=1e-12)
            interp = theory.get_Pk_interpolator(var_pair=("delta_nonu", "delta_nonu"), extrap_kmax=250.0)
            predictions.append(interp.P([0.0, 1.0], [0.01, 0.1, 5.0, 250.0]).copy())
        np.testing.assert_array_equal(predictions[0], predictions[2])
        assert not np.array_equal(predictions[0], predictions[1])
        np.testing.assert_array_equal(total_before, theory.get_Pk_grid()[2])
        with pytest.raises(LoggedError, match="linear cold power only"):
            theory.get_Pk_interpolator(var_pair=("delta_nonu", "delta_nonu"), nonlinear=True)
        with pytest.raises(LoggedError, match="Unsupported matter fields"):
            theory.get_Pk_grid(var_pair=("delta_tot", "delta_nonu"))
    finally:
        model.close()


def test_ratio_against_saved_camb_fields():
    """Compare to independent CAMB cb and total fields, not our own fit."""
    rows = np.loadtxt(Path(__file__).with_name("camb_cold_total_mnu006.txt"))
    theory = emulmps.__new__(emulmps)
    theory.log = logging.getLogger("cold_power_test")
    for omega, h, species, k, z, total_over_cold in rows:
        theory.cold_neutrino_species = int(species)
        parameters = [2.19, 0.96859, 100*h, 0.048, omega, -1.0, 0.0, 7.8, 0.06]
        ratio = theory._cold_power_ratio(np.array([k]), np.array([z]), parameters)[0, 0]
        assert abs(total_over_cold * ratio - 1) < 5e-4
