"""Check the physical smoothing radius and its Cobaya parameter plumbing.

Run with Cocoa's Python from this repository: python -m pytest
emulmps/tests/test_sigma8.py. The synthetic spectra avoid loading networks.
"""

import numpy as np
import pytest

from emulmps.emulmps import emulmps


@pytest.mark.parametrize("h", [0.55, 0.67, 0.85])
def test_sigma8_power_law(h):
    """For P(k)=A/k^2, the top-hat integral is 3A/(10*pi*R).

    All lengths here are physical Mpc. Varying h must therefore change
    R=8/h and the variance; using a fixed radius would fail this test.
    """
    k = np.geomspace(1e-7, 1e3, 50000)
    amplitude = 100.0
    power = (amplitude / k**2)[None, :]
    actual = emulmps._compute_sigma8(
        None, power, k, np.array([0.0]), h=h,
    )
    expected = np.sqrt(3 * amplitude / (10 * np.pi * (8 / h)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_calculate_passes_current_hubble_parameter():
    """A reused theory object must use each evaluation's H0 for sigma8."""
    class Spectrum:
        def get_pks(self, parameters, use_syren=False):
            k = np.geomspace(1e-5, 100, 2000)
            return k, np.array([0.0]), (100 / k**2)[None, :]

        def has_nl_model(self):
            return False

    theory = emulmps.__new__(emulmps)
    theory._emulator = Spectrum()
    theory._emul_param_vector = lambda parameters: [
        2.1, 0.96, parameters["H0"], 0.049, 0.319, -1, 0, 7.8, 0.06,
    ]
    theory._k_keep = lambda k: None
    theory._nl_available = lambda: False
    theory.use_syren = False
    theory._use_camb = False
    theory.output_params = ["sigma8"]
    theory.stop_at_error = True
    values = []
    for hubble in [55.0, 85.0, 55.0]:
        state = {}
        assert theory.calculate(state, H0=hubble)
        values.append(state["derived"]["sigma8"])
    np.testing.assert_allclose(values[1] / values[0], np.sqrt(85 / 55), rtol=2e-5)
    assert values[0] == values[2]
