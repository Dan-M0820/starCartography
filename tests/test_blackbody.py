"""
STELLAR CARTOGRAPHY
Unit tests for physics/blackbody.py

Run with:
    python -m pytest tests/test_blackbody.py
"""

import numpy as np

from physics.blackbody import (
    planck_law,
    normalized_blackbody,
    wien_peak_wavelength,
    wien_peak_wavelength_inverse,
    estimate_temperature_from_peak,
)


def test_planck_law_positive_and_finite():
    wavelength = np.linspace(380, 750, 1000)
    intensity = planck_law(wavelength, 5800)
    assert np.isfinite(intensity).all()
    assert (intensity > 0).all()


def test_planck_law_rejects_nonpositive_temperature():
    wavelength = np.linspace(380, 750, 1000)
    try:
        planck_law(wavelength, 0)
        assert False, "Expected ValueError for zero temperature"
    except ValueError:
        pass


def test_normalized_blackbody_peaks_at_one():
    wavelength = np.linspace(380, 750, 1000)
    curve = normalized_blackbody(wavelength, 5800)
    assert np.isclose(curve.max(), 1.0)


def test_wien_law_round_trip():
    for T in [3000, 5800, 10000, 20000, 40000]:
        peak = wien_peak_wavelength(T)
        recovered = wien_peak_wavelength_inverse(peak)
        assert np.isclose(T, recovered)


def test_wien_law_solar_peak_matches_known_value():
    # The Sun's true peak wavelength is well known to be ~500-502nm.
    peak = wien_peak_wavelength(5778)
    assert 495 <= peak <= 510


def test_estimate_temperature_from_peak_matches_forward_law():
    wavelength = np.linspace(380, 750, 1000)
    true_temperature = 5800.0
    clean_spectrum = normalized_blackbody(wavelength, true_temperature)

    estimated = estimate_temperature_from_peak(wavelength, clean_spectrum)

    assert np.isclose(estimated, true_temperature, atol=50)


def test_estimate_temperature_hot_star_saturates_at_window_edge():
    # A star hotter than ~7500K has its true peak below 380nm, so
    # the estimate should sit near the temperature implied by the
    # window's blue edge (380nm), not the true (much higher) value.
    wavelength = np.linspace(380, 750, 1000)
    hot_spectrum = normalized_blackbody(wavelength, 30000)

    estimated = estimate_temperature_from_peak(wavelength, hot_spectrum)

    expected_edge_temperature = wien_peak_wavelength_inverse(380.0)
    assert np.isclose(estimated, expected_edge_temperature, atol=5)
    assert estimated < 30000  # confirms it under-estimates, as expected
