"""
STELLAR CARTOGRAPHY
Unit tests for physics/stellar_properties.py

Run with:
    python -m pytest tests/test_stellar_properties.py
"""

import numpy as np

from physics.stellar_properties import (
    luminosity_from_stefan_boltzmann,
    radius_from_stefan_boltzmann,
    apparent_brightness,
    luminosity_from_brightness,
    SOLAR_RADIUS_M,
    SOLAR_LUMINOSITY_W,
    LIGHT_YEAR_M,
    watts_to_solar_luminosities,
)


def test_sun_reproduces_one_solar_luminosity():
    L = luminosity_from_stefan_boltzmann(SOLAR_RADIUS_M, 5778)
    assert np.isclose(watts_to_solar_luminosities(L), 1.0, atol=0.02)


def test_stefan_boltzmann_round_trip():
    true_radius = 3.5 * SOLAR_RADIUS_M
    true_temperature = 9000.0

    L = luminosity_from_stefan_boltzmann(true_radius, true_temperature)
    recovered_radius = radius_from_stefan_boltzmann(L, true_temperature)

    assert np.isclose(true_radius, recovered_radius, rtol=1e-9)


def test_stefan_boltzmann_rejects_invalid_inputs():
    try:
        luminosity_from_stefan_boltzmann(-1, 5000)
        assert False
    except ValueError:
        pass

    try:
        luminosity_from_stefan_boltzmann(1e8, 0)
        assert False
    except ValueError:
        pass


def test_inverse_square_round_trip():
    true_luminosity = 50 * SOLAR_LUMINOSITY_W
    distance_m = 20 * LIGHT_YEAR_M

    flux = apparent_brightness(true_luminosity, distance_m)
    recovered_luminosity = luminosity_from_brightness(flux, distance_m)

    assert np.isclose(true_luminosity, recovered_luminosity, rtol=1e-9)


def test_inverse_square_rejects_invalid_inputs():
    try:
        apparent_brightness(-1, 1e17)
        assert False
    except ValueError:
        pass

    try:
        apparent_brightness(1e26, 0)
        assert False
    except ValueError:
        pass
