"""
STELLAR CARTOGRAPHY
Unit tests for analysis/line_identification.py

Run with:
    python -m pytest tests/test_line_identification.py
"""

import numpy as np

from core.spectrum_generator import generate_spectrum, WAVELENGTH_RANGE
from core.preprocessing import smooth_spectrum
from core.star_types import star_type_from_temperature, STAR_TYPE_ORDER
from analysis.line_identification import (
    detect_absorption_lines,
    identify_lines,
    identify_star_type_from_lines,
)

# Kept low deliberately -- line identification is only reliable at
# low noise (see project notes); this mirrors the noise level used
# by the dashboard.
NOISE_LEVEL = 0.015


def test_detect_absorption_lines_finds_real_lines_not_noise():
    rng = np.random.default_rng(5)
    flux = generate_spectrum(15000, noise_level=NOISE_LEVEL, rng=rng)
    cleaned = smooth_spectrum(flux)

    detected = detect_absorption_lines(WAVELENGTH_RANGE, cleaned)

    # A Hydrogen star has exactly 4 designed lines; detection
    # should find a small number close to that, not dozens of
    # noise-driven spurious peaks.
    assert 2 <= len(detected) <= 8


def test_identify_lines_matches_known_hydrogen_lines():
    rng = np.random.default_rng(5)
    flux = generate_spectrum(15000, noise_level=NOISE_LEVEL, rng=rng)
    cleaned = smooth_spectrum(flux)

    detected = detect_absorption_lines(WAVELENGTH_RANGE, cleaned)
    identified = identify_lines(detected)

    labels = {line["label"] for line in identified}
    assert any("Hydrogen" in label for label in labels)


def test_identify_star_type_from_lines_across_all_types():
    rng = np.random.default_rng(123)
    correct = 0
    total = 0

    for temperature in np.random.default_rng(1).uniform(2600, 44000, 40):
        true_type = star_type_from_temperature(temperature)
        flux = generate_spectrum(temperature, noise_level=NOISE_LEVEL, rng=rng)
        cleaned = smooth_spectrum(flux)

        detected = detect_absorption_lines(WAVELENGTH_RANGE, cleaned)
        identified = identify_lines(detected)
        predicted_type, _votes, _total = identify_star_type_from_lines(identified)

        total += 1
        correct += predicted_type == true_type

    # At low noise this should be highly reliable.
    assert correct / total >= 0.9


def test_identify_star_type_returns_none_for_no_lines():
    predicted_type, votes, total = identify_star_type_from_lines([])
    assert predicted_type is None
    assert votes == 0
    assert total == 0
