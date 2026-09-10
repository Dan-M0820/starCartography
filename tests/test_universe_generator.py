"""
STELLAR CARTOGRAPHY
Unit tests for core/universe_generator.py

Run with:
    python -m pytest tests/test_universe_generator.py
"""

import numpy as np

from core.universe_generator import generate_universe, MINIMUM_DISTANCE_LY


def test_reproducible_with_same_seed():
    u1 = generate_universe(n_stars=20, max_distance_ly=60, seed=42)
    u2 = generate_universe(n_stars=20, max_distance_ly=60, seed=42)

    for s1, s2 in zip(u1, u2):
        assert s1["position_ly"] == s2["position_ly"]
        assert s1["temperature_k"] == s2["temperature_k"]


def test_different_with_different_seed():
    u1 = generate_universe(n_stars=20, max_distance_ly=60, seed=42)
    u2 = generate_universe(n_stars=20, max_distance_ly=60, seed=99)

    assert u1[0]["position_ly"] != u2[0]["position_ly"]


def test_distances_within_bounds():
    max_distance = 60
    universe = generate_universe(n_stars=30, max_distance_ly=max_distance, seed=1)

    for star in universe:
        assert MINIMUM_DISTANCE_LY <= star["distance_ly"] <= max_distance


def test_all_physical_values_finite_and_positive():
    universe = generate_universe(n_stars=30, max_distance_ly=60, seed=7)

    for star in universe:
        assert star["temperature_k"] > 0
        assert star["radius_m"] > 0
        assert star["true_luminosity_w"] > 0
        assert star["apparent_flux_w_m2"] > 0
        assert np.isfinite(star["distance_m"])


def test_star_count_matches_request():
    universe = generate_universe(n_stars=15, max_distance_ly=60, seed=1)
    assert len(universe) == 15


def test_color_hex_format():
    universe = generate_universe(n_stars=5, max_distance_ly=60, seed=1)
    for star in universe:
        assert star["color_hex"].startswith("#")
        assert len(star["color_hex"]) == 7
