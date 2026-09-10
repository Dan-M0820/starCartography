"""
STELLAR CARTOGRAPHY
Unit tests for core/star_types.py

Run with:
    python -m pytest tests/test_star_types.py
"""

from core.star_types import (
    star_type_from_temperature,
    get_reference_lines,
    get_radius_range_rsun,
    all_reference_lines,
    STAR_TYPES,
    STAR_TYPE_ORDER,
)


def test_star_type_boundaries():
    assert star_type_from_temperature(3000) == "Molecular star"
    assert star_type_from_temperature(7000) == "Calcium star"
    assert star_type_from_temperature(15000) == "Hydrogen star"
    assert star_type_from_temperature(30000) == "Helium star"


def test_star_type_ranges_are_contiguous_and_non_overlapping():
    ranges = [STAR_TYPES[t]["temperature_range"] for t in STAR_TYPE_ORDER]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]


def test_every_type_has_reference_lines():
    for star_type in STAR_TYPE_ORDER:
        lines = get_reference_lines(star_type)
        assert len(lines) >= 3
        for wavelength, label in lines:
            assert 380 <= wavelength <= 750
            assert isinstance(label, str) and len(label) > 0


def test_every_type_has_a_radius_range():
    for star_type in STAR_TYPE_ORDER:
        lo, hi = get_radius_range_rsun(star_type)
        assert 0 < lo < hi


def test_all_reference_lines_covers_every_type():
    combined = all_reference_lines()
    found_types = {star_type for _, _, star_type in combined}
    assert found_types == set(STAR_TYPE_ORDER)


def test_unknown_star_type_raises():
    try:
        get_reference_lines("Not a real type")
        assert False
    except ValueError:
        pass
