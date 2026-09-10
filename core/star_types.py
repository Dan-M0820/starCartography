"""
STELLAR CARTOGRAPHY
Star Type Definitions

Maps temperature ranges to a simplified set of four star "types",
each associated with real, named absorption lines that dominate a
star's spectrum at that temperature. Used both to generate a
star's synthetic spectrum (which lines to embed) and, separately,
to identify a star's type from its spectrum's detected lines.

This is a simplified four-bin model for teaching purposes, not a
substitute for the full stellar spectral classification system
(O, B, A, F, G, K, M) used in professional astronomy.
"""

# Each entry: temperature range (K), and the real spectral lines
# that dominate a star's absorption spectrum in that range.
# (wavelength_nm, element/molecule label)
STAR_TYPES = {
    "Helium star": {
        "temperature_range": (25000, 45000),
        "lines": [
            (447.1, "Helium I"),
            (468.6, "Helium II"),
            (492.2, "Helium I"),
        ],
        # Simplified radius prior for synthetic generation, in
        # solar radii. Not a precise real mass-radius relation.
        "radius_range_rsun": (5.0, 15.0),
    },
    "Hydrogen star": {
        "temperature_range": (10000, 25000),
        "lines": [
            (410.2, "Hydrogen (H-delta)"),
            (434.0, "Hydrogen (H-gamma)"),
            (486.1, "Hydrogen (H-beta)"),
            (656.3, "Hydrogen (H-alpha)"),
        ],
        "radius_range_rsun": (2.0, 8.0),
    },
    "Calcium star": {
        "temperature_range": (5000, 10000),
        "lines": [
            (393.4, "Ionized Calcium (Ca II K)"),
            (396.8, "Ionized Calcium (Ca II H)"),
            (517.3, "Magnesium (Mg b)"),
            (589.3, "Sodium (Na D)"),
        ],
        "radius_range_rsun": (0.8, 1.5),
    },
    "Molecular star": {
        "temperature_range": (2500, 5000),
        "lines": [
            (495.7, "Titanium Oxide (TiO)"),
            (545.0, "Titanium Oxide (TiO)"),
            (620.0, "Titanium Oxide (TiO)"),
            (705.0, "Titanium Oxide (TiO)"),
        ],
        "radius_range_rsun": (0.2, 0.6),
    },
}

# Ordered from coolest to hottest -- useful for display and for
# picking the correct bin from a temperature value.
STAR_TYPE_ORDER = [
    "Molecular star",
    "Calcium star",
    "Hydrogen star",
    "Helium star",
]

MIN_TEMPERATURE_K = STAR_TYPES["Molecular star"]["temperature_range"][0]
MAX_TEMPERATURE_K = STAR_TYPES["Helium star"]["temperature_range"][1]


def star_type_from_temperature(temperature_k: float) -> str:
    """
    Return the star type name whose temperature range contains
    the given temperature.
    """

    for star_type in STAR_TYPE_ORDER:
        low, high = STAR_TYPES[star_type]["temperature_range"]
        if low <= temperature_k < high:
            return star_type

    # Clamp to the nearest bin if slightly outside the defined range.
    if temperature_k < MIN_TEMPERATURE_K:
        return STAR_TYPE_ORDER[0]
    return STAR_TYPE_ORDER[-1]


def get_reference_lines(star_type: str):
    """
    Return the list of (wavelength_nm, label) reference lines for
    a given star type.
    """

    if star_type not in STAR_TYPES:
        raise ValueError(f"Unknown star type: {star_type}")

    return STAR_TYPES[star_type]["lines"]


def get_radius_range_rsun(star_type: str):
    """
    Return the (min, max) simplified radius prior in solar radii
    used for synthetic generation.
    """

    if star_type not in STAR_TYPES:
        raise ValueError(f"Unknown star type: {star_type}")

    return STAR_TYPES[star_type]["radius_range_rsun"]


def all_reference_lines():
    """
    Return the union of all reference lines across every star
    type, each tagged with the type it belongs to. Used by the
    line-identification module to match detected lines against
    every possibility without knowing the star's type in advance.
    """

    combined = []
    for star_type, info in STAR_TYPES.items():
        for wavelength, label in info["lines"]:
            combined.append((wavelength, label, star_type))
    return combined
