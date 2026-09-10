"""
STELLAR CARTOGRAPHY
Synthetic Universe Generator

Generates a reproducible (seeded) population of synthetic stars
placed in 3D space around a fixed Earth at the origin. Each star
has a true temperature, radius, luminosity, and distance -- the
"ground truth" that the analysis pipeline later tries to
independently recover from the star's spectrum and apparent
brightness alone.
"""

import numpy as np

from core.star_types import (
    star_type_from_temperature,
    get_radius_range_rsun,
    MIN_TEMPERATURE_K,
    MAX_TEMPERATURE_K,
)
from core.color_mapping import blackbody_temperature_to_rgb, rgb_to_hex
from physics.stellar_properties import (
    luminosity_from_stefan_boltzmann,
    apparent_brightness,
    SOLAR_RADIUS_M,
    SOLAR_LUMINOSITY_W,
    LIGHT_YEAR_M,
)

# Stars are placed no closer than this to Earth, to avoid
# division-by-zero / unrealistically extreme brightness values.
MINIMUM_DISTANCE_LY = 4.0


def generate_universe(
    n_stars: int = 30,
    max_distance_ly: float = 60.0,
    seed: int = 42,
) -> list:
    """
    Generate a reproducible synthetic star population.

    Parameters
    ----------
    n_stars : int
        Number of stars to generate.

    max_distance_ly : float
        Maximum distance from Earth (light-years) a star may be
        placed at.

    seed : int
        Random seed. The same seed always produces the same
        population (positions, temperatures, radii).

    Returns
    -------
    list of dict
        Each star dict contains:
            id, position_ly (x, y, z), distance_ly, distance_m,
            temperature_k, star_type, radius_m, radius_rsun,
            true_luminosity_w, true_luminosity_lsun,
            apparent_flux_w_m2, color_rgb, color_hex
    """

    rng = np.random.default_rng(seed)

    stars = []

    for i in range(n_stars):
        # Log-uniform temperature so all four star types are
        # reasonably represented, rather than being dominated by
        # one wide bin.
        log_temp = rng.uniform(
            np.log10(MIN_TEMPERATURE_K),
            np.log10(MAX_TEMPERATURE_K),
        )
        temperature_k = float(10 ** log_temp)

        star_type = star_type_from_temperature(temperature_k)

        radius_lo, radius_hi = get_radius_range_rsun(star_type)
        radius_rsun = rng.uniform(radius_lo, radius_hi)
        radius_m = radius_rsun * SOLAR_RADIUS_M

        true_luminosity_w = luminosity_from_stefan_boltzmann(
            radius_m, temperature_k
        )

        # Random position on a sphere between MINIMUM_DISTANCE_LY
        # and max_distance_ly from Earth (at the origin).
        distance_ly = rng.uniform(MINIMUM_DISTANCE_LY, max_distance_ly)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        position_ly = direction * distance_ly

        distance_m = distance_ly * LIGHT_YEAR_M
        apparent_flux = apparent_brightness(true_luminosity_w, distance_m)

        color_rgb = blackbody_temperature_to_rgb(temperature_k)

        stars.append({
            "id": i,
            "position_ly": tuple(position_ly),
            "distance_ly": float(distance_ly),
            "distance_m": float(distance_m),
            "temperature_k": temperature_k,
            "star_type": star_type,
            "radius_m": float(radius_m),
            "radius_rsun": float(radius_rsun),
            "true_luminosity_w": float(true_luminosity_w),
            "true_luminosity_lsun": float(true_luminosity_w / SOLAR_LUMINOSITY_W),
            "apparent_flux_w_m2": float(apparent_flux),
            "color_rgb": color_rgb,
            "color_hex": rgb_to_hex(color_rgb),
        })

    return stars
