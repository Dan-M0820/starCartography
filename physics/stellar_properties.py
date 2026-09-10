"""
STELLAR CARTOGRAPHY
Stellar Physical Properties

Stefan-Boltzmann Law and the inverse-square law of light,
used both to generate a synthetic star's true properties and to
independently re-derive them from what an observer at Earth would
measure (apparent brightness + distance + spectrum).
"""

import numpy as np


# -----------------------------------------------------------------------
# PHYSICAL CONSTANTS
# -----------------------------------------------------------------------

STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8   # W / (m^2 K^4)

SOLAR_RADIUS_M = 6.957e8         # meters
SOLAR_LUMINOSITY_W = 3.828e26    # watts
LIGHT_YEAR_M = 9.4607304725808e15  # meters


# -----------------------------------------------------------------------
# STEFAN-BOLTZMANN LAW
# -----------------------------------------------------------------------

def luminosity_from_stefan_boltzmann(
    radius_m: float,
    temperature_k: float,
) -> float:
    """
    Forward Stefan-Boltzmann Law: total power radiated by a
    spherical blackbody of given radius and temperature.

        L = 4 * pi * R^2 * sigma * T^4
    """

    if radius_m <= 0:
        raise ValueError("Radius must be greater than zero.")

    if temperature_k <= 0:
        raise ValueError("Temperature must be greater than zero.")

    return (
        4 * np.pi * radius_m**2
        * STEFAN_BOLTZMANN_CONSTANT
        * temperature_k**4
    )


def radius_from_stefan_boltzmann(
    luminosity_w: float,
    temperature_k: float,
) -> float:
    """
    Inverse Stefan-Boltzmann Law: solve for radius given
    luminosity and temperature.

        R = sqrt( L / (4 * pi * sigma * T^4) )
    """

    if luminosity_w <= 0:
        raise ValueError("Luminosity must be greater than zero.")

    if temperature_k <= 0:
        raise ValueError("Temperature must be greater than zero.")

    return np.sqrt(
        luminosity_w
        / (4 * np.pi * STEFAN_BOLTZMANN_CONSTANT * temperature_k**4)
    )


# -----------------------------------------------------------------------
# INVERSE-SQUARE LAW OF LIGHT
# -----------------------------------------------------------------------

def apparent_brightness(
    luminosity_w: float,
    distance_m: float,
) -> float:
    """
    Forward inverse-square law: flux (W/m^2) received at a given
    distance from a source of known luminosity.

        flux = L / (4 * pi * d^2)
    """

    if luminosity_w <= 0:
        raise ValueError("Luminosity must be greater than zero.")

    if distance_m <= 0:
        raise ValueError("Distance must be greater than zero.")

    return luminosity_w / (4 * np.pi * distance_m**2)


def luminosity_from_brightness(
    flux_w_m2: float,
    distance_m: float,
) -> float:
    """
    Inverse of the inverse-square law: solve for luminosity given
    an observed flux and a known distance.

        L = flux * 4 * pi * d^2

    This mirrors how real astronomy actually works: distance is
    obtained independently (e.g. via parallax; here, directly from
    the 3D map position relative to Earth), and combined with
    observed brightness to derive luminosity.
    """

    if flux_w_m2 <= 0:
        raise ValueError("Flux must be greater than zero.")

    if distance_m <= 0:
        raise ValueError("Distance must be greater than zero.")

    return flux_w_m2 * 4 * np.pi * distance_m**2


# -----------------------------------------------------------------------
# UNIT CONVENIENCE
# -----------------------------------------------------------------------

def meters_to_solar_radii(radius_m: float) -> float:
    return radius_m / SOLAR_RADIUS_M


def watts_to_solar_luminosities(luminosity_w: float) -> float:
    return luminosity_w / SOLAR_LUMINOSITY_W


def light_years_to_meters(distance_ly: float) -> float:
    return distance_ly * LIGHT_YEAR_M


def meters_to_light_years(distance_m: float) -> float:
    return distance_m / LIGHT_YEAR_M
