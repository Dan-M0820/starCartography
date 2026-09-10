"""
STELLAR CARTOGRAPHY
Star Analysis Pipeline

Ties together line identification and physics to independently
re-derive a star's properties from what would actually be
observable from Earth: its spectrum, its apparent brightness, and
its distance (read directly from the 3D map, analogous to a real
distance measurement such as parallax).

Pipeline:

    Spectrum
        |
        +--> detect + identify absorption lines --> star type (from lines)
        |
        +--> Wien's Law (brightest wavelength) --> estimated temperature

    Apparent brightness + distance (known)
        |
        +--> inverse-square law --> estimated luminosity

    Estimated luminosity + estimated temperature
        |
        +--> Stefan-Boltzmann Law --> estimated radius

Each estimated value is reported alongside the star's true
(generated) value so the pipeline's accuracy is visible rather
than assumed.
"""

import numpy as np

from core.preprocessing import smooth_spectrum
from core.spectrum_generator import WAVELENGTH_RANGE
from analysis.line_identification import (
    detect_absorption_lines,
    identify_lines,
    identify_star_type_from_lines,
)
from physics.blackbody import estimate_temperature_from_peak, is_peak_near_window_edge
from physics.stellar_properties import (
    luminosity_from_brightness,
    radius_from_stefan_boltzmann,
    watts_to_solar_luminosities,
    meters_to_solar_radii,
)


def analyze_star(
    star: dict,
    raw_flux: np.ndarray,
    wavelength: np.ndarray = WAVELENGTH_RANGE,
) -> dict:
    """
    Run the full independent-estimation pipeline on one star's
    spectrum and known observational quantities.

    Parameters
    ----------
    star : dict
        A star record from core.universe_generator.generate_universe,
        providing the true values this function's estimates will be
        compared against, plus the known apparent flux and distance
        used as inputs to the physics.

    raw_flux : np.ndarray
        The star's generated (noisy) spectrum.

    Returns
    -------
    dict
        Detected/identified lines, identified star type, and
        estimated temperature / luminosity / radius, each paired
        with the corresponding true value and absolute error.
    """

    cleaned_flux = smooth_spectrum(raw_flux)

    # --- Line identification -------------------------------------
    detected_lines = detect_absorption_lines(wavelength, cleaned_flux)
    identified_lines = identify_lines(detected_lines)
    identified_type, type_votes, total_identified = identify_star_type_from_lines(
        identified_lines
    )

    # --- Wien's Law temperature estimate ---------------------------
    estimated_temperature_k = estimate_temperature_from_peak(
        wavelength, cleaned_flux
    )

    # --- Inverse-square law: luminosity from known brightness + distance
    estimated_luminosity_w = luminosity_from_brightness(
        star["apparent_flux_w_m2"], star["distance_m"]
    )

    # --- Stefan-Boltzmann: radius from estimated luminosity + temperature
    estimated_radius_m = radius_from_stefan_boltzmann(
        estimated_luminosity_w, estimated_temperature_k
    )

    # --- Reliability flag for the Wien's Law estimate --------------
    # Checked directly: is the brightest observed point sitting at
    # the window's blue edge? That is the actual symptom of the
    # star's true peak lying in the ultraviolet, below 380nm --
    # checking this directly is more accurate than inferring it
    # from which star type the lines identified, since stars near
    # the top of the Calcium-star range (roughly 7,600-10,000K)
    # can show the same edge-saturation the Hydrogen/Helium stars
    # show.
    temperature_estimate_reliable = not is_peak_near_window_edge(
        wavelength, cleaned_flux
    )

    return {
        "cleaned_flux": cleaned_flux,
        "detected_lines": detected_lines,
        "identified_lines": identified_lines,
        "identified_star_type": identified_type,
        "identified_type_votes": type_votes,
        "identified_type_total": total_identified,
        "type_matches_true": identified_type == star["star_type"],

        "estimated_temperature_k": estimated_temperature_k,
        "true_temperature_k": star["temperature_k"],
        "temperature_error_k": abs(estimated_temperature_k - star["temperature_k"]),
        "temperature_estimate_reliable": temperature_estimate_reliable,

        "estimated_luminosity_w": estimated_luminosity_w,
        "estimated_luminosity_lsun": watts_to_solar_luminosities(estimated_luminosity_w),
        "true_luminosity_lsun": star["true_luminosity_lsun"],

        "estimated_radius_m": estimated_radius_m,
        "estimated_radius_rsun": meters_to_solar_radii(estimated_radius_m),
        "true_radius_rsun": star["radius_rsun"],
    }
