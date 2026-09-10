"""
STELLAR CARTOGRAPHY
Blackbody Radiation / Planck's Law / Wien's Displacement Law

Physics calculations for stellar blackbody radiation and
temperature estimation from an observed spectrum.
"""

import numpy as np


# -----------------------------------------------------------------------
# PHYSICAL CONSTANTS
# -----------------------------------------------------------------------

H_PLANCK = 6.62607015e-34    # J*s
C_LIGHT = 2.99792458e8       # m/s
K_BOLTZMANN = 1.380649e-23   # J/K

# Wien's displacement law constant (nm * K).
WIEN_CONSTANT_NM_K = 2.8977719e6


# -----------------------------------------------------------------------
# PLANCK'S LAW
# -----------------------------------------------------------------------

def planck_law(
    wavelength_nm: np.ndarray,
    temperature_k: float,
) -> np.ndarray:
    """
    Calculate blackbody spectral radiance using Planck's Law.

    temperature_k must be a scalar. To render a family of curves
    across many temperatures, call this once per temperature
    rather than passing an array of temperatures.
    """

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)

    if temperature_k <= 0:
        raise ValueError("Temperature must be greater than zero.")

    wavelength_m = wavelength_nm * 1e-9

    exponent = (H_PLANCK * C_LIGHT) / (
        wavelength_m * K_BOLTZMANN * temperature_k
    )
    exponent = np.clip(exponent, -700, 700)

    intensity = (2 * H_PLANCK * C_LIGHT**2) / (
        wavelength_m**5 * np.expm1(exponent)
    )

    return intensity


def normalized_blackbody(
    wavelength_nm: np.ndarray,
    temperature_k: float,
) -> np.ndarray:
    """
    Generate a blackbody curve normalized to a maximum of 1.
    """

    intensity = planck_law(wavelength_nm, temperature_k)
    maximum = np.max(intensity)

    if maximum <= 0 or not np.isfinite(maximum):
        raise ValueError("Invalid blackbody intensity.")

    return intensity / maximum


# -----------------------------------------------------------------------
# WIEN'S DISPLACEMENT LAW
# -----------------------------------------------------------------------

def wien_peak_wavelength(temperature_k: float) -> float:
    """
    Forward Wien's Law: given a temperature, return the wavelength
    (nm) at which blackbody emission peaks.

        lambda_max = b / T
    """

    if temperature_k <= 0:
        raise ValueError("Temperature must be greater than zero.")

    return WIEN_CONSTANT_NM_K / temperature_k


def is_peak_near_window_edge(
    wavelength_nm: np.ndarray,
    flux: np.ndarray,
    edge_fraction: float = 0.03,
) -> bool:
    """
    Check whether the brightest observed point sits at (or very
    near) the blue edge of the observed wavelength window, which
    is the direct symptom of the star's true blackbody peak lying
    outside the window (in the ultraviolet). This is the actual
    mechanism behind the Wien's Law lower-bound limitation, so it
    is checked directly here rather than inferred indirectly from
    which star type was identified.

    Parameters
    ----------
    edge_fraction : float
        Fraction of the window width (from the blue edge) counted
        as "near the edge".
    """

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    flux = np.asarray(flux, dtype=float)

    peak_index = int(np.argmax(flux))
    edge_index_cutoff = int(edge_fraction * len(wavelength_nm))

    return peak_index <= edge_index_cutoff


def estimate_temperature_from_peak(
    wavelength_nm: np.ndarray,
    flux: np.ndarray,
) -> float:
    """
    Inverse Wien's Law: estimate a star's temperature from the
    wavelength at which its observed spectrum is brightest.

        T = b / lambda_max

    Notes
    -----
    If the star's true blackbody peak lies outside the observed
    wavelength window (this happens for very hot stars, whose peak
    is in the ultraviolet, below 380 nm), the brightest observed
    point will sit at the window's blue edge rather than the true
    peak. In that case this function still returns a value, but it
    is a lower bound on the star's true temperature rather than an
    exact estimate. This is a genuine, well known limitation of
    single-band optical temperature estimation in real astronomy,
    not an error in this implementation.
    """

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if len(wavelength_nm) != len(flux):
        raise ValueError(
            "Wavelength and flux arrays must have the same length."
        )

    if not np.isfinite(flux).all():
        raise ValueError("Flux contains NaN or infinite values.")

    peak_index = int(np.argmax(flux))
    peak_wavelength = wavelength_nm[peak_index]

    return wien_peak_wavelength_inverse(peak_wavelength)


def wien_peak_wavelength_inverse(peak_wavelength_nm: float) -> float:
    """
    T = b / lambda_max
    """

    if peak_wavelength_nm <= 0:
        raise ValueError("Peak wavelength must be greater than zero.")

    return WIEN_CONSTANT_NM_K / peak_wavelength_nm
