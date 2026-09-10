"""
STELLAR CARTOGRAPHY
Synthetic Spectrum Generator

Builds a synthetic stellar spectrum from a temperature: a Planck
blackbody continuum with absorption lines from that temperature's
star type (see core/star_types.py) superimposed, plus
observational noise.
"""

import numpy as np

from physics.blackbody import planck_law
from core.star_types import star_type_from_temperature, get_reference_lines


WAVELENGTH_RANGE = np.linspace(380, 750, 1000)


def generate_spectrum(
    temperature_k: float,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
    wavelength: np.ndarray = WAVELENGTH_RANGE,
) -> np.ndarray:
    """
    Generate a synthetic, normalized spectrum for a star of the
    given temperature.

    Pipeline:
        Planck continuum (from temperature)
              -> absorption lines (from the star's type)
              -> observational noise
              -> normalized flux

    Parameters
    ----------
    temperature_k : float
        The star's true temperature.

    noise_level : float
        Standard deviation of additive Gaussian observational noise.

    rng : np.random.Generator, optional
        Random generator for reproducibility. A new default
        generator is created if not provided.

    Returns
    -------
    np.ndarray
        Normalized flux values (roughly 0-1) at each wavelength.
    """

    if rng is None:
        rng = np.random.default_rng()

    continuum = planck_law(wavelength, temperature_k)
    continuum = continuum / continuum.max()

    star_type = star_type_from_temperature(temperature_k)
    reference_lines = get_reference_lines(star_type)

    flux = continuum.copy()

    for center, _label in reference_lines:
        # Small physical/instrumental variation between stars.
        varied_center = center + rng.normal(0, 0.6)
        depth = rng.uniform(0.25, 0.55)
        width = rng.uniform(2.0, 4.0)

        absorption = depth * np.exp(
            -0.5 * ((wavelength - varied_center) / width) ** 2
        )
        flux = flux * (1.0 - absorption)

    # Observational noise.
    flux = flux + rng.normal(0, noise_level, size=flux.shape)

    # Keep flux non-negative (detectors cannot report negative light).
    flux = np.clip(flux, 0.0, None)

    return flux
