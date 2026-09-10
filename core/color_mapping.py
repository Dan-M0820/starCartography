"""
STELLAR CARTOGRAPHY
Color Mapping

Approximate physical color conversions used purely for
visualization:

1. wavelength_to_rgb -- converts a visible-light wavelength (nm)
   to an approximate perceived RGB color. Used to render the
   rainbow-colored absorption spectrum bar.

2. blackbody_temperature_to_rgb -- converts a star's temperature
   to an approximate visible color. Used to color stars in the 3D
   universe view.

Both are standard, widely published approximation algorithms
(piecewise physical/perceptual models), not exact colorimetry.
"""

import numpy as np


# -----------------------------------------------------------------------
# WAVELENGTH -> RGB
# -----------------------------------------------------------------------

def wavelength_to_rgb(wavelength_nm: float) -> tuple:
    """
    Approximate the perceived RGB color of a single visible-light
    wavelength (piecewise linear model over ~380-750nm, tapering
    intensity near the violet and red edges of human vision).

    Returns
    -------
    (r, g, b) : tuple of int, each 0-255.
    """

    wl = wavelength_nm

    if wl < 380 or wl > 750:
        return (0, 0, 0)

    if wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # Taper intensity near the edges of visibility.
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (750 - wl) / (750 - 700)
    else:
        factor = 1.0

    gamma = 0.8
    r = (r * factor) ** gamma if r > 0 else 0.0
    g = (g * factor) ** gamma if g > 0 else 0.0
    b = (b * factor) ** gamma if b > 0 else 0.0

    return (
        int(round(255 * r)),
        int(round(255 * g)),
        int(round(255 * b)),
    )


# -----------------------------------------------------------------------
# TEMPERATURE -> RGB (approximate blackbody color)
# -----------------------------------------------------------------------

def blackbody_temperature_to_rgb(temperature_k: float) -> tuple:
    """
    Approximate a blackbody's perceived color from its
    temperature, using the widely published piecewise polynomial
    approximation (valid roughly 1000K-40000K).

    Returns
    -------
    (r, g, b) : tuple of int, each 0-255.
    """

    temp = np.clip(temperature_k, 1000, 40000) / 100.0

    # Red
    if temp <= 66:
        red = 255.0
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)

    # Green
    if temp <= 66:
        green = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)

    # Blue
    if temp >= 66:
        blue = 255.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    red = np.clip(red, 0, 255)
    green = np.clip(green, 0, 255)
    blue = np.clip(blue, 0, 255)

    return (int(round(red)), int(round(green)), int(round(blue)))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert an (r, g, b) tuple to a '#rrggbb' hex string."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"
