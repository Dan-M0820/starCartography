"""
STELLAR CARTOGRAPHY
Spectrum Preprocessing

Smooths a raw synthetic spectrum before it is used for absorption
line detection, Wien's Law temperature estimation, or display.
Without this step, per-point observational noise creates many
spurious "absorption lines" that swamp the real ones.
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_spectrum(
    flux: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to reduce point-to-point noise
    while preserving the shape of real absorption features.
    """

    flux = np.asarray(flux, dtype=float)

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")

    if len(flux) < window_length:
        raise ValueError(
            "Spectrum is shorter than the smoothing window length."
        )

    return savgol_filter(flux, window_length=window_length, polyorder=polyorder)
