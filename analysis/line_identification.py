"""
STELLAR CARTOGRAPHY
Absorption Line Identification

Detects absorption lines in an observed spectrum and matches
them against the real reference lines for each star type,
producing an independent type identification based purely on
which elements/molecules are detected -- separate from, and a
cross-check against, the Wien's-Law temperature-based
classification.
"""

import numpy as np
from scipy.signal import find_peaks

from core.star_types import STAR_TYPES, STAR_TYPE_ORDER


def detect_absorption_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    prominence: float = 0.05,
    distance: int = 10,
) -> list:
    """
    Detect absorption dips in a spectrum.

    Returns
    -------
    list of dict
        Each dict has "wavelength" and "depth" for one detected
        absorption line.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    absorption_signal = 1.0 - flux

    peaks, properties = find_peaks(
        absorption_signal,
        prominence=prominence,
        distance=distance,
    )

    detected = []
    for peak_index, depth in zip(peaks, properties["prominences"]):
        detected.append({
            "wavelength": float(wavelength[peak_index]),
            "depth": float(depth),
        })

    return detected


def identify_lines(
    detected_lines: list,
    tolerance_nm: float = 6.0,
) -> list:
    """
    Match each detected line against the union of all star types'
    reference lines, within a tolerance window. This does NOT
    know or assume the star's actual type -- it checks every
    possibility and reports the closest match.

    Returns
    -------
    list of dict
        Each input line augmented with "label" (matched element,
        or "Unidentified") and "matched_star_type" (or None).
    """

    identified = []

    for line in detected_lines:
        best_match = None
        best_distance = tolerance_nm + 1

        for star_type, info in STAR_TYPES.items():
            for ref_wavelength, ref_label in info["lines"]:
                distance = abs(line["wavelength"] - ref_wavelength)
                if distance <= tolerance_nm and distance < best_distance:
                    best_distance = distance
                    best_match = (ref_label, star_type)

        identified.append({
            **line,
            "label": best_match[0] if best_match else "Unidentified",
            "matched_star_type": best_match[1] if best_match else None,
        })

    return identified


def identify_star_type_from_lines(identified_lines: list) -> tuple:
    """
    Determine the most likely star type from a set of identified
    lines, by counting how many detected lines matched each type's
    reference set.

    Returns
    -------
    (star_type, match_count, total_identified) : tuple
        star_type is None if no lines matched anything.
    """

    votes = {star_type: 0 for star_type in STAR_TYPE_ORDER}

    for line in identified_lines:
        if line["matched_star_type"] is not None:
            votes[line["matched_star_type"]] += 1

    total_identified = sum(votes.values())

    if total_identified == 0:
        return None, 0, 0

    best_type = max(votes, key=votes.get)
    return best_type, votes[best_type], total_identified
