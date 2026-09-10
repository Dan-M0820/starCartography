"""
STELLAR CARTOGRAPHY
Spectrum Visualization

Renders a star's spectrum two ways, stacked together:

1. A classic rainbow absorption-line spectrum bar -- a continuous
   band of color across wavelength, with black gaps where
   absorption lines remove light. This is the "read it like a
   photograph" view.

2. A standard flux-vs-wavelength curve below it, so the same
   dips can also be read quantitatively.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.color_mapping import wavelength_to_rgb


def build_absorption_spectrum_image(
    wavelength: np.ndarray,
    flux: np.ndarray,
) -> np.ndarray:
    """
    Build an RGB image array representing the spectrum as a
    continuous color band with black absorption gaps.

    Each pixel's base color comes from its wavelength; its
    brightness is then scaled down by how much flux is missing at
    that point (1 - flux), so deep absorption lines appear as
    black gaps in the color band.

    Returns
    -------
    np.ndarray, shape (1, len(wavelength), 3), dtype uint8
    """

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    flux_clipped = np.clip(flux, 0.0, 1.0)

    row = np.zeros((len(wavelength), 3), dtype=np.uint8)
    for i, wl in enumerate(wavelength):
        r, g, b = wavelength_to_rgb(wl)
        brightness = flux_clipped[i]
        row[i] = (
            int(r * brightness),
            int(g * brightness),
            int(b * brightness),
        )

    # A few rows tall purely so it renders as a visible band
    # rather than a single pixel-thin line.
    image = np.tile(row, (40, 1, 1))

    return image


def plot_absorption_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    identified_lines: list | None = None,
    title: str = "Absorption Line Spectrum",
) -> go.Figure:
    """
    Combined figure: the colored absorption-line bar on top, the
    flux curve below, sharing a wavelength axis.

    Parameters
    ----------
    identified_lines : list of dict, optional
        Output of analysis.line_identification.identify_lines --
        used to label detected lines with their matched element
        on the flux curve.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    image = build_absorption_spectrum_image(wavelength, flux)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.06,
        subplot_titles=("", "Flux"),
    )

    fig.add_trace(
        go.Image(
            z=image,
            x0=float(wavelength[0]),
            dx=float(wavelength[-1] - wavelength[0]) / len(wavelength),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=flux,
            mode="lines",
            name="Flux",
            line=dict(color="white", width=1.5),
        ),
        row=2, col=1,
    )

    if identified_lines:
        for line in identified_lines:
            fig.add_annotation(
                x=line["wavelength"],
                y=1.0,
                yref="y2",
                text=line["label"],
                showarrow=True,
                arrowhead=1,
                textangle=-90,
                font=dict(size=9),
                yshift=10,
            )

    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=2, col=1)

    fig.update_layout(
        title=title,
        height=420,
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=20),
    )

    return fig
