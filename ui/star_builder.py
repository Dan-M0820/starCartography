"""
STELLAR CARTOGRAPHY
Star Builder

A single, interactive star. Instead of exploring a generated
field, here you build one star by hand:

    Temperature   --> star color (blackbody) + peak wavelength (Wien's Law)
    Luminosity    --> brightness / glow
    Temperature + Luminosity --> Radius (Stefan-Boltzmann Law, solved for R)
    Mass          --> surface gravity, approx. main-sequence lifetime,
                       and a comparison against real mass-luminosity /
                       mass-radius scaling relations
    Composition   --> which elements/molecules appear as absorption
                       lines in the spectrum

The star graphic and spectrum below update live from these values,
using the same physics modules as the Star Cartography tab (Planck's
Law, Wien's Law, Stefan-Boltzmann Law, and the blackbody color model).
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core.color_mapping import blackbody_temperature_to_rgb, rgb_to_hex
from core.spectrum_generator import WAVELENGTH_RANGE
from core.star_types import star_type_from_temperature
from physics.blackbody import planck_law, wien_peak_wavelength
from physics.stellar_properties import (
    SOLAR_LUMINOSITY_W,
    meters_to_solar_radii,
    radius_from_stefan_boltzmann,
)
from visualization.spectrum_plot import plot_absorption_spectrum


# -----------------------------------------------------------------------
# GAS / ABSORPTION LINE LIBRARY
#
# Same real spectral lines used in core/star_types.py, regrouped by
# element/molecule so they can be toggled independently of temperature
# here (in the main tab, a star's lines are fixed by its type; here
# the user chooses the composition directly).
# -----------------------------------------------------------------------

GAS_LINES = {
    "Hydrogen": [410.2, 434.0, 486.1, 656.3],
    "Helium": [447.1, 468.6, 492.2],
    "Ionized Calcium": [393.4, 396.8],
    "Magnesium": [517.3],
    "Sodium": [589.3],
    "Titanium Oxide": [495.7, 545.0, 620.0, 705.0],
}

DEFAULT_GASES_BY_TYPE = {
    "Helium star": ["Helium", "Hydrogen"],
    "Hydrogen star": ["Hydrogen"],
    "Calcium star": ["Ionized Calcium", "Magnesium", "Sodium"],
    "Molecular star": ["Titanium Oxide"],
}


# -----------------------------------------------------------------------
# SPECTRUM GENERATION (custom composition, not tied to star type)
# -----------------------------------------------------------------------

def _generate_custom_spectrum(temperature_k, line_wavelengths, rng):
    """
    Same pipeline as core/spectrum_generator.generate_spectrum
    (Planck continuum -> absorption lines -> noise), but the line
    list is whatever the user picked rather than being looked up
    from the star's temperature-derived type.
    """

    continuum = planck_law(WAVELENGTH_RANGE, temperature_k)
    continuum = continuum / continuum.max()

    flux = continuum.copy()
    for center in line_wavelengths:
        varied_center = center + rng.normal(0, 0.6)
        depth = rng.uniform(0.3, 0.55)
        width = rng.uniform(2.0, 4.0)
        absorption = depth * np.exp(
            -0.5 * ((WAVELENGTH_RANGE - varied_center) / width) ** 2
        )
        flux = flux * (1.0 - absorption)

    flux = flux + rng.normal(0, 0.012, size=flux.shape)
    flux = np.clip(flux, 0.0, None)
    return flux


# -----------------------------------------------------------------------
# STAR GRAPHIC (SVG, regenerated every rerun from current parameters)
# -----------------------------------------------------------------------

def _lighten(hex_color, amount=0.7):
    """Blend a hex color toward white -- used for the hot inner core."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_star_svg(temperature_k, luminosity_lsun, radius_rsun):
    """
    Build a self-contained SVG of the star: a soft blurred glow sized
    by luminosity, four diffraction-style spikes (as seen in real
    long-exposure star photography), and a crisp bright core sized by
    the physically derived radius -- all colored by the temperature's
    blackbody color. A slow pulse animation keeps it feeling alive.
    """

    color = rgb_to_hex(blackbody_temperature_to_rgb(temperature_k))
    core_color = _lighten(color, 0.75)

    log_l = float(np.log10(max(luminosity_lsun, 1e-6)))
    log_r = float(np.log10(max(radius_rsun, 1e-4)))

    core_px = float(np.clip(30 + 14 * (log_r + 2), 18, 150))
    glow_px = float(np.clip(core_px * (1.6 + 0.35 * (log_l + 4)), core_px * 1.3, 420))
    spike_len = float(np.clip(glow_px * 1.35, 60, 480))
    spike_width = float(np.clip(core_px * 0.12, 2, 14))

    view = 900
    cx = cy = view / 2
    pulse_seconds = float(np.clip(5.5 - temperature_k / 12000, 2.2, 5.0))

    svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;">
    <svg width="{view}" height="{view}" viewBox="0 0 {view} {view}"
         xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
      <defs>
        <radialGradient id="bg" cx="50%" cy="50%" r="75%">
          <stop offset="0%" stop-color="#0b0e1a"/>
          <stop offset="100%" stop-color="#02030a"/>
        </radialGradient>
        <radialGradient id="glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{core_color}" stop-opacity="0.9"/>
          <stop offset="35%" stop-color="{color}" stop-opacity="0.5"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="core" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#ffffff"/>
          <stop offset="30%" stop-color="{core_color}"/>
          <stop offset="100%" stop-color="{color}"/>
        </radialGradient>
        <linearGradient id="spike" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="{color}" stop-opacity="0"/>
          <stop offset="50%" stop-color="{core_color}" stop-opacity="0.85"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </linearGradient>
        <filter id="blurGlow"><feGaussianBlur stdDeviation="14"/></filter>
        <filter id="blurSpike"><feGaussianBlur stdDeviation="5"/></filter>
      </defs>

      <rect x="0" y="0" width="{view}" height="{view}" fill="url(#bg)"/>

      <style>
        .pulse {{
          animation: pulse {pulse_seconds}s ease-in-out infinite;
          transform-origin: {cx}px {cy}px;
        }}
        @keyframes pulse {{
          0%, 100% {{ opacity: 1; transform: scale(1); }}
          50%      {{ opacity: 0.90; transform: scale(1.035); }}
        }}
      </style>

      <g class="pulse">
        <circle cx="{cx}" cy="{cy}" r="{glow_px}" fill="url(#glow)" filter="url(#blurGlow)"/>
        <g filter="url(#blurSpike)">
          <rect x="{cx - spike_len}" y="{cy - spike_width / 2}"
                width="{spike_len * 2}" height="{spike_width}" fill="url(#spike)"/>
          <rect x="{cx - spike_width / 2}" y="{cy - spike_len}"
                width="{spike_width}" height="{spike_len * 2}" fill="url(#spike)"/>
          <rect x="{cx - spike_len * 0.6}" y="{cy - spike_width * 0.4}"
                width="{spike_len * 1.2}" height="{spike_width * 0.8}"
                fill="url(#spike)" transform="rotate(45 {cx} {cy})"/>
          <rect x="{cx - spike_len * 0.6}" y="{cy - spike_width * 0.4}"
                width="{spike_len * 1.2}" height="{spike_width * 0.8}"
                fill="url(#spike)" transform="rotate(-45 {cx} {cy})"/>
        </g>
        <circle cx="{cx}" cy="{cy}" r="{core_px}" fill="url(#core)"/>
      </g>
    </svg>
    </div>
    """
    return svg


# -----------------------------------------------------------------------
# MAIN RENDER FUNCTION (called from app.py inside its own tab)
# -----------------------------------------------------------------------

def render():
    st.title("Star Builder")
    st.caption(
        "Build a single star by hand. Set its temperature, luminosity, "
        "mass, and which elements/molecules show up in its spectrum -- "
        "the star and its absorption spectrum below respond live, using "
        "the same physics as the Star Cartography tab."
    )

    with st.expander("How this works", expanded=False):
        st.markdown(
            "This runs the same physics as the other tab, in the "
            "**forward** direction (parameters -> star), instead of the "
            "reverse (spectrum -> parameters):"
        )
        st.code(
            "Temperature\n"
            "  +-- blackbody color (star's visible color)\n"
            "  +-- peak wavelength, Wien's Law: lambda_max = b / T\n\n"
            "Luminosity + Temperature (Stefan-Boltzmann Law, solved for R)\n"
            "  --> Radius:  R = sqrt( L / (4*pi*sigma*T^4) )\n\n"
            "Mass + Radius --> Surface gravity (relative to the Sun)\n"
            "Mass + Luminosity --> Approx. main-sequence lifetime\n\n"
            "Chosen elements/molecules --> absorption lines embedded\n"
            "  directly in the spectrum (real wavelengths, same table\n"
            "  used to identify star types in the other tab)",
            language=None,
        )

    col_controls, col_star = st.columns([1, 1.25], gap="large")

    with col_controls:
        st.markdown("##### Temperature & Light")
        temperature_k = st.slider(
            "Temperature (K)", 2500, 45000, 5772, step=50, key="sb_temperature"
        )
        peak_wavelength = wien_peak_wavelength(temperature_k)
        if peak_wavelength < 380:
            band_note = "in the ultraviolet, beyond visible light"
        elif peak_wavelength > 750:
            band_note = "in the infrared, beyond visible light"
        else:
            band_note = "within the visible range"
        st.caption(
            f"Peak emission wavelength (Wien's Law): "
            f"**{peak_wavelength:.0f} nm** -- {band_note}."
        )

        log_l = st.slider(
            "Luminosity (log\u2081\u2080 L\u2609)", -3.0, 6.0, 0.0, step=0.1,
            key="sb_log_luminosity",
        )
        luminosity_lsun = 10 ** log_l
        st.caption(f"Luminosity: **{luminosity_lsun:,.4g} L\u2609**")

        mass_msun = st.slider(
            "Mass (M\u2609)", 0.05, 60.0, 1.0, step=0.05, key="sb_mass"
        )

        st.markdown("##### Composition")
        star_type = star_type_from_temperature(temperature_k)
        selected_gases = st.multiselect(
            "Elements / molecules present in the absorption spectrum",
            options=list(GAS_LINES.keys()),
            default=DEFAULT_GASES_BY_TYPE[star_type],
            key="sb_gases",
        )

    # ---------------------------------------------------------------
    # Physics (forward direction)
    # ---------------------------------------------------------------
    luminosity_w = luminosity_lsun * SOLAR_LUMINOSITY_W
    radius_m = radius_from_stefan_boltzmann(luminosity_w, temperature_k)
    radius_rsun = meters_to_solar_radii(radius_m)

    surface_gravity_rel = mass_msun / max(radius_rsun, 1e-6) ** 2
    lifetime_gyr = 10.0 * (mass_msun / max(luminosity_lsun, 1e-6))

    mass_predicted_radius_rsun = mass_msun ** 0.8
    mass_predicted_luminosity_lsun = mass_msun ** 3.5

    color_hex = rgb_to_hex(blackbody_temperature_to_rgb(temperature_k))

    with col_star:
        components.html(
            _render_star_svg(temperature_k, luminosity_lsun, radius_rsun),
            height=520,
            scrolling=False,
        )
        st.markdown(
            f"<div style='text-align:center;font-weight:600;"
            f"color:{color_hex};text-shadow:0 0 6px {color_hex}88;'>"
            f"{star_type} &middot; {temperature_k:,} K</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Derived Properties")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Radius (Stefan\u2013Boltzmann)", f"{radius_rsun:.3g} R\u2609")
    c2.metric("Surface Gravity (rel. to Sun)", f"{surface_gravity_rel:.2f}\u00d7")
    c3.metric("Approx. Main-Sequence Lifetime", f"{lifetime_gyr:.2f} Gyr")
    c4.metric("Spectral Class (simplified)", star_type)

    with st.expander("Compare against a real main-sequence star of this mass"):
        st.caption(
            "Rough power-law approximations for main-sequence stars "
            "(R \u221d M^0.8, L \u221d M^3.5), shown only as a sanity check "
            "against the radius/luminosity you set directly above -- your "
            "star doesn't have to sit on the main sequence."
        )
        cc1, cc2 = st.columns(2)
        cc1.metric("Main-sequence-predicted Radius", f"{mass_predicted_radius_rsun:.2f} R\u2609")
        cc2.metric(
            "Main-sequence-predicted Luminosity",
            f"{mass_predicted_luminosity_lsun:,.3g} L\u2609",
        )

    st.divider()
    st.markdown("#### Absorption Spectrum")

    gas_names = list(GAS_LINES.keys())
    seed = int(temperature_k * 10) % (2**31)
    seed += sum((gas_names.index(g) + 1) * 97 for g in selected_gases)
    rng = np.random.default_rng(seed % (2**31))

    line_wavelengths = [wl for gas in selected_gases for wl in GAS_LINES[gas]]
    flux = _generate_custom_spectrum(temperature_k, line_wavelengths, rng)

    identified_lines = [
        {"wavelength": wl, "label": gas}
        for gas in selected_gases
        for wl in GAS_LINES[gas]
    ]

    st.plotly_chart(
        plot_absorption_spectrum(
            WAVELENGTH_RANGE,
            flux,
            identified_lines=identified_lines,
            title="Your Star's Absorption Spectrum",
        ),
        use_container_width=True,
    )

    if not selected_gases:
        st.info(
            "No elements/molecules selected -- this shows the pure "
            "blackbody continuum with no absorption lines."
        )


def main():
    """Standalone entry point, in case this tab is ever run on its own."""
    st.set_page_config(page_title="Star Builder", layout="wide")
    render()


if __name__ == "__main__":
    main()
