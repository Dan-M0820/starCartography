"""
STELLAR CARTOGRAPHY
Streamlit Dashboard

A seeded, explorable 3D star field with Earth fixed at the
origin. Selecting a star shows its absorption-line spectrum, its
identified star type, and physical properties (temperature,
luminosity, radius) independently re-derived from that spectrum
and its known distance/brightness -- compared against the star's
true generated values.

Run with:
    streamlit run app.py
"""

import inspect

import numpy as np
import pandas as pd
import streamlit as st

from core.universe_generator import generate_universe
from core.spectrum_generator import generate_spectrum, WAVELENGTH_RANGE
from analysis.star_analysis import analyze_star
from visualization.universe_plot import plot_universe_3d
from visualization.spectrum_plot import plot_absorption_spectrum

NOISE_LEVEL = 0.015  # fixed; kept low so line identification stays reliable


@st.cache_data(show_spinner="Generating universe...")
def _build_universe(n_stars, max_distance_ly, seed):
    return generate_universe(n_stars=n_stars, max_distance_ly=max_distance_ly, seed=seed)


@st.cache_data(show_spinner=False)
def _build_spectra(_universe_key, n_stars, max_distance_ly, seed):
    """
    Generate each star's spectrum once, deterministically. _universe_key
    is unused except to key the cache alongside the same parameters
    used to build the universe.
    """
    universe = generate_universe(n_stars=n_stars, max_distance_ly=max_distance_ly, seed=seed)
    rng = np.random.default_rng(seed + 500)
    spectra = {}
    for star in universe:
        spectra[star["id"]] = generate_spectrum(
            star["temperature_k"], noise_level=NOISE_LEVEL, rng=rng
        )
    return spectra


def render():
    st.title("Stellar Cartography")
    st.caption(
        "A synthetic, explorable star field -- select a star to see its "
        "absorption-line spectrum and calculate its properties the way "
        "astronomers actually do: from light alone."
    )

    with st.expander("How this works", expanded=False):
        st.markdown(
            "Each star's **spectrum** carries dark absorption lines from "
            "the elements/molecules present at its temperature. "
            "Matching those lines identifies the **star type**. The "
            "spectrum's brightest wavelength gives an estimated "
            "**temperature** (Wien's Law). Combined with the star's "
            "known distance from Earth and its apparent brightness, the "
            "**luminosity** follows from the inverse-square law, and the "
            "**radius** follows from the Stefan-Boltzmann Law."
        )
        st.code(
            "Spectrum\n"
            "  +-- absorption lines --> Star type\n"
            "  +-- brightest wavelength (Wien's Law) --> Temperature\n\n"
            "Apparent brightness + distance from Earth (inverse-square law)\n"
            "  --> Luminosity\n\n"
            "Luminosity + Temperature (Stefan-Boltzmann Law)\n"
            "  --> Radius",
            language=None,
        )

    with st.sidebar:
        st.header("Universe Settings")
        n_stars = st.slider("Number of stars", 10, 80, 30, step=5)
        max_distance_ly = st.slider("Max distance (light-years)", 20, 150, 60, step=10)
        seed = st.number_input("Random seed", value=42, step=1)

    universe = _build_universe(n_stars, max_distance_ly, seed)
    spectra = _build_spectra(None, n_stars, max_distance_ly, seed)

    st.subheader("Star Field")
    st.caption(
        "Earth is fixed at the origin. Star color reflects true "
        "temperature; size reflects true radius (not to physical scale)."
    )

    fig_universe = plot_universe_3d(universe)

    clicked_id = None
    supports_selection = "on_select" in inspect.signature(st.plotly_chart).parameters

    if supports_selection:
        try:
            event = st.plotly_chart(
                fig_universe,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="universe_chart",
            )
            if event and event.get("selection", {}).get("points"):
                point = event["selection"]["points"][0]
                if point.get("customdata") is not None:
                    clicked_id = int(point["customdata"][0])
        except Exception:
            st.plotly_chart(fig_universe, use_container_width=True)
    else:
        st.plotly_chart(fig_universe, use_container_width=True)
        st.caption(
            "This Streamlit version does not support Plotly point "
            "selection -- use the dropdown below to select a star."
        )

    default_index = 0
    if clicked_id is not None:
        default_index = next(i for i, s in enumerate(universe) if s["id"] == clicked_id)

    selected_index = st.selectbox(
        "Select a star"
        + (" (or click a point on the map above)" if supports_selection else ""),
        options=list(range(len(universe))),
        index=default_index,
        format_func=lambda i: (
            f"Star #{universe[i]['id']} ({universe[i]['star_type']}, "
            f"{universe[i]['distance_ly']:.1f} ly away)"
        ),
    )

    star = universe[selected_index]
    raw_flux = spectra[star["id"]]
    result = analyze_star(star, raw_flux)

    st.divider()
    st.markdown(f"### Star #{star['id']} -- {star['star_type']} (true type)")

    st.plotly_chart(
        plot_absorption_spectrum(
            WAVELENGTH_RANGE,
            result["cleaned_flux"],
            identified_lines=[
                line for line in result["identified_lines"]
                if line["label"] != "Unidentified"
            ],
            title=f"Star #{star['id']} Absorption Line Spectrum",
        ),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Identified Type (from lines)", result["identified_star_type"] or "Unidentified")
    col2.metric("True Type", star["star_type"])
    col3.metric(
        "Type Match",
        "Yes" if result["type_matches_true"] else "No",
    )

    st.divider()
    st.markdown("#### Temperature")
    col4, col5, col6 = st.columns(3)
    col4.metric("True Temperature", f"{result['true_temperature_k']:.0f} K")
    col5.metric("Estimated (Wien's Law)", f"{result['estimated_temperature_k']:.0f} K")
    col6.metric("Error", f"{result['temperature_error_k']:.0f} K")

    if not result["temperature_estimate_reliable"]:
        st.info(
            "This star is hot enough that its true brightness peak lies "
            "in the ultraviolet, below this simulation's visible window "
            "(380-750 nm). Wien's Law can only report a **lower bound** "
            "here, not an exact temperature -- this is a genuine "
            "limitation of optical-only temperature estimation for hot "
            "stars, not an error. The line-identified type above (based "
            "on which elements are detected) is the more reliable "
            "indicator for stars this hot."
        )

    st.divider()
    st.markdown("#### Distance, Luminosity, and Radius")
    st.caption(
        "Distance is read directly from the 3D map (Earth is the fixed "
        "reference point) -- this mirrors how real astronomers use an "
        "independently measured distance (e.g. via parallax) rather than "
        "deriving it from the spectrum itself."
    )

    col7, col8, col9 = st.columns(3)
    col7.metric("Distance from Earth", f"{star['distance_ly']:.1f} ly")
    col8.metric("True Luminosity", f"{result['true_luminosity_lsun']:.2f} L\u2609")
    col9.metric("Estimated Luminosity", f"{result['estimated_luminosity_lsun']:.2f} L\u2609")
    st.caption(
        "Luminosity is derived from known distance + apparent brightness "
        "(inverse-square law) -- since both inputs are treated as known "
        "here, this recovers the true luminosity essentially exactly."
    )

    col10, col11, col12 = st.columns(3)
    col10.metric("True Radius", f"{star['radius_rsun']:.2f} R\u2609")
    col11.metric("Estimated Radius", f"{result['estimated_radius_rsun']:.2f} R\u2609")
    radius_error_pct = abs(
        result["estimated_radius_rsun"] - star["radius_rsun"]
    ) / star["radius_rsun"] * 100
    col12.metric("Error", f"{radius_error_pct:.0f}%")

    if not result["temperature_estimate_reliable"]:
        st.caption(
            "Radius is derived from luminosity and temperature "
            "(Stefan-Boltzmann Law). Since the temperature estimate above "
            "is only a lower bound for this star, the radius estimate is "
            "correspondingly unreliable -- an underestimated temperature "
            "inflates the calculated radius."
        )

    with st.expander("Detected absorption lines"):
        if result["identified_lines"]:
            line_table = pd.DataFrame([
                {
                    "Wavelength (nm)": round(line["wavelength"], 1),
                    "Depth": round(line["depth"], 3),
                    "Matched Element/Molecule": line["label"],
                }
                for line in result["identified_lines"]
            ])
            st.dataframe(line_table, use_container_width=True, hide_index=True)
        else:
            st.write("No absorption lines detected above the significance threshold.")


def main():
    """Standalone entry point (sets page config itself). When this tab is
    hosted alongside others (see app.py), app.py sets the page config once
    and calls render() directly instead."""
    st.set_page_config(page_title="Stellar Cartography", layout="wide")
    render()


if __name__ == "__main__":
    main()
