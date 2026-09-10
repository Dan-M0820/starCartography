"""
STELLAR CARTOGRAPHY
Application Entry Point

Two tabs, same project:
  1. Star Cartography -- explore a generated star field, pick a star,
     and derive its properties from its spectrum (unchanged).
  2. Star Builder -- build a single star by hand and watch it respond
     live to temperature, luminosity, mass, and composition.

Run with:
    streamlit run app.py
"""

import streamlit as st

from ui.dashboard import render as render_cartography
from ui.star_builder import render as render_star_builder

st.set_page_config(page_title="Stellar Cartography", layout="wide")

tab_cartography, tab_builder = st.tabs(["\U0001F30C Star Cartography", "\u2B50 Star Builder"])

with tab_cartography:
    render_cartography()

with tab_builder:
    render_star_builder()
