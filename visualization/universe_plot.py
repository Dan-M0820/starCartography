"""
STELLAR CARTOGRAPHY
3D Universe Visualization

Interactive 3D star field, with Earth fixed at the origin so that
distance from Earth can be read directly from the plot.
"""

import numpy as np
import plotly.graph_objects as go


def plot_universe_3d(stars: list) -> go.Figure:
    """
    Render the synthetic star population in 3D, with Earth marked
    at the origin, star color reflecting true temperature (via
    approximate blackbody color), and marker size reflecting
    (visually, not to physical scale) stellar radius.
    """

    xs = [s["position_ly"][0] for s in stars]
    ys = [s["position_ly"][1] for s in stars]
    zs = [s["position_ly"][2] for s in stars]
    colors = [s["color_hex"] for s in stars]
    ids = [s["id"] for s in stars]
    types = [s["star_type"] for s in stars]
    temps = [s["temperature_k"] for s in stars]
    distances = [s["distance_ly"] for s in stars]

    # Visual-only size scaling (sqrt to compress the large dynamic
    # range of radii into a readable marker size range).
    radii = np.array([s["radius_rsun"] for s in stars])
    sizes = 6 + 10 * np.sqrt(radii / radii.max())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.9,
                        line=dict(width=0.5, color="white")),
            customdata=np.stack([ids, types, temps, distances], axis=-1),
            hovertemplate=(
                "Star #%{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Temperature: %{customdata[2]:.0f} K<br>"
                "Distance: %{customdata[3]:.1f} ly"
                "<extra></extra>"
            ),
            name="Stars",
        )
    )

    # Earth, fixed at the origin.
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=8, color="#3b82f6", symbol="diamond"),
            text=["Earth"],
            textposition="top center",
            hovertemplate="Earth (reference point)<extra></extra>",
            name="Earth",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (light-years)",
            yaxis_title="Y (light-years)",
            zaxis_title="Z (light-years)",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=10),
        showlegend=False,
    )

    return fig
