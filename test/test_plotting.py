import pytest
from copy import copy
import p4
import hoomd
import numpy as np

INTERACTION = p4.Interaction(
    hoomd_class=hoomd.md.pair.LJ,
    initial_args={},
    default_params=dict(r_cut=0, params=dict(epsilon=1, sigma=1)),
    typed_params={
        ("A", "B"): dict(r_cut=5, params=dict(epsilon=0.5, sigma=0.5)),
        ("A", "C"): dict(r_cut=6, params=dict(epsilon=0.25, sigma=0.25)),
    }

)

@pytest.mark.parametrize("type_pairs", [None, [("A", "B")]])
@pytest.mark.parametrize("pair_styles", [{}, {"default": dict(color="black")}])
@pytest.mark.parametrize("cmap", [None, "plotly"])
@pytest.mark.parametrize("ylim", [None, [0, 1]])
@pytest.mark.parametrize("exclude_default", [False, True])
@pytest.mark.parametrize("marker_size", [10])
@pytest.mark.parametrize("line_width", [4])
@pytest.mark.parametrize("mode", ["markers+lines"])
@pytest.mark.parametrize("show_axes", [True])
@pytest.mark.parametrize("show_ticks", [False])
@pytest.mark.parametrize("show_grid", [False])
@pytest.mark.parametrize("show_border", [True])
@pytest.mark.parametrize("width", [500])
@pytest.mark.parametrize("height", [500])
def test_interaction_plot(
    type_pairs,
    pair_styles,
    cmap,
    ylim,
    exclude_default,
    marker_size,
    line_width,
    mode,
    show_axes,
    show_ticks,
    show_grid,
    show_border,
    width,
    height,
):
    """Ensure interaction plotting does not error for valid parameters."""
    fig, tr = INTERACTION.plot(
        r=np.linspace(0, 10, 10),
        type_pairs=type_pairs,
        pair_styles=pair_styles,
        cmap=cmap,
        ylim=ylim,
        exclude_default=exclude_default,
        marker_size=marker_size,
        line_width=line_width,
        mode=mode,
        show_axes=show_axes,
        show_ticks=show_ticks,
        show_grid=show_grid,
        show_border=show_border,
        width=width,
        height=height,
    )