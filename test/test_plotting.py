import coxeter
import pytest
from copy import copy
import p4
import hoomd
import numpy as np
from pathlib import Path

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

CUBE_VERTICES = [
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
]
CUBE_FACES = [
    [0, 2, 6, 4],
    [0, 4, 5, 1],
    [4, 6, 7, 5],
    [0, 1, 3, 2],
    [2, 3, 7, 6],
    [1, 5, 7, 3],
]
OCTAHEDRON_VERTICES = [
    [ 0,  0, -1],
    [-1, -1,  0],
    [ 1, -1,  0],
    [ 1,  1,  0],
    [-1,  1,  0],
    [ 0,  0,  1],
]
BODY = p4.Body(
    primary_type="A",
    secondary_types=["B", "C"],
    positions_by_type=dict(
        B=CUBE_VERTICES,
        C=OCTAHEDRON_VERTICES,
    )
)

@pytest.mark.parametrize("type_shapes", [{}, {"A": coxeter.shapes.ConvexPolyhedron(CUBE_VERTICES)}])
@pytest.mark.parametrize("type_styles", [{}, {"A": dict(color="yellow")}])
@pytest.mark.parametrize("ignore_types", [[], ["C"]])
@pytest.mark.parametrize("slice", [{}, dict(z=0), dict(z=0, y=0)])
@pytest.mark.parametrize("schematic_slice", [False, True])
@pytest.mark.parametrize("schematic_slice_scale", [1])
@pytest.mark.parametrize("schematic_slice_color", ["red"])
@pytest.mark.parametrize("schematic_slice_opacity", [1])
@pytest.mark.parametrize("schematic_slice_line_width", [10])
@pytest.mark.parametrize("show_legend", [True])
def test_body_plot(
    type_shapes,
    type_styles,
    ignore_types,
    slice,
    schematic_slice,
    schematic_slice_scale,
    schematic_slice_color,
    schematic_slice_opacity,
    schematic_slice_line_width,
    show_legend,
):
    """Ensure body plotting does not error for valid parameters."""
    _, _ = BODY.plot(
        type_shapes=type_shapes,
        type_styles=type_styles,
        ignore_types=ignore_types,
        slice=slice,
        schematic_slice=schematic_slice,
        schematic_slice_scale=schematic_slice_scale,
        schematic_slice_color=schematic_slice_color,
        schematic_slice_opacity=schematic_slice_opacity,
        schematic_slice_line_width=schematic_slice_line_width,
        show_legend=show_legend,
    )

FIELD = p4.Field.from_csv(str(Path(__file__).parent / "data/field-uft.csv"))
FIELD = FIELD.subset(q0=1.0, q1=0.0, q2=0.0, q3=0.0)

@pytest.mark.parametrize("quantity", ["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("slice", [dict(), dict(z=0), dict(z=0, y=0)])
@pytest.mark.parametrize("clim", [None, [-1, 1]])
@pytest.mark.parametrize("contours", [None, 12])
@pytest.mark.parametrize("cmap", ["Viridis"])
@pytest.mark.parametrize("fill_nan_with_inf", [False])
@pytest.mark.parametrize("show_cbar", [True])
@pytest.mark.parametrize("show_axes", [True])
@pytest.mark.parametrize("show_title", [True])
@pytest.mark.parametrize("show_ticks", [True])
@pytest.mark.parametrize("show_grid", [False])
@pytest.mark.parametrize("show_border", [True])
@pytest.mark.parametrize("marker_mode_1d", ["markers+lines"])
@pytest.mark.parametrize("marker_color_1d", ["red"])
@pytest.mark.parametrize("marker_size_1d", [10])
@pytest.mark.parametrize("line_width_1d", [4])
def test_field_plot(
    quantity,
    vectors,
    slice,
    clim,
    contours,
    cmap,
    fill_nan_with_inf,
    show_cbar,
    show_axes,
    show_title,
    show_ticks,
    show_grid,
    show_border,
    marker_mode_1d,
    marker_color_1d,
    marker_size_1d,
    line_width_1d,
):
    """Ensure field plotting does not error for valid parameters."""
    # Skip cases where contours is None and slice is 3D
    if len(slice) == 0 and contours is None:
        return
    
    # Skip cases where slice is 1D and vectors is True
    if len(slice) == 2 and vectors:
        return
    
    _, _ = FIELD.plot(
        quantity=quantity,
        vectors=vectors,
        slice=slice,
        clim=clim,
        contours=contours,
        cmap=cmap,
        fill_nan_with_inf=fill_nan_with_inf,
        show_cbar=show_cbar,
        show_axes=show_axes,
        show_title=show_title,
        show_ticks=show_ticks,
        show_grid=show_grid,
        show_border=show_border,
        marker_mode_1d=marker_mode_1d,
        marker_color_1d=marker_color_1d,
        marker_size_1d=marker_size_1d,
        line_width_1d=line_width_1d,
    )
