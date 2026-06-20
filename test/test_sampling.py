# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import pytest
import numpy as np
import rowan
import scipy

import p4


@pytest.mark.parametrize("box", [[3, 3, 3], [4, 4, 4]])
@pytest.mark.parametrize("resolution", [[2, 2, 2], [3, 3, 3], [4, 4, 4]])
def test_positions_on_regular_grid(box, resolution):
    """Ensure that sampled positions lie on a regular grid."""
    positions = p4.positions_on_regular_grid(box, resolution)

    # Check shape
    assert positions.shape == (int(np.prod(resolution)), 3)
    
    # Check min and max
    for i, L in enumerate(box):
        assert positions[:,i].min() == -L/2
        assert positions[:,i].max() == L/2
    
    # Check number of values
    for i, n in enumerate(resolution):
        assert len(np.unique(positions[:,i])) == n
    
    # Check that spacing is constant
    for i in [0, 1, 2]:
        unique_values = np.unique(positions[:,i])
        second_diff = np.diff(unique_values, n=2)
        assert (
            len(second_diff) == 0
            or np.isclose(second_diff, [0 for _ in second_diff]).all()
        )

@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_orientations_about_axis(n, axis, k):
    """Ensure that sampled orientations are correctly distributed."""
    orientations = p4.orientations_about_axis(n, axis, k)
    axes, angles = rowan.to_axis_angle(orientations)

    # Check shape
    assert n == orientations.shape[0]

    # Check axes are all correct (special case: [1,0,0,0] -> [0,0,0], [0])
    for i, a in enumerate(axes):
        assert (
            (
                np.isclose(a, [0, 0, 0]).all()
                and np.array_equal(orientations[i], [1, 0, 0, 0])
            )
            or np.isclose(a, axis).all()
        )

    # Check for correct min and max angles
    assert angles.min() == 0
    assert np.isclose(angles.max(), (2*np.pi / k) * (1 - (1 / n)))

    # Check that spacing is constant
    second_diff = np.diff(angles, n=2)
    assert (
        len(second_diff) == 0
        or np.isclose(second_diff, [0 for _ in second_diff]).all()
    )

def calculate_volumes(sv):
    """Calculate the hypervolumes of the cells of a 4D spherical voronoi."""
    volumes = []
    vertices = sv.vertices
    for indices in sv.regions:
        volumes.append(scipy.spatial.ConvexHull(vertices[indices]).volume)
    return volumes

def test_orientations_from_fibonacci_lattice_no_group():
    """Ensure that the fibonacci lattice yields orientations that approximately evenly spaced.
    
    This test calculates the spherical vornoi diagram of the quaternions,
    computes the volume of each vornoi cell, and then checks that the relative
    standard deviation of those volumes is smaller than some threshold.
    """
    n = 100
    qs = p4.orientations_from_fibonacci_lattice(n)
    sv = scipy.spatial.SphericalVoronoi(qs)
    vs = calculate_volumes(sv)
    mean = np.mean(vs)
    rsd = np.std(vs, mean=mean) / mean
    assert rsd < 0.3

if __name__ == "__main__":
    # Generate plots of hypervolume SD and RSD
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ns = np.arange(50, 2000, 50)
    sds = []
    rsds = []

    for n in ns:
        qs = p4.orientations_from_fibonacci_lattice(n)
        sv = scipy.spatial.SphericalVoronoi(qs)
        vs = calculate_volumes(sv)
        mean = np.mean(vs)
        sd = np.std(vs, mean=mean)
        rsd = sd / mean
        
        sds.append(sd)
        rsds.append(rsd)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=ns, y=rsds, name="RSD"),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=ns, y=sds, name="SD"),
        secondary_y=False
    )

    fig.update_layout(
        title_text="Distribution of orientation spacing vs number of samples",
        xaxis_title_text="Number of samples",
    )
    fig.update_yaxes(title_text="RSD of hypervolume", secondary_y=True),
    fig.update_yaxes(title_text="SD of hypervolume", secondary_y=False),

    fig.update_layout(template="simple_white")

    fig.show()