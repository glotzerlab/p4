.. _sampling:

========
Sampling
========

.. p4 measures continuous fields at discrete positions and orientations, so fields
.. can never be measured with complete accuracy. Nevertheless, there are some steps
.. you can take to ensure that your measurements achieve the degree of accuracy
.. that you need.

Sampling Positions
++++++++++++++++++

The :py:meth:`~p4.system.System.measure` method accepts any number of positions
anywhere in 3D space. Plotting is only supported for fields with positions
sampled from a regular grid (i.e., the intervals between x, y, and z values
must be constant).

To generate positions on a regular grid, use
:py:func:`p4.positions_on_regular_grid`. Use sparse grids (20 - 50 positions
per side) when prototyping, and dense grids (100+ positions per side) when
creating figures for publication.

To evaluate the coverage of your position sampling, use
:py:func:`p4.plot_positions`. For example, see the cubic body below.

.. code-block:: python
    
    import p4
    import coxeter

    cube = coxeter.families.PlatonicFamily.get_shape("Cube")

    body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube.vertices)
    )

    sample_positions = p4.positions_on_regular_grid(
        box=[2, 2, 2],
        resolution=[10, 10, 10]
    )

    _, body_trace = body.plot(type_shapes=dict(A=cube))
    fig, _ = p4.plot_positions(positions=sample_positions, color="black", size=2)
    fig.add_traces(body_trace)
    fig.show()

.. raw:: html
    :file: ../data/sampling-positions.html


Sampling Orientations
+++++++++++++++++++++

The :py:meth:`~p4.system.System.measure` method accepts the following options
for orientation samples:

1. one orientation, which is broadcast across all positions

   * *(to be used when the probe is isotropic)*

2. an array of orientations, which is also broadcast across all positions

   * *(to be used when the probe is anisotropic, and orientations must be
     sampled identically at every position)*

3. an array of arrays of orientations of the same length as the positions

   * *(to be used when the probe is anisotropic and the user wants to sample
     orientations differently at different positions)*

Option 1 is trivial (just use ``orientations=[1, 0, 0, 0]``). Convenience
functions are provided for option 2:

* :py:func:`p4.orientations_about_axis` is suitable when a system's analyte and
  probe are affectively 2D. This function generates orientations represented by
  angles evenly spaced around a rotation axis. When the probe has a rotational
  symmetry about that same axis, the ``k`` argument can be used to reduce the
  required number of orientations.
* :py:func:`p4.orientations_from_fibonacci_lattice` is suitable in all other
  cases. This function generates orientations that are (approximately) evenly
  distributed across orientation-space.

Sampling many orientations at every position quickly becomes very
computationally expensive. Use the function
:py:func:`p4.plot_field_vs_number_of_orientations` to see how field values at
various positions change depending on the number of sampled orientations.

*This analysis is especially important when creating figures for publication*,
since it is easy to create data that appear high-resolution (due to dense
position sampling), but are actually low-accuracy (due to sparse orientation
sampling).

We can apply this analysis to our cubic body example from above.

.. code-block:: python

    other_body = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube.vertices)
    )

    alj = p4.Interaction(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args={},
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            shape=dict(vertices=[], faces=[])
        ),
        typed_params={
            ("C", "A"): dict(
                r_cut=5,
                params=dict(epsilon=1, sigma_i=0.1, sigma_j=0.1, alpha=0)
            ),
            "A": dict(shape=dict(vertices=cube.vertices, faces=cube.faces)),
            "C": dict(shape=dict(vertices=cube.vertices, faces=cube.faces))
        }
    )
    gauss = p4.Interaction(
        hoomd_class=hoomd.md.pair.Gaussian,
        initial_args={},
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=0.2),
        ),
        typed_params={
            ("D", "B"): dict(
                r_cut=5,
                params=dict(epsilon=-1, sigma=0.125)
            )
        }
    )

    system = p4.System(
        probe=other_body,
        analyte=body,
        interactions=[alj, gauss]
    )

    fig, _ = p4.plot_field_vs_number_of_orientations(
        system=system,
        n_orientations=np.arange(100, 600, 25),
        positions=[[float(i)]*3 for i in np.linspace(1, 1.25, 6)],
        method="fibonacci_lattice",
    )
    fig.show()

.. raw:: html
    :file: ../data/sampling-orientations.html
