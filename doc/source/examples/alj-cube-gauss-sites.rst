^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ALJ Cube with Gaussian Sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    import p4
    import hoomd

    probe = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube_vertices(2))
    )
    analyte = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube_vertices(2))
    )

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma_i=1,
                    sigma_j=1,
                    alpha=0
                ),
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            typed_params={
                ("A", "C"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=2,
                        sigma_i=2,
                        sigma_j=2,
                        alpha=0
                    )
                ),
                "A": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                ),
                "C": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                )
            }
        ),
        p4.Interaction(
            hoomd_class=hoomd.md.pair.Gaussian,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma=1,
                )
            ),
            typed_params={
                ("B", "D"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=-5,
                        sigma=0.25,
                    )
                )
            }
        )
    ]

    system = p4.System(probe, analyte, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.measure(
        quantities=["U", "F", "T"],
        position_resolutions=[10, 10, 10],
        orientation_resolutions=[1, 1, 5],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube-with-gauss-sites-uft.csv",
        outside_cutoff=10,
        n_processes=n_processes,
    )

    field = p4.Field.from_csv("alj-cube-with-gauss-sites-uft.csv")
    aggregated_field = field.aggregate_over_orientations("U", "mean")

    figure, _ = aggregated_field.plot(
        clim=[-1, 1],
        slice=dict(z=0),
        contours=None,
        fill_nan_with_inf=True
    )
    figure.show()

.. raw:: html
   :file: ../_static/alj-cube-with-gauss-sites-uft.html

