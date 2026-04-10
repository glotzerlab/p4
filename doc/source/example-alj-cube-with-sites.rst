=====================================
Probe an ALJ cube with gaussian sites
=====================================

.. code-block:: python

    import p4
    import hoomd

    def cube_vertices(side_length):
        s = side_length
        return [
            [-s/2, -s/2, -s/2],
            [-s/2, -s/2,  s/2],
            [-s/2,  s/2, -s/2],
            [-s/2,  s/2,  s/2],
            [ s/2, -s/2, -s/2],
            [ s/2, -s/2,  s/2],
            [ s/2,  s/2, -s/2],
            [ s/2,  s/2,  s/2]
        ]

    def cube_faces():
        return [
            [0, 2, 6, 4],
            [0, 4, 5, 1],
            [4, 6, 7, 5],
            [0, 1, 3, 2],
            [2, 3, 7, 6],
            [1, 5, 7, 3],
        ]

    probe = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=get_cube_vertices(2))
    )
    analyte = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=get_cube_vertices(2))
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
                        vertices=get_cube_vertices(2),
                        faces=get_cube_faces()
                    )
                ),
                "C": dict(
                    shape=dict(
                        vertices=get_cube_vertices(2),
                        faces=get_cube_faces()
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

    system.probe_potential(
        position_resolutions=[100, 100, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube-with-gauss-sites.csv",
        outside_cutoff=10,
        n_processes=-1,
    )

    field = p4.Field.from_csv("alj-cube-with-gauss-sites.csv", "mean")
    field.plot(fill_nan_with_inf=True)

.. image:: ../assets/alj-cube-gauss-sites-plot.png
