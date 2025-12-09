from typing import Literal
import probe_potential as pp
import hoomd

import util


# TODO: add pixel exclusion
# TODO: add results processing
# TODO: add parallelization for a single statepoint

def cube_verts():
    verts = [(-0.5, -0.5, -0.5),
             (-0.5, -0.5,  0.5),
             (-0.5,  0.5, -0.5),
             (-0.5,  0.5,  0.5),
             ( 0.5, -0.5, -0.5),
             ( 0.5, -0.5,  0.5),
             ( 0.5,  0.5, -0.5),
             ( 0.5,  0.5,  0.5)]
    return verts

def cube_faces():
    faces = [[0, 2, 6],
              [6, 4, 0],
              [5, 0, 4],
              [5,1,0],
              [5,4,6],
              [5,6,7],
              [3,2,0],
              [3,0,1],
              [3,6,2],
              [3,7,6],
              [3,1,5],
              [3,5,7]]
    return faces

def square_verts():
    verts = [
        [-0.5, -0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [ 0.5, -0.5, 0.0]
    ]
    return verts



def process_csv(
    csv_filename: str,
    handle_orientation: Literal["mean", "min"]
):
    """Revise CSV file, preparing it for plotting.
    
    This processing has the following steps:
    1. Rename columns
    2. Average over orientations and then drop orientation columns.
    3. Add rows and columns to ensure a completely uniform X/Y/Z grid.
    4. Change all NaN values to Inf

    Parameters
    ----------
    csv_filename : str
        The path to the CSV file to process.
    handle_orientation : 'mean' or 'min'
        Whether to average over all orientations or take the minimum potential.
    """
    df = pd.read_csv(csv_filename)

    # Rename columns
    new_names = {
        "       x        ": "x",
        "       y        ": "y",
        "       z        ": "z",
        "       q0       ": "q0",
        "       q1       ": "q1",
        "       q2       ": "q2",
        "       q3       ": "q3",
        "Simulation.timestep": "t",
        "md.compute.ThermodynamicQuantities.potential_energy": "PE"
    }
    df.rename(columns=new_names, inplace=True)

    # Average over orientation, then drop orientation columns
    if handle_orientation == "mean":
        df = df.groupby(["x", "y", "z"]).mean()
    elif handle_orientation == "min":
        df = df.groupby(["x", "y", "z"]).min()
    else:
        raise ValueError("handle_orientation must be 'mean' or 'min'.")
    
    df = df.reset_index(level=[0,1,2])
    df = df.drop(labels=["q0", "q1", "q2", "q3"], axis=1)

    # Ensure a completely uniform grid
    # df = enforce_uniform_grid(df)   # TODO: fix 2 so that I can use it here

    # Replace NaN with Inf
    df = df.replace(to_replace=np.nan, value=np.inf)

    # Re-save processed log table
    df.to_csv(csv_filename.split(".")[-2] + "_processed.csv", mode="w")



def test_lj_sphere():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel("A")

    interaction_model = {
        "LJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=1.0,
                    sigma=0.5
                ),
                r_cut=2.0
            ),
            probe_cutoff_outside_callable=lambda: [2,2,2],
            probe_cutoff_inside_callable=lambda: [0,0,0]
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        orientation_symmetries= [1, 1, 1],
        nlist=nlist,
        csv_filename="test-lj-sphere.csv"
    )

    process_csv("test-lj-sphere.csv", "mean")

def test_lj_sites():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel(
        primary_type="A",
        secondary_types=["B"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )

    interaction_model = {
        "LJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            yes_types=["B", "P"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma=0.2
                ),
                r_cut=2.0
            ),
            probe_cutoff_outside_callable=lambda: [2,2,2],
            probe_cutoff_inside_callable=lambda: [0,0,0]
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        orientation_symmetries= [1, 1, 1],
        nlist=nlist,
        csv_filename="test-lj-sites.csv",
        gsd_filename="test-lj-sites.gsd",
    )

    process_csv("test-lj-sites.csv", "mean")

def test_alj_cube():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel("A")

    interaction_model = {
        "ALJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(
                shape=dict(
                    vertices=cube_verts(),
                    faces=cube_faces()
                )
            ),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=2.0
            ),
            probe_cutoff_outside_callable=lambda: [2,2,2],
            probe_cutoff_inside_callable=lambda: [0,0,0]
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 10],
        orientation_symmetries= [1, 1, 4],
        nlist=nlist,
        csv_filename="test-alj-cube.csv",
        gsd_filename="test-alj-cube.gsd"
    )

    process_csv("test-alj-cube.csv", "mean")

def test_alj_cube_with_eg_sites():
    probe_model = pp.ParticleModel(
        primary_type="P",
        secondary_types=["P2"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )
    analyte_model = pp.ParticleModel(
        primary_type="A",
        secondary_types=["B"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )

    interaction_model = {
        "ALJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(
                shape=dict(
                    vertices=cube_verts(),
                    faces=cube_faces()
                )
            ),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=2.0
            ),
            probe_cutoff_outside_callable=lambda: [2,2,2],
            probe_cutoff_inside_callable=lambda: [0,0,0]
        ),
        "EG": pp.Interaction(
            hoomd_class=hoomd.md.pair.ExpandedGaussian,
            initial_inputs=dict(
                default_r_cut=1.0,
                default_r_on=0.0,
            ),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=0.1,
                    delta=0.0
                ),
                r_cut=0.0
            ),
            yes_types=["B", "P2"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=-1.0,
                    sigma=0.1,
                    delta=0.2
                ),
                r_cut=2.0
            ),
            probe_cutoff_outside_callable=lambda: [2,2,2],
            probe_cutoff_inside_callable=lambda: [0,0,0]
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 4],
        orientation_symmetries= [1, 1, 4],
        nlist=nlist,
        csv_filename="test-alj-cube-with-eg-sites.csv",
        gsd_filename="test-alj-cube-with-eg-sites.gsd",
    )

    process_csv("test-alj-cube-with-eg-sites.csv", "mean")

if __name__ == "__main__":
    # test_lj_sphere()              # looks good
    # test_lj_sites()               # looks good
    # test_alj_cube()               # looks good
    test_alj_cube_with_eg_sites()   # looks good
