# Pairwise Potential Particle Probe

Probe the effective potential energy around a central particle using HOOMD-blue.


Potential paper titles:
    - Toward the rational design of pairwise particle interaction models


## Example usage
```python
# Probe the potential around a LJ sphere
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
        )
    )
}

system = pp.System(probe_model, analyte_model, interaction_model)

nlist = hoomd.md.nlist.Cell(10)

system.probe_potential(
    position_resolutions=[30, 30, 1],
    orientation_resolutions=[1, 1, 1],
    orientation_symmetries=[1, 1, 1],
    interactions_to_include=["LJ"],
    nlist=nlist,
    csv_filename="test-lj-sphere.csv",
    probe_cutoff_shape=None,
    probe_cutoff_outside_distance=lambda _: 2.0,
    probe_cutoff_inside_distance=lambda _: 0.2
)
```

## To-Do

- Allow the interaction model to specify different parameters for different interacting type pairs
-  Allow user to probe systems with an analyte frame (with multiple particles), rather than just a single analyte particle. This could involve re-structuring the API to something like
    - Probe class
        - contains probe particle model and probing simulation logic
    - System class
        - contains analyte particle models (can be more than one now) as well as a reference frame of positions and orientations to reset to after every "time step"
    - Interaction class
        - same as now
    - Interaction model is same as now (just a dict of strings and Interactions)
- Allow user to save interaction models to json
- double-check all indexing/array orientation logic
- refine Field dimensionality handling
- allow user to save Field to npz
- allow user to merge fields by addition or overlay
- write tests
- allow user to specify a way to skip orientations that correspond to overlaps (this would prevent the averaging problem that Josh pointed out). This could look like:
    - have high orientation resolution (e.g. 10)
    - at each position and orientation
        - check if the probe and analyte would overlap
        - if yes, remove the orientation
        - tally the number of rejected orientations - if the number of accepted ones falls below some threshold (e.g. 2), then remove that position
    - this filtering would happen **before** the parallelization/run_probe step
- make it so user doesn't have to define a separate probe particle - this would mean having to change the way we get the probe particle index, but it would make it easier to write a user-facing simulation parser
- allow user to use mc potentials - this would support writing a user-facing simulation parser