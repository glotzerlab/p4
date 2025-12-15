# Pairwise Potential Particle Probe

Probe the effective potential energy around a central particle using HOOMD-blue.

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

- Allow the particle model to provide secondary orientations just like it does now for positions. These orientations would get used in `util.add_rigid_constraint()`.
- Allow the interaction model to specify different parameters for different interacting type pairs
-  Allow user to probe systems with an analyte frame (with multiple particles), rather than just a single analyte particle. This could involve re-structuring the API to something like
    - Probe class
        - contains probe particle model and probing simulation logic
    - System class
        - contains analyte particle models (can be more than one now) as well as a reference frame of positions and orientations to reset to after every "time step"
    - Interaction class
        - same as now
    - Interaction model is same as now (just a dict of strings and Interactions)