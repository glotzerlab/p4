# p4 - Pairwise Potential Particle Probe

Probe the effective potential landscape around an analyte using HOOMD-blue.


Potential paper titles:
    - Toward the rational design of pairwise particle interaction models


## Example usage
```python
# Probe the potential around a LJ sphere
probe_model = p4.BodyModel("P")
analyte_model = p4.BodyModel("A")

interaction_model = {
    "LJ": p4.Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args=dict(),
        no_single_typed_attributes=dict(),
        no_pair_typed_attributes=dict(
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

system = p4.System(probe_model, analyte_model, interaction_model)

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

`Interaction`
- [x] Allow the interaction model to specify different parameters for different interacting type pairs
  ~~- [ ] Add support for HPMC potentials (contributes to goal of writing a simulation parser)~~
    ~~- to accomplish this, I think  `InteractionModel` would need to be its own class that checks to make sure every constituent `Interaction` has the same  simulation type (MD or HPMC). Ideally, it wouldn't matter (see HOOMD-rs), but for HOOMD-blue it certainly does.~~
- [x] write tests
- [ ] add parsing from `hoomd.md.pair.Pair` subclasses

`BodyModel`
- [x] write tests
- [ ] add parsing from `hoomd.md.constraint.Rigid`, `hoomd.Snapshot`, and `hoomd.Simulation`

`System`
- [ ] remove need for a separate probe `ParticleModel`, generating a probe particle on-the-fly to investigate the provided interactions (contributes to goal of writing a simulation parser)
  - to accomplish this, need to change how index of probe particle is calculated. Does that index ever change?
  - PROBLEM: what if you aren't interested in body A - body A interactions, but rather in body A - body B? How do you investigate those with no notion of a probe particle? The notion of a System would have to change - it would be a collection of named particle models and a single interaction model (Or maybe a combination of a Particle Model and an Interaction Model - a ParticleModel being a named collection of BodyModels...)
- [ ] add method for calculating an array of positions and orientations that do not produce effective overlaps (this would prevent the averaging problem that Josh pointed out). This could look like:
  - have high orientation resolution (e.g. 10)
  - at each position and orientation
      - check if the probe and analyte would overlap
      - if yes, remove the orientation
      - tally the number of rejected orientations - if the number of accepted ones falls below some threshold (e.g. 2), then remove that position
  - this filtering would happen **before** the parallelization/run_probe step
  - alternatively, have a dynamic meshing approach like with ChIMES (maybe do both?)
- [ ] write tests
- [ ] add parsing from `hoomd.Simulation`

`Field`
- [ ] refine dimensionality handling (always 3D? if not, need checks when calling various methods)
- [ ] method for saving to npz
- [ ] methods or overloads for adding/overlaying multiple fields to create a new one
- [ ] write tests

`util`
  - [ ] functions for saving/creating interaction models to/from json


Other
  - [ ] double-check all indexing/array orientation logic
  - [ ] Allow user to probe systems with an analyte frame (with multiple particles), rather than just a single analyte particle. This could involve re-structuring the API to something like
      - Probe class
          - contains probe particle model and probing simulation logic
      - System class
          - contains analyte particle models (can be more than one now) as well as a reference frame of positions and orientations to reset to after every "time step"


## Roadmap

### Open to group members
functionality
  - [ ] parse hoomd objects to create Interaction, ParticleModel, System
  - [ ] plot 1D and 2D slices of Fields

tests
  - [ ] non-validation coverage 100%

documentation
  - [ ] API + 1 example

### Open to public
TBD

### Publication
TBD

### 'Completion'
TBD
