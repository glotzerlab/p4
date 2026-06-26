# p4 - Patchy Particle Prototyper in Python

p4 is a package for creating and analyzing models of particles and their interactions, based on [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) and [Plotly](https://plotly.com/). Users can measure energy, force, and torque fields created by rigid bodies of particles interacting via pairwise potentials, and then interactively plot those bodies and fields in 3D, 2D, or 1D.

p4 has three objectives:

1. to make it easy to see the effective shape of a rigid body's interaction fields, rather than inferring it from simulation results;
2. to make it easy to share rigid body models and interaction models without requiring specific software; and
3. to make it easier to use HOOMD-blue for particle systems research.

p4 accomplishes these objectives through an API that is designed to be

- *declarative*, enabling easy serialization/deserialization to/from JSON;
- *comprehensive*, accommodating arbitrary combinations of HOOMD-blue pair potentials; and
- *interactive*, providing a rich plotting interface for rigid body models and fields.

## Setup

### Install p4

1. Using your environment manager, activate or create an environment with `python>=3.11` and `hoomd>=5.0.0`.
2. Clone the repository.
3. Install the other dependencies:

```
pip install <path_to_p4>
```

### Run unit tests

1. Using your environment manager, install `pytest`.
2. Run pytest:

```
pytest <path_to_p4>/test
```

### Build Documentation

1. Install documentation requirements:

```
pip install <path_to_p4>[dev]
```

2. Navigate to p4's `doc` directory and run

```
make html
```

Once the docs finish building, the homepage can be found at `<path_to_p4>/doc/build/html/index.html`.
