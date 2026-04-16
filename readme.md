# p4 - Patchy Particle Prototyper in Python

Rapidly prototype new particle and interaction models and evaluate existing models using [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/).

p4 provides a simple declarative interface for specifying particle models and pairwise interaction models, and for measuring the energy and force fields that those models create. Particle models and their fields can then be easily plotted with methods built on the interactive plotting library [Plotly](https://plotly.com/).

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
pip install <path_to_p4>[docs]
```

2. Navigate to p4's `doc` directory and run

```
make html
```

Once the docs finish building, the homepage can be found at `<path_to_p4>/doc/build/html/index.html`.
