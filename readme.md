# p4 - Patchy Particle Prototyper in Python

p4 is a package for creating and analyzing models of particles and their interactions, based on [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) and [Plotly](https://plotly.com/). Users can measure energy, force, and torque fields created by rigid bodies of particles interacting via pairwise potentials, and then interactively plot those bodies and fields in 3D, 2D, or 1D.

p4 has three objectives:

1. to make it easy to see the effective shape of a rigid body's interaction fields, rather than inferring it from simulation results;
2. to make it easy to share rigid body models and interaction models without requiring specific software; and
3. to make it easier to use HOOMD-blue for patchy particle systems research.

p4 accomplishes these objectives through an API that is designed to be

- *declarative*, enabling easy serialization/deserialization to/from JSON;
- *comprehensive*, accommodating arbitrary combinations of HOOMD-blue pair potentials; and
- *interactive*, providing a rich plotting interface for rigid body models and fields.

## Setup

### Install p4

p4 is distributed through [conda-forge](https://anaconda.org/channels/conda-forge/packages/p4/overview)
and its source code is available in this repository. You can install it from
conda-force using a package manager or from source using [git](https://git-scm.com/)
and [pip](https://pip.pypa.io/en/stable/). **Since p4 requires HOOMD-blue, it currently cannot be installed on Windows.**

#### Installation from conda-forge

**Pixi**

```bash
cd <path_to_your_project>
pixi init
pixi add python hoomd p4
```

**Micromamba**

```bash
micromamba create -n <your_env_name> python hoomd p4
micromamba activate <your_env_name>
```

**Mamba**

```bash
mamba create -n <your_env_name> python hoomd p4
micromamba activate <your_env_name>
```

#### Installation from source

1. Clone this repository to your local computer.

   ```bash
   git clone https://github.com/glotzerlab/p4.git
   ```

2. In your terminal, install p4 into your environment in editable mode.

   **Pixi**

   Open your `pixi.toml` file and add the following lines.

   ```
   [pypi-dependencies]
   p4 = { path = "<path_to_p4>", editable = true }
   ```

   **Micromamba**

   ```bash
   cd <path_to_p4>
   pip install -e .
   ```

   **Mamba**

   ```bash
   cd <path_to_p4>
   pip install -e .
   ```

### Run unit tests and build documentation

To run tests and build docs, you must have installed p4 from source.

1. Using your package manager, install the developer dependencies.

   **Pixi**
   
   Open your `pixi.toml` file and edit the p4 line to read as follows.

   ```
   p4 = { path = "<path_to_p4>", editable = true, extras = ["dev"] }
   ```

   **Micromamba**

   ```bash
   cd <path_to_p4>
   pip install -e .[dev]
   ```

   **Mamba**

   ```bash
   cd <path_to_p4>
   pip install -e .[dev]
   ```

2. Run the unit tests with pytest

   ```bash
   pytest <path_to_p4>/test
   ```

3. build the documentation using the provided build script

   ```bash
   cd <path_to_p4>/doc
   make html
   ```

   Once the docs finish building, the homepage can be found at `<path_to_p4>/doc/build/html/index.html`.

## Citing p4

If you use p4 in your research, we ask that you acknowledge its use in relevant publications, preprints, talks, and posters by citing this GitHub repository. A manuscript is currently in preparation by Joseph Burkhart and Sharon C. Glotzer. Once the manuscript is accepted, we will ask that you cite both the repository and the publication.
