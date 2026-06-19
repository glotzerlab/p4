.. _accelerating-workflows:

======================
Accelerating Workflows
======================

In the previous sections, we learned how to use p4 to specify particle and
interaction models, and how to measure the fields that these models create. This
is one of the primary goals of p4.

But p4 has another goal: making HOOMD-blue easier to learn and faster to use.
In :ref:`quickstart`, we saw that users can create p4 objects directly from
HOOMD-blue objects. The inverse is also possible---creating HOOMD-blue objects
from p4 objects---and this functionality allows p4 to provide a more concise,
beginner-friendly interface for portions of HOOMD-blue's API. This, along with
p4's built-in JSON support, enable p4 to accelerate many HOOMD-blue workflows.


.. _creating-hoomd-blue-objects-from-p4:

Creating HOOMD-blue objects from p4
+++++++++++++++++++++++++++++++++++

To compare p4's interface with HOOMD-blue's, let's create an example MD
simulation with each package. We will create an MD simulation for the following
moderately complex system.

* There are two kinds of rigid bodies:

  1. A cube-shaped body
 
     * Central particle: "A"
     * Constituent particles: "B" on cube vertices, "C" on face centroids
     * Mass of central particle: 2

  2. An octahedron-shaped body

     * Central particle: "D"
     * Constituent particles: "E" on octahedron vertices, "F" on face centroids
     * Mass of central particle: 4

* Cube vertices ("B") are attracted to octahedron faces ("F")
* Cube faces ("C") are attracted to octahedron vertices ("E")
* Cube core and octahedron core repel each other with an anisotropic potential

* The initial simulation state has the cube and octahedron bodies arranged in a
  10 x 10 x 1 grid with alternating bodies in each column.

Let's say both p4 and HOOMD-blue start with the following code.

.. code-block:: python

    import coxeter

    cube = coxeter.families.PlatonicFamily.get_shape("Cube")
    octa = coxeter.families.PlatonicFamily.get_shape("Octahedron")

There are three main tasks for creating our simulation:

* creating the initial state,
* creating the rigid constraint, and
* creating the pair potentials.

HOOMD-blue users typically use a rigid constraint to add constituent particles
to the simulation state, so we start by creating a rigid constraint.


Creating the rigid constraint
-----------------------------

.. grid:: 2

    .. grid-item::

        .. rubric:: p4
            :class: grid-header

        .. code-block:: python

            import p4

            # Create both bodies
            cube_body = p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(
                    B=cube.vertices,
                    C=cube.face_centroids
                ),
                mass=2
            )
            octa_body = p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(
                    E=octa.vertices,
                    F=octa.face_centroids
                ),
                mass=4
            )

            # Convert to rigid constraint
            rigid = cube_body.to_hoomd_rigid()
            rigid = octa_body.to_hoomd_rigid(rigid)

    .. grid-item::

        .. rubric:: HOOMD-blue
            :class: grid-header

        .. code-block:: python

            import hoomd
            import numpy as np

            # Create empty rigid constraint
            rigid = hoomd.md.constrain.Rigid()

            # Create iterator
            iterator = zip(
                [cube, octa],   # shapes
                ["A", "D"],     # central types
                ["B", "E"],     # vertex types
                ["C", "F"],     # face types
            )

            # Add the body definitions
            for s, ct, vt, ft in iterator:
                rigid.body[ct] = {
                    "constituent_types": (
                        [vt for _ in s.vertices]
                        + [ft for _ in s.face_centroids]
                    ),
                    "positions": np.vstack((
                        s.vertices,
                        s.face_centroids
                    )),
                    "orientations": np.array(
                        [[1,0,0,0] for _ in s.vertices]
                        + [[1,0,0,0] for _ in s.faces]
                    )
                }

With some thought, it is possible to make the HOOMD-blue code as concise as p4,
as long as the user can figure out an appropriate iterator. While this is
straightforward for our example system, it may be much harder for other systems.

p4 does not require a complex iterator to be concise, and also offers the user
the ability to plot the bodies to make sure they are defined correctly.


Creating the initial state
--------------------------

.. grid:: 2

    .. grid-item::

        .. rubric:: p4
            :class: grid-header

        .. code-block:: python

            # Calculate initial positions
            grid_positions = (
                p4.positions_on_regular_grid(
                    box=[20, 20, 2],
                    resolution=[10, 10, 1]
                )
            )

            # Create an arrangement from bodies
            arrangement = p4.Arrangement(
                bodies=[cube_body, octa_body],
                positions_by_type=dict(
                    A=grid_positions[0::2],
                    D=grid_positions[1::2]
                )
            )

            # Convert arrangement to snapshot
            snapshot = arrangement.to_hoomd_snapshot()

    .. grid-item::

        .. rubric:: HOOMD-blue
            :class: grid-header

        .. code-block:: python

            import gsd.hoomd
            from itertools import product

            # Create empty frame
            frame = gsd.hoomd.Frame()

            # Calculate positions
            xs = np.linspace(-10, 10, 10, endpoint=True)
            ys = np.linspace(-10, 10, 10, endpoint=True)
            positions = np.array(
                [[x, y, 0] for x, y in product(xs, ys)]
            )

            # Calculate mass and typeid for each position
            types = ["A", "B", "C", "D", "E", "F"]
            typeids = []
            masses = []
            for i, _ in enumerate(positions):
                if i % 2 == 0:  # even positions are A
                    typeids.append(types.index("A"))
                    masses.append(2)
                
                else:   # odd positions are D
                    typeids.append(types.index("D"))
                    masses.append(4)

            # Calculate orientations
            orientations = np.array(
                [[1,0,0,0] for _ in positions]
            )

            # Construct frame and convert to snapshot
            frame.particles.N = positions.shape[0]
            frame.particles.types = types
            frame.particles.typeid = typeids
            frame.particles.position = positions
            frame.particles.orientation = orientations
            frame.particles.mass = masses
            frame.configuration.box = [25, 25, 25, 0, 0, 0]

            snapshot = hoomd.Snapshot.from_gsd_frame(
                gsd_snap=frame,
                communicator=hoomd.communicator.Communicator()
            )

            # Use rigid to add secondary particles
            sim = hoomd.Simulation(
                device=hoomd.device.CPU()
            )
            sim.create_state_from_snapshot(snapshot)
            rigid.create_bodies(sim.state)

            # Get simulation state as a snapshot
            final_snapshot = sim.state.get_snapshot()

For this task, p4 is far more concise and understandable than HOOMD-blue. The
composition-oriented API allows p4 to re-use the bodies created in the previous
step to handle most of the logic surrounding particle data. In contrast,
even with ``rigid.create_bodies()``, HOOMD-blue requires the user to calculate
all of the particle data category by category.

Many HOOMD-blue users mitigate this annoyance by abstracting the various
subtasks into separate functions that they can chain together, but this approach
does not reduce the actual amount of code and can make it difficult for
collaborators to follow their code.


Creating the Pair Potentials
----------------------------

.. grid:: 2

    .. grid-item::

        .. rubric:: p4
            :class: grid-header

        .. code-block:: python

            # Create interactions
            attraction = p4.Interaction(
                hoomd_class=hoomd.md.pair.Gaussian,
                initial_args={},
                default_params=dict(
                    r_cut=0,
                    params=dict(epsilon=0, sigma=1)
                ),
                typed_params={
                    ("B", "F"): dict(
                        r_cut=4,
                        params=dict(
                            epsilon=-1, sigma=0.25
                        )
                    ),
                    ("C", "E"): dict(
                        r_cut=4,
                        params=dict(
                            epsilon=-1, sigma=0.25
                        )
                    ),
                }
            )

            repulsion = p4.Interaction(
                hoomd_class=hoomd.md.pair.aniso.ALJ,
                initial_args={},
                default_params=dict(
                    r_cut=0,
                    params=dict(
                        sigma_i=0.2, epsilon=0,
                        sigma_j=0.2, alpha=0
                    ),
                    shape=dict(vertices=[], faces=[])
                ),
                typed_params={
                    ("A", "D"): dict(
                        r_cut=4,
                        params=dict(
                            sigma_i=0.2, epsilon=1,
                            sigma_j=0.2, alpha=0
                        )
                    ),
                    "A": dict(
                        shape=dict(
                            vertices=cube.vertices,
                            faces=cube.faces
                        )
                    ),
                    "D": dict(
                        shape=dict(
                            vertices=octa.vertices,
                            faces=octa.faces
                        )
                    ),
                }
            )

            # Convert interactions to hoomd pairs
            nlist = hoomd.md.nlist.Tree(2)
            gauss = attraction.to_hoomd_pair(nlist)
            alj = repulsion.to_hoomd_pair(nlist)

    .. grid-item::

        .. rubric:: HOOMD-blue
            :class: grid-header

        .. code-block:: python

            nlist=hoomd.md.nlist.Tree(2)

            # Create gaussian atttraction
            gauss = hoomd.md.pair.Gaussian(nlist)

            # Set gaussian defaults
            gauss.r_cut.default = 0
            gauss.params.default = dict(
                epsilon=0, sigma=1
            )

            # Set gaussian typed parameters
            gauss.r_cut[("B", "F")] = 4
            gauss.r_cut[("C", "E")] = 4
            gauss.params[("B", "F")] = dict(
                epsilon=-1, sigma=0.25
            )
            gauss.params[("C", "E")] = dict(
                epsilon=-1, sigma=0.25
            )

            # Create alj repulsion
            alj = hoomd.md.pair.aniso.ALJ(nlist)

            # Set alj defaults
            alj.r_cut.default = 0
            alj.params.default = dict(
                sigma_i=0.2, epsilon=0,
                sigma_j=0.2, alpha=0
            )
            alj.shape.default = dict(
                vertices=[], faces=[]
            )

            # Set alj typed parameters
            alj.r_cut[("C", "E")] = 4
            alj.params[("A", "D")] = dict(
                sigma_i=0.2, epsilon=1,
                sigma_j=0.2, alpha=0
            )
            alj.shape["A"] = dict(
                vertices=cube.vertices,
                faces=cube.faces
            )
            alj.shape["D"] = dict(
                vertices=octa.vertices,
                faces=octa.faces
            )

p4 is not quite as concise as HOOMD-blue for this task, but offers two
advantages. First, pair potentials are validated on instantiation, rather
than on ``simulation.run()``, ensuring that users do not have to backtrack
through their code to fix improperly parameterized potentials. Second, p4 offers
an interaction plotting method to help users quickly determine appropriate
parameter values.


Leveraging p4's JSON support with Signac
++++++++++++++++++++++++++++++++++++++++

Many HOOMD-blue users also use `Signac`_, a Python-based data management
framework, to structure and organize projects involving many simulations across
combinations of parameters. The atomic unit of organization in Signac is the
"job", which is essentially a folder, and each job contains a JSON file (called
the "state point") that stores unique parameters for a simulation. By iterating
over the jobs in a project, Signac users can parameterize and run many different
simulations while keeping their data separate.

.. _Signac: https://signac.readthedocs.io/en/latest/

p4 allows the user to import and export every class except ``Field`` to and
from JSON files. This makes it possible to store rigid bodies, initial
simulation states, and pair potentials inside Signac state points. Consider the
following example.

Let's make a signac project for investigating a Lennard-Jones fluid with
variable initial density. There are three state point parameters: ``epsilon``,
``sigma``, and ``rho``. We first initialize the Signac project.

.. code-block::

    import signac

    project = signac.init_project()

Next, we populate the Signac project with jobs.

.. grid:: 2

    .. grid-item::

        .. rubric:: p4 approach
            :class: grid-header

        .. code-block::

            from pathlib import Path
            import p4

            iterator = zip(
                [0.1, 0.5, 1],
                [0.1, 0.5, 1],
                [0.2, 0.3, 0.5]
            )

            for e, s, r in iterator:
                statepoint = dict(
                    epsilon=e,
                    sigma=s,
                    rho=r
                )
                job = (
                    project
                        .open_job(statepoint)
                        .init()
                )

                # Create arrangement using rho
                arrangement = p4.Arrangement(
                    bodies=[p4.Body("A")],
                    positions_by_type=dict(A=[...])
                )

                # Create pair using epsilon and sigma
                interaction = p4.Interaction(
                    hoomd_class=hoomd.md.pair.LJ,
                    initial_args={},
                    default_params=dict(
                        r_cut=4,
                        epsilon=e,
                        sigma=s
                    ),
                    typed_params={}
                )

                # Write arrangement and interaction
                # into the state point
                sp_path = (
                    Path(job.path)
                    / "signac_statepoint.json"
                )
                arrangement.to_json(sp_path)
                interaction.to_json(sp_path)

    .. grid-item::

        .. rubric:: Typical approach
            :class: grid-header

        .. code-block::

            iterator = zip(
                [0.1, 0.5, 1],
                [0.1, 0.5, 1],
                [0.2, 0.3, 0.5]
            )

            for e, s, r in iterator:
                statepoint = dict(
                    epsilon=e,
                    sigma=s,
                    rho=r
                )
                job = (
                    project
                        .open_job(statepoint)
                        .init()
                )

Finally, we iterate over the jobs, calling a function to make simulations from
the parameters in the statepoint.

.. grid:: 2

    .. grid-item::

        .. rubric:: p4 approach
            :class: grid-header

        .. code-block::

            def make_sim(job):
                sp_path = (
                    Path(job.path)
                    / "signac_statepoint.json"
                )

                # Make simulation
                sim = hoomd.Simulation(
                    hoomd.device.CPU()
                )

                # Get initial state from state point
                snapshot = (
                    p4.Arrangement
                        .from_json(sp_path)
                        .to_hoomd_snapshot()
                )
                sim.create_state_from_snapshot(
                    snapshot
                )

                # Get potential from state point
                pair = (
                    p4.Interaction
                        .from_json(sp_path)
                        .to_hoomd_pair()
                )
                integrator = hoomd.md.Integrator(0.1)
                integrator.forces.append(pair)
                sim.operations.integrator = integrator

                # Add loggers, writers, etc...
                # then return the simulation

            for job in project.find_jobs():
                simulation = make_sim(job)
                simulation.run(1)

    .. grid-item::

        .. rubric:: Typical approach
            :class: grid-header

        .. code-block::

            import hoomd

            def make_snapshot(job):
                rho = job.sp.rho
                
                # Make the snapshot
                # using rho...
                # then return the snapshot

            def make_pair(job):
                epsilon = job.sp.epsilon
                sigma = job.sp.sigma

                # Make the pair using
                # epsilon and sigma...
                # then return the pair

            def make_sim(job):
                # Make simulation
                sim = hoomd.Simulation(
                    hoomd.device.CPU()
                )

                # Make and add initial state
                snapshot = make_snapshot(job)
                sim.create_state_from_snapshot(
                    snapshot
                )

                # Make and add the potential
                pair = make_pair(job)
                integrator = hoomd.md.Integrator(0.1)
                integrator.forces.append(pair)
                sim.operations.integrator = integrator

                # Add loggers, writers, etc...
                # then return the simulation

            for job in project.find_jobs():
                simulation = make_sim(job)
                simulation.run(1)

What is the point of the p4 approach? By storing pair potentials and initial
states in the job statepoint, the task of mapping a state point to HOOMD-blue
objects is moved from the job-iteration step back to the project population
step. This change helps users reduce the number of functions that must operate
on a job, making it easier to expand a project with new statepoint parameters.
