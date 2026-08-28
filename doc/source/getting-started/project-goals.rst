.. _project-goals:

=============
Project Goals
=============

p4 is designed to solve three challenges in particle systems research.


1. Learning to use Particle Simulation Software
+++++++++++++++++++++++++++++++++++++++++++++++

Particle simulation software is complicated, and their programming interfaces
often appear convoluted to students and new researchers. The typical onboarding
experience involves following along with step-wise tutorials covering simple
systems, and then adapting the tutorial code to more complex systems. Beginners
often struggle with this process because they cannot immediately see how their
changes to the code translate to changes in their computational models. With
HOOMD-blue in particular, new users often struggle to understand the deferred
validation of simulation components (particle positions, interactions, etc.),
which cause errors to happen many lines after the actual problem.

p4 provides a more beginner-friendly API for HOOMD-blue's Molecular Dynamics
(MD) functionality as well as a simple plotting interface so that a user can
immediately see the state of their simulation, the geometry of their rigid
bodies, and the strength and lengthscales of their interaction potential curves.
All p4 objects are continuously validated so that errors are never deferred.


2. Sharing Particle and Interaction Models
++++++++++++++++++++++++++++++++++++++++++

Previously, there has been no standard format for encoding and sharing models
of non-atomistic particle systems and interactions. Instead, researchers
typically translate their models into equations and pictures for the purpose of
publication, and sometimes include their simulation source code as supplementary
files. This poses a problem for other researchers who wish to replicate their
results. Translating pictures and equations back into simulation code is
labor-intensive and may not account for details of the original simulation
procedure. Using the original source code often requires replicating (or at
least approximating) the software environment of the original researchers, and
environment details are rarely published. As a result, the process of
reproducing simulation results often takes weeks or months of work.

p4 provides a set of minimal JSON schemas for encoding and sharing particle and
interaction models. While the schemas are intended for use with HOOMD-blue, they
do not require a specific version or other software dependencies (other than p4,
of course). Moreover, the JSON file format is both human and machine-readable,
and is well-suited to web-based tools.


3. Seeing the Effective Shapes of Patchy Particles
++++++++++++++++++++++++++++++++++++++++++++++++++

Designing particle and interaction models to represent patchy particles with
specific shapes is often difficult. Small changes in the distribution of
constituent particles can significantly alter a rigid body's effective
energy and force fields. There are no publicly available tools for measuring
these fields within popular particle simulation engines, which means that in
order to determine the best combination of constituent particle positions and
interaction potentials, researchers must evaluate the accuracy of their models
solely on the collective behavior of particles in their simulations.

p4 provides a toolkit for directly measuring and interactively visualizing the
effective shapes of patchy particles modelled from rigid bodies and pairwise
potentials.


A Simple Tool for Complex Problems
++++++++++++++++++++++++++++++++++

These are the challenges that p4 is designed to address. The package is designed
first and foremost to be friendly to new programmers, but also robust enough to
meet the more advanced needs of experienced simulators. It provides a simple
but flexible interface for sharing arbitrarily complex particle and interaction
models for HOOMD-blue, and for measuring, visualizing, and publishing the energy
and force fields that those models produce.
