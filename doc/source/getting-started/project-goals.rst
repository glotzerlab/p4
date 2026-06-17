.. _project-goals:

=============
Project Goals
=============

The core motivation of p4 is that it should be easy to ground particle
interaction models in the physical systems they represent.

Problems with Pure Phenomenology
================================

Every year, dozens of studies use molecular dynamics simulations to investigate
the experimentally observed behavior of particle systems. Most often, the
particle and interaction models used in these simulations are adapted from
previous or constructed ad hoc, and then validated by evaluating whether
they appear to reproduce trends in the phase behavior observed from experiments.
Properly, these models are called `phenomenological`_, since they are designed
to reproduce the *phenomena* observed in experiment.

.. _phenomenological: https://en.wikipedia.org/wiki/Phenomenological_model

There are several problems with purely phenomenological particle models.

First, while they may reproduce observed phases, they may not be the most
*effective* models for doing so---perhaps a different potential energy curve or
a different patch distribution might assemble the target phases faster and with
less computational overhead. Moreover, if subtle changes in a model can
significantly affect the rate of assembly, it cannot provide reliable
information about the assembly kinetics in experiment.

Second, if a model's validity is only evaluated on the basis of the phases it
assembles and their relative positions in parameter space, there is no guarantee
that the model's individual building blocks resemble the those of the physical
system. This means that the model can only provide limited insight into
*why* phase transitions happen, and that the model's behavior beyond its
phenomenologically validated parameter-space may be completely unphysical.

Third, models that are purely phenomenological are hard to extend to new
systems, because they may include features that have no effect on how they
simulate their intended system but may significantly affect how they simulate
other similar systems.

The Reproducibility Crisis
==========================

Compounding these problems is the lack of a standard, portable format for
encoding and sharing particle and interaction models between researchers.
Publications render these models into pictures and equations, but any
researcher who has attempted to replicate simulated phase behavior knows
that translating these images and symbols back into simulation code is almost
always extremely challenging.

The simulation of particle phase behavior can be highly sensitive to minutiae
in the computational procedure. Initial system configuration and the
trajectories of system temperature and volume can play a significant role in
determining the phases observed to assemble in simulation, and these details are
often neglected in publications of simulation results. If a researcher is trying
to replicate a model's published results, and if that model's main published
justification is merely that it "reproduces trends in experimental phase
behavior", it can be almost impossible to determine whether the problem lies in
the researcher's re-implementation of the model, the details of the researcher's
simulation procedure, or in the originally published model. This problem is
part of the heart of computational particle science's reproducibility crisis.

The Problem of Accessibility
============================

Finally, learning to simulate complex particle systems is hard. Particle
simulation software is complicated, and their programming interfaces often
appear convoluted to students and new researchers. The typical onboarding
experience involves following along with step-wise tutorials, which often do
not teach new users how to organize their simulation procedure code to
accommodate large, multi-system projects. This makes it that much harder to
package a project's code for long-term storage, which is essential for
reproducibility.

A Simple Tool for Complex Problems
==================================

These are the problems that p4 is designed to address. This package is designed
first and foremost to be friendly to new programmers, but also robust enough to
meet the more advanced needs of experienced simulators. It provides a simple
but flexible interface for sharing arbitrarily complex particle and interaction
models for HOOMD-blue, and for measuring, visualizing, and publishing the energy
and force fields that those models produce.

