# Getting started

## Overview
What p4 is, basic use, note tight integration with hoomd and use of plotly

## Installation

## Basic usage - starting from an existing simulation

## Project philosophy
- Creator's note - this is the tool I wish I had
- ease of use
  - Dependencies should be minimized
  - declarative is preferrable to imperative
    - easier to learn - less syntax to track, fewer steps to remember
    - lets you serialize and deserialize
  - evaluate later, but validate *now*
  - write for users, but keep maintainers as a close second
    - no private classes, minimal private methods
    - simple module structure
    - robust, future-hardened testing framework

# User Guide

## 1. How p4 works
Brief overview (with diagrams) explaning the different meanings for the classes and their uses in the core targeted workflow

## 2. Specifying particle models
- intro to particle types
- brief discussion of hoomd typeparams - deferred evaluation means deferred validation
- Basic use - single particle, multiparticle, multiparticle with orientation
- plotting

## 3. Specifying interaction models
- lazy evaluation, instant validation
- basic use
- plotting?

## 4. Measuring the fields of particles and interaction models

## 5. Advanced Topics
### 5.1. Evaluating many different model parameters
Basically parameter sweeps with Signac

- storing p4 data in signac statepoints
- combining plotly charts together with subfigures

### 5.2. Evaluating large and complex systems
Basically large body (protein?) and frame analytes

- choosing CSV vs other data storage format
- visualizing large datasets with VTK


# Tutorials

## 1. Basic plotting
### 1.1. Plotting simulation state
### 1.2. Plotting interaction potential curves
### 1.3. Exporting figures for publication
Demonstrate export to HTML for interactive visualizations.

## 2. Evaluate energy, force, and torque fields of particle interaction models
### 2.1. Lennard-Jones Sphere
Show field dependence on epsilon and sigma

### 2.2. Anisotropic Lennard-Jones cube
Show shape curvature dependence on the parameters, and show how unphysical torque spikes arise

### 2.3. Whether to use a single particle model with an anisotropic potential vs a rigid body model with an isotropic potential

### 2.4. Evaluate previously-used models: a case study of Lin et al 2017

## 3. Optimizing interaction parameters to achieve an effective shape

## 4. Collecting training data for a ML potential

