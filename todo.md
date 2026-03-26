## To-do

`Body`
- [ ] add a to_rigid method

`System`
- [ ] add API for sampling strategies
  - [ ] uniform grid
  - [ ] uniform grid with geometrically-defined gap
  - [ ] shape overlap exclusion
  - [ ] dynamical meshing?

`Field`
- [ ] refine dimensionality handling (always 3D? if not, need checks when calling various methods)
- [ ] method for saving to npz
- [ ] methods or overloads for adding/overlaying multiple fields to create a new one
- [ ] write tests

`util`
- [ ] functions for saving/creating interaction models to/from json


Other
  - [ ] Add support for Frame analytes. This would require
    - [ ] Writing measurements concurrently to disk, rather than storing in StringIO
    - [ ] Support VTK in Field IO (plotly can't handle datasets that large)
    - [ ] Body saving to VTK

## Roadmap

### Open to group members
functionality
  - [x] parse hoomd objects to create Interaction, ParticleModel, System
  - [x] plot 1D and 2D slices of Fields

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
