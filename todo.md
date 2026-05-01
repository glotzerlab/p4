## To-do

`System`
- [ ] add API for sampling strategies
  - [ ] uniform grid
  - [ ] uniform grid with geometrically-defined gap
  - [ ] shape overlap exclusion
  - [ ] dynamical meshing?

`Interaction`
- [x] add plotting
- [ ] add autogenerate dict method

Other
- [ ] Add support for Frame analytes. This would require
  - [ ] Writing measurements concurrently to disk, rather than storing in StringIO
  - [ ] Support VTK in Field IO (plotly can't handle datasets that large)
  - [ ] Body saving to VTK
- [ ] Add state/frame plotting
- [x] write basic error-checking tests for plotting
- [ ] add validation tests

Docs
- [ ] switch to MyST
- [ ] move HTML to separate build step
- [x] fix syntax highlighting for bash and license

## Roadmap

### Open to group members
tests
  - [ ] non-validation coverage 100%

### Open to public
TBD

### Publication
TBD

### 'Completion'
TBD
