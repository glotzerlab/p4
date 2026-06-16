# To-do

### Open to public
- [ ] revise project goals
- [ ] fix body image wrapping problem
- [ ] tests for top-level functions
- [ ] add remaining top-level plotting functions

### Publication
TBD

### 'Completion'
- [ ] add autogenerate params function for interactions
- [ ] Change from StringIO to some other data storage/retention approach
- [ ] Add support for VTK
- [ ] add shape overlap exclusion sampling
- [ ] add dynamical mesh sampling
- [ ] fix issue where passing n_processes=-1 with super low resolution causes StopIteration error in table - the fix is to restrict the number of processes allowed based on the resolution
- [ ] add plotly js to directory and have html files point to it

Possible performance improvements:
  - [ ] switch from nested for-loop to single computation over many spaced-out configurations