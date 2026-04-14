.. _for-developers:

==============
For Developers
==============

Design Principles
=================

p4 was created by and is mostly maintained by the `Glotzer Group`_, a team of
undergraduate, graduate, and post-doctoral researchers at the University of
Michigan. Turnover rate in academic research groups is much higher than in
private companies, and even in computational research groups, "new hires"
typically have little or no experience in software development, much less
software maintenance. Consequently, the core design goal for p4 is that it
should be easy to use and easy to maintain. To meet this goal, the design of
p4 follows three fundamental principles: simplicity, flexibility, and
interactivity.

.. _Glotzer Group: https://glotzerlab.engin.umich.edu/


Simplicity
----------

p4 is designed to be structurally and semantically simple.

Structurally, p4 has a minimal number of classes that have separate, clearly
defined roles within the package's targeted workflows. There is no class
hierarchy or inheritance, and no auxiliary classes. Private/underscored methods
are kept to a minimum, and there are no private classes.

Semantically, p4 contains numerous named intermediate variables to aid readers
in understanding its various implementations. List comprehensions and for loops
routinely use multi-word variable names to ensure that the purposes of those
variables are clear. Non-docstring comments supplement this objective where
necessary.

Going forward, future maintainers should follow this principle by prioritizing
a small, atomic type system with implementations that focus on clarity over
concision.


Flexibility
-----------

p4 is designed to be flexible, capable of accommodating arbitrarily complex
models expressed in HOOMD-blue's stable but continuously maintained rigid body
and pairwise potential systems. It achieves this flexibility by wrapping those
systems in a thin declarative API that is exhaustively tested, ensuring that it
can accommodate any model that HOOMD-blue can handle.

Going forward, future maintainers should follow this principle by adhering
closely to HOOMD-blue's internal systems, only wrapping its existing
functionality.


Interactivity
-------------

p4 is designed to be interactive, accelerating the iterative
hypothesize-test-evaluate development loop through a robust plotting interface
and immediate and detailed error handling. All classes can be visualized in
interactive plots, and instance validation always happens on instantiation,
providing immediate responsiveness to users' code.

Going forward, future maintainers should follow this principle by ensuring that
users can visualize new classes wherever possible, and by including
comprehensive validation and error detection as close to instantiation as they
can.


Making Contributions
====================

Other developers are welcome to contribute new features or bug fixes to p4 by
making pull requests on the package repository on `Github`_. Contributions
should follow the *spirit* of the design principles described above, and the
*letter* of the style, docs, and testing rules described below.

.. _github: https://github.com/glotzerlab/p4/pulls


Code Style
----------

All code contributed to p4 must follow these rules.

1. Lines should not exceed 80 characters in length.

2. Constants, variables, functions, and modules should be named in
   `snake case`_ (``this_is_a_method_name``), while classes should be named in
   `camel case`_ with the first letter always capitalized
   (``ThisIsAClassName``).

3. Every method, class, and function, no matter how small, must have a
   docstring. For very simple functions or methods, such as property getters,
   this docstring may be a single line summary. All other docstrings must be
   follow specific formatting requirements---see `Documentation`_.

4. All imports should be placed at the top of the module file.

.. _snake case: https://en.wikipedia.org/wiki/Snake_case
.. _camel case: https://en.wikipedia.org/wiki/Camel_case

Documentation
-------------

Additions and changes to classes and
functions---\ `even underscored ones`_\ ---must be fully described in
docstrings that follow the `NumPy style`_. The API documentation is
automatically generated from the docstrings with `Sphinx`_, and so docstrings
must be written in `reStructuredText`_.

.. _even underscored ones: https://peps.python.org/pep-0008/#descriptive-naming-styles
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

To ensure that your docstrings are correctly rendered by Sphinx, please build
the documentation locally before opening a pull request.

To build the documentation, first install the dependencies.

.. code:: bash

   pip install <path_to_p4>[docs]

Then navigate to the ``doc`` directory and run

.. code:: bash

   make html

Once the documentation is built, the homepage can be found at
``<path_to_p4>/doc/build/html/index.html``. Open this file in your browser and
navigate to the relevant page in the API section to check that your docstrings
are correct.

If your changes require the addition of new pages to the API section, create new
``.rst`` files whose names and contents are parallel to existing files.


Unit tests
----------

New and changed functionality must be fully covered by unit tests, which must
be written in clearly named and documented files in the ``test`` directory. To
ensure that changes and additions do not break existing functionality, please
run all unit tests locally before opening a pull request.

To run unit tests, first install ``pytest`` and then run the command

.. code:: bash

   pytest <path_to_p4>/test
