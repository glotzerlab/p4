==============
For Developers
==============

Design Principles
=================

The core motivation behind the creation of **p4** is that *it should be easy
for less-experienced computational scientists to systematically measure and
interactively view the potential energy fields around arbitrarily complex
bodies of particles*. This motivation is broken down into the following principles:
accessibility, flexibility, interactivity, and maintainability.

Accessibility
-------------

[TODO]


Flexibility
-----------

[TODO]


Interactivity
-------------

[TODO]


Maintainability
---------------

The code-base of **p4** is mostly be maintained by the `Glotzer Group <https://glotzerlab.engin.umich.edu/>`_,
a team of undergraduate, graduate, and post-doctoral researchers at the University of Michigan.
Turnover rate in academic research groups is much higher than in private companies, and even in
computational research groups, "new hires" typically have little or no experience in software
development, much less maintenance. Correspondingly, just as **p4** is designed for ease-of-use, it is
also designed for ease-of-maintenance...

[TODO]


Making Contributions
====================

Developers are welcome to contribute new features or bug fixes to **p4** by
making pull requests on the package repository on `Github`_. Contributions should
follow the *spirit* of the design principles described above, and the *letter* of
the style, docs, and testing rules described below.

.. _github: https://github.com/glotzerlab/p4/pulls


Code Style
----------

[TODO]


Documentation
-------------

Additions and changes to classes and functions---`even underscored ones <https://peps.python.org/pep-0008/#descriptive-naming-styles>`_---must
be fully described in docstrings that follow the `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The API
documentation is automatically generated from the docstrings with `Sphinx <https://www.sphinx-doc.org/en/master/>`_, which
requires that they be written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

To ensure that your docstrings are correctly rendered by Sphinx, please build the documentation locally before opening
a pull request.

To build the documentation, first install the dependencies:

.. code:: bash

   pip install <path_to_p4>[docs]

Then navigate to the ``doc`` directory and run

.. code:: bash

   make html

Once the documentation is built, the homepage can be found at ``<path_to_p4>/doc/build/html/index.html``.
Open this file in your browser and navigate to the relevant page in the API section to check that your docstrings are correct.

If your changes require the addition of new pages to the API section, create new ``.rst`` files whose
names and contents are parallel to existing files.

Unit tests
----------

New and changed functionality must be fully covered by unit tests, which must
be written in clearly named and documented files in the ``test`` directory. To
ensure that changes and additions do not break existing functionality, please
run all unit tests locally before opening a pull request.

To run unit tests, first install ``pytest`` and then run the command

.. code:: bash

   pytest <path_to_p4>/test
