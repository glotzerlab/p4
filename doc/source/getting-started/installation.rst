.. _installation:

============
Installation
============

p4 is distributed through `conda-forge`_ and its source code is available on
`GitHub`_. You can install it from conda-forge using a package manager or from
source using `git`_ and `pip`_. **Since p4 requires HOOMD-blue, it currently
cannot be installed on Windows.**

.. _conda-forge: https://anaconda.org/channels/conda-forge/packages/p4/overview
.. _GitHub: https://github.com/glotzerlab/p4
.. _git: https://git-scm.com/
.. _pip: https://pip.pypa.io/en/stable/


Setup
+++++

Using your package manager, create an environment with ``python>=3.14`` and
``hoomd``.

.. tab-set::

   .. tab-item:: Pixi

      .. code-block::

         cd <path_to_your_project>
         pixi init
         pixi add python hoomd

   .. tab-item:: Micromamba

      .. code-block::

         micromamba create -n <your_env_name> python hoomd

   .. tab-item:: Mamba

      .. code-block::

         mamba create -n <your_env_name> python hoomd


Installation from conda-forge
+++++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: Pixi

      .. code-block::

         cd <path_to_your_project>
         pixi add p4

   .. tab-item:: Micromamba

      .. code-block::

         micromamba activate <your_env_name>
         micromamba install p4

   .. tab-item:: Mamba

      .. code-block::

         mamba activate <your_env_name>
         mamba install p4


Installation from source
++++++++++++++++++++++++

1. Clone the repository to your local computer.

   .. code-block::
      
      git clone https://github.com/glotzerlab/p4.git

2. In your terminal, navigate to p4's root directory and install its
   dependencies.

   .. code-block::

      cd <path_to_p4>
      pip install .
