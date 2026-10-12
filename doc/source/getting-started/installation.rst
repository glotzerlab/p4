.. _installation:

============
Installation
============

p4 is distributed through `conda-forge`_ and its source code is available on
`GitHub`_. You can install it from conda-forge using a package manager or from
source using `git`_. **Since p4 requires HOOMD-blue, it currently
cannot be installed on Windows.**

.. _conda-forge: https://anaconda.org/channels/conda-forge/packages/p4/overview
.. _GitHub: https://github.com/glotzerlab/p4
.. _git: https://git-scm.com/


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


.. _installation-from-source:

Installation from source
++++++++++++++++++++++++

1. Clone the repository to your local computer.

   .. code-block::
      
      git clone https://github.com/glotzerlab/p4.git

2. In your terminal, install p4 into your environment in editable mode.

   .. tab-set::

      .. tab-item:: Pixi

         .. code-block::

            cd <path_to_your_project>
         
         Open your ``pixi.toml`` file and add the following lines.

         .. code-block::

            [pypi-dependencies]
            p4 = { path = "<path_to_p4>", editable = true }

      .. tab-item:: Micromamba

         .. code-block::

            micromamba activate <your_env_name>
            cd <path_to_p4>
            pip install -e .

      .. tab-item:: Mamba

         .. code-block::

            mamba activate <your_env_name>
            cd <path_to_p4>
            pip install -e .


.. _installing-dev-dependencies:

Installing developer dependencies
+++++++++++++++++++++++++++++++++

To contribute to p4, you will need to install additional packages for running
unit tests and building the documentation.

.. tab-set::

   .. tab-item:: Pixi

      Developer dependencies are already installed.

   .. tab-item:: Micromamba

      .. code-block::

         cd <path_to_p4>
         pip install -e .[dev]

   .. tab-item:: Mamba

      .. code-block::

         cd <path_to_p4>
         pip install -e .[dev]
