.. _installation:

============
Installation
============

Currently, installation is only possible from source. Follow these
steps to install p4.

1. Using your environment manager, activate or create an environment
   with ``python>=3.11`` and ``hoomd>=5.0.0``. For example, with
   `mamba <https://mamba.readthedocs.io/en/latest/#>`_ the command would be

.. code-block::

   mamba create -n <your_name> python>=3.11 hoomd>=5.0.0

2. Clone the repository to your local computer. If you are using Git from
   the terminal, the command would be

.. code-block::

   git clone https://github.com/glotzerlab/p4.git

3. In your terminal, navigate to p4's root directory and install the python
   dependencies.

.. code-block::

   pip install <path_to_p4>