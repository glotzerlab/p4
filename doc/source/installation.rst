============
Installation
============

Currently, installation is only possible from source. Follow these
steps to install **p4**.

#. Using your environment manager, activate or create an environment
   with ``python>=3.11`` and ``hoomd>=5.0.0``. For example, with
   `mamba <https://mamba.readthedocs.io/en/latest/#>`_ the command would be

.. code:: bash

   mamba create -n <your_name> python>=3.11 hoomd>=5.00

#. Clone the repository to your local computer. If you are using Git from
   the terminal, the command would be

.. code:: bash

   git clone https://github.com/glotzerlab/p4.git

#. Install the python dependencies.

.. code-block:: bash

   pip install <path_to_p4>