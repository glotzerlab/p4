=======
Schemas
=======


Common Types
++++++++++++

+-----------------------+-------------------------------------------------------------+--------+
| Name                  | Type                                                        | Shape  |
+=======================+=============================================================+========+
| :stype:`Positions`    | :py:class:`list`\ [:py:class:`list`\ [:py:class:`float`\ ]] | (P, 3) |
+-----------------------+-------------------------------------------------------------+--------+
| :stype:`Orientations` | :py:class:`list`\ [:py:class:`list`\ [:py:class:`float`\ ]] | (P, 4) |
+-----------------------+-------------------------------------------------------------+--------+
| :stype:`MoI`          | :py:class:`list`\ [:py:class:`float`\ ]                     | (3,)   |
+-----------------------+-------------------------------------------------------------+--------+

.. stype:: Positions

    :Type: :py:class:`list`\ [:py:class:`list`\ [:py:class:`float`\ ]]
    :Shape: (P, 3)

    Positions in 3D space.

.. stype:: Orientations

    :Type: :py:class:`list`\ [:py:class:`list`\ [:py:class:`float`\ ]]
    :Shape: (P, 4)

    Orientations in 3D space expressed as quaternions that follow the
    ``[w,x,y,z]`` convention.

.. stype:: MoI

    :Type: :py:class:`list`\ [:py:class:`float`\ ]
    :Shape: (3,)

    Moment of Inertia (MoI) in 3D space expressed as a vector of the diagonal
    terms of the full MoI tensor.


----


.. _body-schema:

Body Schema
+++++++++++

+--------------------------------+----------------------------------------------------------------+----------------------+----------------------------+
| Name                           | Type                                                           | Shape                | Default                    |
+================================+================================================================+======================+============================+
| :bfield:`primary_type`         | :py:class:`str`                                                |                      |                            |
+--------------------------------+----------------------------------------------------------------+----------------------+----------------------------+
| :bfield:`mass`                 | :py:class:`float`                                              |                      | ``1``                      |
+--------------------------------+----------------------------------------------------------------+----------------------+----------------------------+
| :bfield:`moi`                  | :stype:`MoI`                                                   | (3,)                 | ``[1,1,1]``                |
+--------------------------------+----------------------------------------------------------------+----------------------+----------------------------+
| :bfield:`secondary_types`      | :py:class:`list`\ [:py:class:`str`\ ]                          | (T,)                 |                            |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------------+
|                                |                                                                |**Keys** | **Values** |                            |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------------+
| :bfield:`positions_by_type`    | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]    |  (T,)   |   (P, 3)   |                            |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------------+
| :bfield:`orientations_by_type` | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Orientations`\ ] |  (≤T,)  |   (P, 4)   | ``[[1,0,0,0], ...]``       |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------------+


.. bfield:: primary_type

    :Type: :py:class:`str`
    :Required: Yes

    The name of the type of the primary (i.e., *central*) particle.

.. bfield:: mass

    :Type: :py:class:`float`
    :Required: No

    The mass of the primary type. If not provided, defaults to ``1``.

.. bfield:: moi

    :Type: :stype:`MoI`
    :Shape: (3,)
    :Required: No

    The moment of inertia of the primary type. If not provided, defaults to
    ``[1,1,1]``.

.. bfield:: secondary_types

    :Type: :py:class:`list`\ [:py:class:`str`\ ]
    :Required: No
    :Shape: (T,)

    The names of the types of the secondary (i.e., *constituent*) particles.

.. bfield:: positions_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]
    :Shape: keys: (T,); values: (P, 3)
    :Required: Only if :bfield:`secondary_types` is not empty.
    :Restrictions: Keys must equal :bfield:`secondary_types`.

    A mapping from secondary types to one or more positions. Every secondary
    particle must have at least one position.

.. bfield:: orientations_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Orientations`\ ]
    :Shape: keys: (≤T,); values: (P, 4)
    :Required: No
    :Restrictions: Keys are limited to those in :bfield:`positions_by_type`. If
        provided for some type ``t``, the number of oientations must equal the
        number of positions in ``positions_by_type[t]``.

    A mapping from secondary types to orientations in quaternion form. If not
    provided for some secondary type ``t``, defaults to an array of
    ``[1,0,0,0]`` quaternions with the same length as ``positions_by_type[t]``.


----


.. _arrangement-schema:

Arrangement Schema
++++++++++++++++++

+--------------------------------+----------------------------------------------------------------+-------------------+----------------------+
| Name                           | Type                                                           | Shape             | Default              |
+================================+================================================================+===================+======================+
| :afield:`bodies`               | :py:class:`list`\ [:py:class:`~p4.body.Body`\ ]                | (B,)              |                      |
+--------------------------------+----------------------------------------------------------------+--------+----------+----------------------+
|                                |                                                                |**Keys**|**Values**|                      |
+--------------------------------+----------------------------------------------------------------+--------+----------+----------------------+
| :afield:`positions_by_type`    | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]    |  (B,)  |  (P, 3)  |                      |
+--------------------------------+----------------------------------------------------------------+--------+----------+----------------------+
| :afield:`orientations_by_type` | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Orientations`\ ] |  (≤B,) |  (P, 4)  | ``[[1,0,0,0], ...]`` |
+--------------------------------+----------------------------------------------------------------+--------+----------+----------------------+


.. afield:: bodies

    :Type: :py:class:`list`\ [:py:class:`~p4.body.Body`\ ]
    :Shape: (B,)
    :Required: Yes
    :Restrictions: Each body must have a unique :bfield:`primary_type`.

    The bodies which are placed and rotated as separate instances within the
    arrangement.

.. afield:: positions_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]
    :Shape: keys: (B,); values: (P, 3)
    :Required: Yes
    :Restrictions: Keys must equal ``[b.primary_type for b in bodies]``.

    The mapping from body primary types to one or more positions. Every body
    must have at least one position.

.. afield:: orientations_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]
    :Shape: keys: (≤B,); values: (P, 4)
    :Required: No
    :Restrictions: Keys are limited to those in :afield:`positions_by_type`. If
        provided for some ``t``, the number of orientations equal the number of
        positions in ``positions_by_type[t]``.

    The mapping from body primary types to orientations in quaternion form. If
    not provided for some primary type ``t``, defaults to an array of
    ``[1,0,0,0]`` quaternions with the same length as ``positions_by_type[t]``.


----


.. _interaction-schema:

Interaction Schema
++++++++++++++++++

+--------------------------+--------------------------------------------+
| Name                     | Type                                       |
+==========================+============================================+
| :ifield:`hoomd_class`    | Subclass of :py:class:`hoomd.md.pair.Pair` |
+--------------------------+-----+--------------------------------------+
| :ifield:`initial_args`   | :py:class:`dict`                           |
+--------------------------+-----+--------------------------------------+
| :ifield:`default_params` | :py:class:`dict`                           |
+--------------------------+-----+--------------------------------------+
| :ifield:`typed_params`   | :py:class:`dict`                           |
+--------------------------+-----+--------------------------------------+

.. ifield:: hoomd_class

    :Type: Subclass of :py:class:`hoomd.md.pair.Pair`
    :Required: Yes
    :Restrictions: Cannot be one of the following classes:
        :py:class:`~hoomd.md.pair.aniso.AnisotropicPair`,
        :py:class:`~hoomd.md.pair.aniso.Patchy`,
        :py:class:`~hoomd.md.pair.friction.FrictionalPair`.

    A HOOMD-blue MD pairwise potential type.

.. ifield:: initial_args

    :Type: dict
    :Required: Yes
    :Restrictions: Valid keys and values depend on :ifield:`hoomd_class`.

    The argument names and values (excluding ``nlist``) required to instantiate
    the HOOMD class. The contents of this field are passed directly to the class
    constructor as 
    
    .. code-block::
        
        hoomd_class(nlist=nlist, **initial_args)
    
    Consult the HOOMD-blue documentation for your class to determine the
    required argument names and values. For example, if ``hoomd_class`` is
    :py:class:`~hoomd.md.pair.pair.DPD`, a valid ``initial_args`` is
    
    .. code-block::
        
        dict(kT=1, default_r_cut=6)

.. ifield:: default_params

    :Type: dict
    :Required: Yes
    :Restrictions: Valid keys and values depend on :ifield:`hoomd_class`.

    The parameter names and values required to set default parameters in the
    instantiated HOOMD class. The contents of this field are used to directly
    set the instance's typeparam defaults with

    .. code-block::

        for param_name, param_value in default_params.items():
            getattr(instance, param_name).default = param_value
    
    Consult the HOOMD-blue documentation for your class to determine the
    required parameter names and values. For example, if ``hoomd_class`` is
    :py:class:`~hoomd.md.pair.pair.LJ`, a valid ``default_params`` is
    
    .. code-block::
        
        dict(r_cut=0, params=dict(epsilon=1, sigma=1))

.. ifield:: typed_params

    :Type: dict
    :Required: Yes, although it may be empty
    :Restrictions: Valid values depend on :ifield:`hoomd_class`.

    The names of particle types and pairs of types and the parameter
    dictionaries to apply to them in the HOOMD class after it has been
    instantiated and its defaults have been set. The contents of this field are
    used to directly set the instance's typeparams for specific particle types
    and pairs with

    .. code-block::

        for type_name, params in self.typed_params.items():
            for param_name, param_value in params.items():
                getattr(instance, param_name)[type_name] = param_value
    
    This field may be empty. In that case, the instance's default parameters
    are used for every particle type and pair of types.

    Consult the HOOMD-blue documentation for your class to determine the
    allowed parameter names and values, *and to determine which parameters
    are used for single types vs pairs of types*. For example, if
    ``hoomd_class`` is :py:class:`~hoomd.md.pair.aniso.ALJ`, a valid
    ``typed_params`` for a system with a single particle type "A" is
    
    .. code-block::
        
        {
            "A": dict(
                shape=dict(vertices=[], faces=[])
            ),
            ("A", "A"): dict(
                r_cut=6,
                params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=0)
            )
        }


----


.. _system-schema:

System Schema
+++++++++++++

+------------------------+----------------------------------------------------------------------+-------+
| Name                   | Type                                                                 | Shape |
+========================+======================================================================+=======+
| :sfield:`probe`        | :py:class:`~p4.body.Body`                                            |       |
+------------------------+----------------------------------------------------------------------+-------+
| :sfield:`analyte`      | :py:class:`~p4.body.Body` or :py:class:`~p4.arrangement.Arrangement` |       |
+------------------------+----------------------------------------------------------------------+-------+
| :sfield:`interactions` | :py:class:`list`\ [:py:class:`~p4.interaction.Interaction`\ ]        | (I,)  |
+------------------------+----------------------------------------------------------------------+-------+

.. sfield:: probe

    :Type: :py:class:`~p4.body.Body`
    :Required: Yes
    :Restrictions: The probe's :bfield:`primary_type` must not overlap with the
        analyte's types unless the probe and analyte are identical bodies.

    The probe, which is translated and rotated around the system's analyte
    during the field measurement process. Recorded F and T values correspond to
    F and T exerted on the probe by the analyte. 

.. sfield:: analyte

    :Type: :py:class:`~p4.body.Body` or :py:class:`~p4.arrangement.Arrangement`
    :Required: Yes
    :Restrictions: The analyte's :bfield:`primary_type` ( or multiple types, if
        the analyte is an :py:class:`~p4.arrangement.Arrangement`) must not
        overlap with the probe's types unless the probe and analyte are
        identical bodies.

    The analyte, which remains static at the simulation box's center during the
    field measurement process.

.. sfield:: interactions

    :Type: :py:class:`list`\ [:py:class:`~p4.interaction.Interaction`\ ]
    :Shape: (I,)
    :Required: Yes

    The interactions that collectively produce U, F, and T fields when the probe
    and analyte are within ``r_cut``.
