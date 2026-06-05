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

+--------------------------------+----------------------------------------------------------------+----------------------+----------------------+
| Name                           | Type                                                           | Shape                | Default              |
+================================+================================================================+======================+======================+
| :bfield:`primary_type`         | :py:class:`str`                                                |                      |                      |
+--------------------------------+----------------------------------------------------------------+----------------------+----------------------+
| :bfield:`secondary_types`      | :py:class:`list`\ [:py:class:`str`\ ]                          | (T,)                 |                      |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+
|                                |                                                                |**Keys** | **Values** |                      |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+
| :bfield:`positions_by_type`    | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]    |  (T,)   |   (P, 3)   |                      |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+
| :bfield:`orientations_by_type` | :py:class:`dict`\ [:py:class:`str`\ , :stype:`Orientations`\ ] |  (≤T,)  |   (P, 4)   | ``[[1,0,0,0], ...]`` |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+
| :bfield:`mass_by_type`         | :py:class:`dict`\ [:py:class:`str`\ , :py:class:`float`\ ]     |  (≤T,)  |            | ``1``                |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+
| :bfield:`moi_by_type`          | :py:class:`dict`\ [:py:class:`str`\ , :stype:`MoI`\ ]          |  (≤T,)  |   (3,)     | ``[1,1,1]``          |
+--------------------------------+----------------------------------------------------------------+---------+------------+----------------------+


.. bfield:: primary_type

    :Type: :py:class:`str`
    :Required: Yes

    The name of the type of the primary (i.e., *central*) particle.

.. bfield:: secondary_types

    :Type: :py:class:`list`\ [:py:class:`str`\ ]
    :Required: No
    :Shape: (T,)

    The names of the types of the secondary (i.e., *constituent*) particles.

.. bfield:: positions_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]
    :Shape: keys: (T,); values: (P, 3)
    :Required: Yes
    :Restrictions: Keys must equal :bfield:`secondary_types`.

    A mapping from secondary types to one or more positions. Every secondary
    particle must have at least one position.

.. bfield:: orientations_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Orientations`\ ]
    :Shape: keys: (≤T,); values: (P, 4)
    :Required: No
    :Restrictions: Keys are limited to those in :bfield:`positions_by_type`. If provided for some type ``t``, the number of oientations must equal the number of positions in ``positions_by_type[t]``.

    A mapping from secondary types to orientations in quaternion form. If not
    provided for some secondary type ``t``, defaults to an array of
    ``[1,0,0,0]`` quaternions with the same length as ``positions_by_type[t]``.

.. bfield:: mass_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :py:class:`float`\ ]
    :Shape: keys: (≤T,)
    :Required: No
    :Restrictions: Keys are limited to those in :bfield:`positions_by_type`.

    The mapping from secondary type names to mass. If not provided for some
    type, defaults to ``1``.

.. bfield:: moi_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`MoI`\ ]
    :Shape: keys: (≤T,); values: (P, 4)
    :Required: No
    :Restrictions: Keys are limited to those in :bfield:`positions_by_type`.

    The mapping from secondary type names to moment of inertia. If not provided
    for some type, defaults to ``[1,1,1]``.


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
    *must* have at least one position.

.. afield:: orientations_by_type

    :Type: :py:class:`dict`\ [:py:class:`str`\ , :stype:`Positions`\ ]
    :Shape: keys: (≤B,); values: (P, 4)
    :Required: No
    :Restrictions: Keys are limited to those in :afield:`positions_by_type`. If provided for some ``t``, the number of orientations equal the number of positions in ``positions_by_type[t]``.

    The mapping from body primary types to orientations in quaternion form. If
    not provided for some primary type ``t``, defaults to an array of
    ``[1,0,0,0]`` quaternions with the same length as ``positions_by_type[t]``.



----


.. _interaction-schema:

Interaction Schema
++++++++++++++++++

TODO


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
    :Restrictions: The probe's :bfield:`primary_type` must not overlap with the analyte's types unless the probe and analyte are identical bodies.

    The probe, which is translated and rotated around the system's analyte
    during the field measurement process. Recorded F and T values correspond to
    F and T exerted on the probe by the analyte. 

.. sfield:: analyte

    :Type: :py:class:`~p4.body.Body` or :py:class:`~p4.arrangement.Arrangement`
    :Required: Yes
    :Restrictions: The analyte's :bfield:`primary_type` ( or multiple types, if the analyte is an :py:class:`~p4.arrangement.Arrangement`) must not overlap with the probe's types unless the probe and analyte are identical bodies.

    The analyte, which remains static at the simulation box's center during the
    field measurement process.

.. sfield:: interactions

    :Type: :py:class:`list`\ [:py:class:`~p4.interaction.Interaction`\ ]
    :Shape: (I,)
    :Required: Yes

    The interactions that collectively produce U, F, and T fields when the probe
    and analyte are within ``r_cut``.
