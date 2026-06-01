# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from __future__ import annotations
from collections import defaultdict
from inspect import signature
import json
import os
import coxeter
import gsd
import hoomd
import numpy as np
import plotly
from copy import copy
from pathlib import Path

import rowan
from . import util
from .type_aliases import positions_like, orientations_like, moi_like

# TODO: check default moi


class Body:
    """The names and spatial data for a body's primary and secondary types.
   
    When secondary types **are not** provided, the body represents a
    simple particle with a single type.
    
    When secondary types **are** provided, the body represents a rigid body
    with a central particle (``primary_type``) and one or more constituent
    particles (``secondary_types``). In this case, the positions for each
    secondary type must also be provided.

    Optionally, the user may provide orientations (for each secondary type),
    and masses and moments of inertia (for the primary type and each
    secondary type). If not provided, orientation defaults to ``(1, 0, 0, 0)``,
    mass defaults to ``1``, and moment of inertia defaults to ``[1, 1, 1]``.

    .. code-block:: python
        :caption: A cubic body with primary particle 'A' at the center and secondary particles 'B' at the vertices.

        import p4

        body = p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type=dict(
                B=[
                    [-1, -1, -1],
                    [-1, -1,  1],
                    [-1,  1, -1],
                    [-1,  1,  1],
                    [ 1, -1, -1],
                    [ 1, -1,  1],
                    [ 1,  1, -1],
                    [ 1,  1,  1]
                ]
            )
        )

    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    positions_by_type : dict[str, positions_like], optional
        A mapping of secondary particle type names to positions. Required if
        ``secondary_types`` is provided, otherwise ignored.
    orientations_by_type : dict[str, orientations_like], optional
        A mapping of secondary particle type names to orientation(s) in
        quaternion form. Can only be provided if ``secondary_types`` and
        ``positions_by_type`` are also provided. If not provided,
        secondary types have a default orientation of ``(1, 0, 0, 0)``.
    mass_by_type : dict[str, float], optional
        A mapping of primary and secondary particle type names to mass.
        If not provided for a given type, that type's mass defaults to ``1``.
        In contrast to positions and orientations, only one mass is allowed per
        type.
    moi_by_type : dict[str, moi_like], optional
        A mapping of primary and secondary particle type names to moment of
        inertia (MoI), expressed as a 3-vector containing the diagonal terms of
        the MoI tensor. If not provided for a given type, that
        type's MoI defaults to ``[1, 1, 1]``. In contrast to positions and
        orientations, only one MoI is allowed per type.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] | None = None,
        positions_by_type: dict[str, positions_like] | None = None,
        orientations_by_type: dict[str, orientations_like] | None = None,
        mass_by_type: dict[str, float] | None = None,
        moi_by_type: dict[str, moi_like] | None = None
    ):
        # Create defaults
        if secondary_types is None:
            secondary_types = []
        if positions_by_type is None:
            positions_by_type = {}
        if orientations_by_type is None:
            orientations_by_type = {}
        if mass_by_type is None:
            mass_by_type = {}
        if moi_by_type is None:
            moi_by_type = {}
        
        # Sanitize input dictionaries
        positions_by_type = util.sanitize(positions_by_type)
        orientations_by_type = util.sanitize(orientations_by_type)
        mass_by_type = util.sanitize(mass_by_type)
        moi_by_type = util.sanitize(moi_by_type)

        # Validate inputs
        self.validate(
            primary_type=primary_type,
            secondary_types=secondary_types,
            positions_by_type=positions_by_type,
            orientations_by_type=orientations_by_type,
            mass_by_type=mass_by_type,
            moi_by_type=moi_by_type,
        )

        # Set instance attributes
        self._primary_type = str(primary_type)
        self._secondary_types = secondary_types
        self._positions_by_type = positions_by_type
        self._orientations_by_type = orientations_by_type
        self._mass_by_type = mass_by_type
        self._moi_by_type = moi_by_type
        
    @classmethod
    def validate(
        cls,
        primary_type: str | None = None,
        secondary_types: list[str] | None = None,
        positions_by_type: dict[str, positions_like] | None = None,
        orientations_by_type: dict[str, orientations_like] | None = None,
        mass_by_type: dict[str, float] | None = None,
        moi_by_type: dict[str, moi_like] | None = None,
    ):
        """Ensure the keyword arguments adhere to the :ref:`body schema`."""
        # Ensure primary type is coercable to str
        if primary_type:
            try:
                _ = str(primary_type)
            except:
                raise TypeError("`primary_type` must be a string.")

        # Ensure that all secondary types are coercable to str
        if secondary_types:
            try:
                for t in secondary_types:
                    _ = str(t)
            except:
                raise TypeError("All secondary type names must be strings.")
        
        # Ensure that all secondary types are unique
        if secondary_types:
            unique = []
            for t in secondary_types:
                if t not in unique:
                    unique.append(t)
            if len(unique) != len(secondary_types):
                raise ValueError("All secondary type names must be unique.")

        # Ensure positions are provided for every secondary type
        if secondary_types:
            if not positions_by_type:
                positions_by_type = {}
            if ts := [t for t in secondary_types if t not in positions_by_type]:
                raise ValueError(
                    f"Missing required key(s) in `positions_by_type`: "
                    + f"'{"', '".join(ts)}'. Positions must be provided for "
                    + "all secondary types."
                )
        
        # Ensure that all positions are 3-vectors
        if positions_by_type:
            if any(
                len(p) != 3 for ps in positions_by_type.values() for p in ps
            ):
                raise ValueError("All positions must be vectors of length 3.")
        
        # Ensure that for every secondary type, the number of provided
        # orientations matches the number of provided positions
        if (
            secondary_types
            and positions_by_type
            and orientations_by_type
        ):
            ts = [
                t
                for t in secondary_types
                if (
                    t in orientations_by_type
                    and (
                        len(positions_by_type[t])
                        != len(orientations_by_type[t])
                    )
                )
            ]
            if ts:
                raise ValueError(
                    "Mismatched numbers of orientations and positions for the "
                    + f"following secondary type(s): '{"', '".join(ts)}'. For "
                    + "every secondary type, the numbers of positions and "
                    + "orientations must be the same."
                )
        
        # Ensure that all orientations are quaternions
        if orientations_by_type:
            if any(
                len(o) != 4 for os in orientations_by_type.values() for o in os
            ):
                raise ValueError(
                    "All orientations must be vectors of length 4."
                )
        
        # Ensure that all masses are coercable to float
        if mass_by_type:
            try:
                for m in mass_by_type.values():
                    _ = float(m)
            except:
                raise TypeError("All masses must be floats.")

        # Ensure that all mois are 3-vectors
        if moi_by_type:
            if any(len(moi) != 3 for moi in moi_by_type.values()):
                raise ValueError(
                    "All moments of inertia must be vectors of length 3."
                )
        
    # ------------------------------- PROPERTIES -------------------------------

    @property
    def primary_type(self) -> str:
        """The type of the primary particle."""
        return self._primary_type

    @primary_type.setter
    def primary_type(self, value: str):
        """Set the type of the primary particle."""
        self.validate(primary_type=value)
        self._primary_type = str(value)

    @property
    def secondary_types(self) -> list[str]:
        """The types of the secondary particles."""
        return self._secondary_types

    @property
    def positions_by_type(self) -> dict[str, list[list[float]]]:
        """A mapping from secondary types to positions in 3D space."""
        return self._positions_by_type

    @property
    def orientations_by_type(self) -> dict[str, list[list[float]]]:
        """A mapping from secondary types to orientations as quaternions.
        
        If not specified for a type, defaults to an array of ``(1,0,0,0)``
        quaternions.
        """
        return self._orientations_by_type

    @orientations_by_type.setter
    def orientations_by_type(self, value: dict[str, orientations_like]):
        """Set a mapping from secondary types to orientations."""
        self.validate(
            secondary_types=self._secondary_types,
            positions_by_type=self._positions_by_type,
            orientations_by_type=value
        )
        self._orientations_by_type = util.sanitize(value)

    @property
    def mass_by_type(self) -> dict[str, float]:
        """A mapping from primary and secondary types to mass.
        
        Each type may only have a single mass.

        If not specified for a type, defaults to ``1``.
        """
        return self._mass_by_type

    @mass_by_type.setter
    def mass_by_type(self, value: dict[str, float]):
        """Set a mapping from primary and secondary types to mass."""
        self.validate(mass_by_type=value)
        self._mass_by_type = util.sanitize(value)

    @property
    def moi_by_type(self) -> dict[str, list[float]]:
        """A mapping from primary and secondary types to moment of inertia.
        
        Moment of Inertia is expressed as a 3-vector. Each type may only have a
        single Moment of Inertia.

        If not specified for a type, defaults to ``[1, 1, 1]``.
        """
        return self._moi_by_type

    @moi_by_type.setter
    def moi_by_type(self, value: dict[str, moi_like]):
        """Set a mapping from primary and secondary types to moment of inertia."""
        self.validate(moi_by_type=value)
        self._moi_by_type = util.sanitize(value)

    # ------------------------------- OPERATIONS -------------------------------

    def add(
        self,
        name: str,
        positions: positions_like,
        orientations: orientations_like | None = None,
        mass: float | None = None,
        moi: moi_like | None = None,
    ):
        """Add a new secondary particle type.
        
        :meta operation:

        Parameters
        ----------
        name : str
            The name of the secondary type to add.
        positions : list[list[float]]
            The positions for the new secondary type.
        orientations : list[list[float]], optional
            The orientations for the new secondary type.
        mass : float, optional
            The mass for the new secondary type.
        moi : list[float], optional
            The moi for the new secondary type.
        """
        if name in self.secondary_types:
            raise ValueError(
                f"The new type is already included in this body's secondary "
                + "types."
            )

        # Create new data structures
        positions_by_type = {name: positions}
        
        if orientations is None:
            orientations_by_type = {}
        else:
            orientations_by_type = {name: orientations}
        
        mass_by_type = {} if mass is None else {name: mass}
        moi_by_type = {} if moi is None else {name: moi}

        # Validate
        self.validate(
            secondary_types=self._secondary_types + [name],
            positions_by_type={**self._positions_by_type, **positions_by_type},
            orientations_by_type={
                **self._orientations_by_type,
                **orientations_by_type
            },
            mass_by_type={**self._mass_by_type, **mass_by_type},
            moi_by_type={**self._moi_by_type, **moi_by_type},
        )
        
        # Modify the data
        self._secondary_types.append(name)
        self._positions_by_type.update(util.sanitize(positions_by_type))
        self._orientations_by_type.update(util.sanitize(orientations_by_type))
        self._mass_by_type.update(util.sanitize(mass_by_type))
        self._moi_by_type.update(util.sanitize(moi_by_type))

    def remove(self, name: str):
        """Remove a secondary particle type.

        :meta operation:
        
        Parameters
        ----------
        name : str
            The name of the secondary type to remove.
        """
        if name not in self.secondary_types:
            raise ValueError(
                f"The type is not included in this body's secondary types."
            )

        del self._secondary_types[self._secondary_types.index(name)]
        del self._positions_by_type[name]
        if name in self._orientations_by_type:
            del self._orientations_by_type[name]
        if name in self._mass_by_type:
            del self._mass_by_type[name]
        if name in self._moi_by_type:
            del self._moi_by_type[name]

    def update(
        self,
        name: str,
        positions: positions_like | None = None,
        orientations: orientations_like | None = None,
        mass: list[list[float]] | None = None,
        moi: moi_like | None = None,
    ):
        """Update data for a secondary particle type.

        :meta operation:
        
        Parameters
        ----------
        name : str
            The name of the secondary type to update.
        positions : positions_like
            The positions for the new secondary type.
        orientations : orientations_like, optional
            The orientations for the new secondary type.
        mass : float, optional
            The mass for the new secondary type.
        moi : moi_like, optional
            The moi for the new secondary type.
        """
        if name not in self.secondary_types:
            raise ValueError(
                f"The type is not included in this body's secondary types."
            )

        # Calculate updated data
        positions_by_type = copy(self._positions_by_type)
        if positions is not None:
            positions_by_type.update({name: positions})

        orientations_by_type = copy(self._orientations_by_type)
        if orientations is not None:
            orientations_by_type.update({name: orientations})
        
        mass_by_type = copy(self._mass_by_type)
        if mass is not None:
            mass_by_type.update({name: mass})

        moi_by_type = copy(self._moi_by_type)
        if moi is not None:
            moi_by_type.update({name: moi})

        # Validate
        self.validate(
            secondary_types=self._secondary_types,
            positions_by_type=positions_by_type,
            orientations_by_type=orientations_by_type,
            mass_by_type=mass_by_type,
            moi_by_type=moi_by_type,
        )
        
        # Modify the data
        self._positions_by_type.update(util.sanitize(positions_by_type))
        self._orientations_by_type.update(util.sanitize(orientations_by_type))
        self._mass_by_type.update(util.sanitize(mass_by_type))
        self._moi_by_type.update(util.sanitize(moi_by_type))

    # ---------------------------------- FROM ----------------------------------

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        include_singles: bool = False
    ) -> list[Body]:
        """Parse a HOOMD-blue `Simulation`_ to create bodies.

        .. _Simulation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html

        The returned list contains bodies defined in the simulation's state
        (see :py:meth:`~p4.Body.from_hoomd_snapshot`) and in the rigid
        constraint if there is one (see py:meth:`~p4.Body.from_hoomd_rigid`). If
        the constraint and the state have contradictory body definitions, the
        state takes precedence.

        .. note:

            If the simulation state specifies multiple masses or moments of
            inertia for a single particle type, the first values are used.

        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        include_singles : bool, default=False
            Whether to include single-particle bodies when parsing the
            simulation.
        """
        if (
            simulation.operations.integrator is None
            or simulation.operations.integrator.rigid is None
        ):
            bodies = cls.from_hoomd_snapshot(
                simulation.state.get_snapshot(),
                include_singles
            )
        
        else:
            bodies_from_rigid = cls.from_hoomd_rigid(
                simulation.operations.integrator.rigid,
                include_singles
            )
            bodies_from_state = cls.from_hoomd_snapshot(
                simulation.state.get_snapshot(),
                include_singles
            )

            # Bodies from state take precedence
            bodies = copy(bodies_from_state)
            for body in bodies_from_rigid:
                if not any(b.primary_type == body.primary_type for b in bodies):
                    bodies.append(body)
        
        if include_singles:
            return bodies
        
        else:
            return [b for b in bodies if b.secondary_types]

    @classmethod
    def from_hoomd_snapshot(
        cls,
        snapshot: hoomd.Snapshot,
        include_singles: bool = False
    ) -> list[Body]:
        """Parse a HOOMD-blue `Snapshot`_ to create one or more bodies.

        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html

        Body definitions in a snapshot are encoded in `particles.data`_, an
        array of integers that specify body ids as the indices of central
        particles.

        .. _particles.data: https://gsd.readthedocs.io/en/latest/schema-hoomd.html#chunk-particles-body
        
        TODO: wrap to fix image problem

        Parameters
        ----------
        snapshot : hoomd.Snapshot
            The snapshot to parse.
        include_singles : bool, default=False
            Whether to include single-particle bodies when parsing the
            simulation.
        """
        pdata = snapshot.particles
        
        bodies = []
        for bodyid in np.unique(pdata.body):
            # -1 indicates single-particle - process it later
            if bodyid == -1:
                continue

            body_indices = np.nonzero(pdata.body == bodyid)[0]
            primary_index = body_indices.min()
            secondary_indices = [i for i in body_indices if i != primary_index]
            
            # Primary particle
            primary_type = pdata.types[pdata.typeid[primary_index]]
            primary_position = pdata.position[primary_index]
            primary_orientation = pdata.orientation[primary_index]
            primary_mass = float(pdata.mass[primary_index])
            primary_moi = pdata.moment_inertia[primary_index].tolist()

            # Only add single particle bodies if singles are included and there
            # isn't already one with the same primary type
            if (
                not secondary_indices
                and include_singles
                and not any(b.primary_type == primary_type for b in bodies)
            ):
                bodies.append(Body(
                    primary_type=primary_type,
                    mass_by_type=(
                        {}
                        if primary_mass == 1
                        else {f"{primary_type}": primary_mass}
                    ),
                    moi_by_type=(
                        {}
                        if primary_moi in [[0, 0, 0], [1, 1, 1]]
                        else {f"{primary_type}": primary_moi}
                    ),
                ))
            
            # Only add multi-particle bodies if there isn't already one with the
            # same primary type
            elif not any(b.primary_type == primary_type for b in bodies):
                secondary_types = []
                positions_by_type = defaultdict(list)
                orientations_by_type = defaultdict(list)
                if primary_mass == 1:
                    mass_by_type = {} 
                else:
                    mass_by_type = {f"{primary_type}": primary_mass}
                if primary_moi in [[0, 0, 0], [1, 1, 1]]:
                    moi_by_type = {} 
                else:
                    moi_by_type = {f"{primary_type}": primary_moi}

                for i in secondary_indices:
                    t = pdata.types[pdata.typeid[i]]
                    if t not in secondary_types:
                        secondary_types.append(t)
                    
                    positions_by_type[t].append(
                        (pdata.position[i] - primary_position).tolist()
                    )
                    orientations_by_type[t].append(
                        rowan.divide(pdata.orientation[i], primary_orientation) # TODO: test order
                            .tolist()
                    )
                    
                    mass_by_type[t] = float(pdata.mass[i])
                    moi_by_type[t] = pdata.moment_inertia[i].tolist()
                
                for t, mass in copy(mass_by_type).items():
                    if mass == 1:
                        del mass_by_type[t]

                for t, moi in copy(moi_by_type).items():
                    if moi in [[0, 0, 0], [1, 1, 1]]:
                        del moi_by_type[t]

                bodies.append(Body(
                    primary_type=primary_type,
                    secondary_types=secondary_types,
                    positions_by_type=dict(positions_by_type),
                    orientations_by_type=dict(orientations_by_type),
                    mass_by_type=mass_by_type,
                    moi_by_type=moi_by_type
                ))
            
        # Particles with bodyid == -1 are all guaranteed to be single-particles.
        # For each one, only add it if there isn't already a body with its
        # primary type
        if include_singles:
            single_indices = np.nonzero(pdata.body == -1)[0]

            for i in single_indices:
                primary_type = pdata.types[pdata.typeid[i]]
                primary_mass = float(pdata.mass[i])
                primary_moi = pdata.moment_inertia[i].tolist()

                if not any(b.primary_type == primary_type for b in bodies):
                    bodies.append(Body(
                        primary_type=primary_type,
                        mass_by_type=(
                            {}
                            if primary_mass == 1
                            else {f"{primary_type}": primary_mass}
                        ),
                        moi_by_type=(
                            {}
                            if primary_moi in [[0, 0, 0], [1, 1, 1]]
                            else {f"{primary_type}": primary_moi}
                        ),
                    ))

        return bodies

    @classmethod
    def from_hoomd_rigid(
        cls,
        rigid: hoomd.md.constrain.Rigid,
        include_singles: bool = False
    ) -> list[Body]:
        """Parse a HOOMD-blue `rigid constraint`_ to create bodies.

        .. _rigid constraint: https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/constrain/rigid.html
        
        Body definitions in a rigid constraint are encoded in `Rigid.body`_.

        .. _Rigid.body: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html#hoomd.md.constrain.Rigid.body
        
        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid
            The constraint that defines rigid bodies.
        include_singles : bool, default=False
            Whether to include single-particle bodies when parsing the rigid
            constraint.
        """
        # Hoomd does not detect nested body definitions until sim.run(), so a
        # check is needed here
        for p_t in rigid.body.keys():
            for k, v in rigid.body.items():
                if v is not None:
                    if p_t in v["constituent_types"] and rigid.body[p_t] is not None:
                        raise ValueError("Nested bodies are not supported.")

        def unique(strings):
            """Find unique values in a list of strings."""
            searched = []
            for s in strings:
                if s not in searched:
                    searched.append(s)
            return searched
        
        def data_by_type(types, data):
            """Return a mapping of unique types to their corresponding data."""
            d = {}
            for t in unique(types):
                d[t] = [x for i, x in enumerate(data) if types[i] == t]
            return d
        
        # Construct bodies
        bodies = []
        empty_dict = dict(constituent_types=[], positions=[], orientations=[])
        for p_t in rigid.body.keys():
            if rigid.body[p_t] is not None and rigid.body[p_t] != empty_dict:
                bodies.append(cls(
                    primary_type=p_t,
                    secondary_types=unique(rigid.body[p_t]["constituent_types"]),
                    positions_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["positions"]]
                    ),
                    orientations_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["orientations"]]
                    )
                ))
            elif (
                include_singles
                and (
                    rigid.body[p_t] is None
                    or rigid.body[p_t].to_base() == empty_dict
                )
            ):
                bodies.append(cls(primary_type=p_t))

        return bodies

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str = "p4.body"
    ) -> Body:
        """Create a body from JSON.

        a JSON path may be provided to control the location that the body data
        is retrieved from. See :meth:`~p4.Body.to_json` for an explanation of
        JSON path formatting.

        .. note::
            The internal data structures for the ``Body`` class have native
            JSON analogues, so the JSON representation is simply
            ``Body.__dict__``.

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str, default='p4.body'
            The location within the JSON file to retrieve the body's
            representation from.
        
        Raises
        ------
        ValueError
            If the JSON file does not have the keys and values required for
            instantiating a Body.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        if json_path == ".":
            data = cls._convert_json_dict(data)            
        
        else:
            current_container = data
            for name in json_path.split("."):
                current_container = current_container[name]
            data = cls._convert_json_dict(current_container)
        
        required_args = [
            "primary_type",
            "secondary_types",
            "positions_by_type",
            "orientations_by_type",
            "mass_by_type",
            "moi_by_type"
        ]
        for required_arg in required_args:
            if required_arg not in data:
                raise ValueError(
                    f"Required arg {required_arg} not found in '{filename}' "
                    + f"at path '{json_path}'."
                )
        for key in copy(data):
            if key not in required_args:
                del data[key]

        return cls(**data)

    @classmethod
    def _convert_json_dict(cls, json_dict: dict) -> dict:
        """Convert a JSON-compliant dict into an instantiation-ready dict.
        
        NOTE: this apparently useless method is included here for convenience
        in the JSON import method in System. It may be refactored out of
        existence later.
        """
        return json_dict

    # ----------------------------------- TO -----------------------------------

    def to_hoomd_rigid(
        self,
        rigid: hoomd.md.constrain.Rigid | None = None,
    ) -> hoomd.md.constrain.Rigid:
        """Convert the body to a HOOMD-blue `rigid constraint`_.

        An existing rigid constraint may be passed to this method, in which case
        this body is merely added to it.

        .. _rigid constraint: https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/constrain/rigid.html

        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid, optional
            An existing constraint instance to use. If not provided, a new one
            is created.
        """
        types = []
        positions = []
        orientations = []

        for t in self.secondary_types:
            types.extend([t for _ in self.positions_by_type[t]])
            positions.extend(self.positions_by_type[t])
            orientations.extend(
                self.orientations_by_type.get(
                    t,
                    [[1, 0, 0, 0] for _ in self.positions_by_type[t]]
                )
            )

        if rigid is None:
            rigid = hoomd.md.constrain.Rigid()

        rigid.body[self.primary_type] = {
            "constituent_types": types,
            "positions": positions,
            "orientations": orientations
        }

        return rigid

    def to_hoomd_snapshot(self) -> hoomd.Snapshot:
        """Export the body to a HOOMD-blue `Snapshot`_.
        
        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html
        """
        frame = gsd.hoomd.Frame()
        
        types = [self.primary_type] + self.secondary_types
        typeids = [0]
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0]], dtype=np.float32)
        masses = [self.mass_by_type.get(self.primary_type, 1)]
        mois = np.array(
            [self.moi_by_type.get(self.primary_type, [1, 1, 1])],
            dtype=np.float32
        )
        bodyids = [0] if self.secondary_types else [-1]

        for t, ps in self.positions_by_type.items():
            os = self.orientations_by_type.get(t, [[1, 0, 0, 0] for _ in ps])
            mass = self.mass_by_type.get(t, 1)
            moi = self.moi_by_type.get(t, [1, 1, 1])
            tid = types.index(t)

            typeids.extend([tid for _ in ps])
            positions = np.vstack((positions, ps))
            orientations = np.vstack((orientations, os))
            masses.extend([mass for _ in ps])
            mois = np.vstack((mois, [moi for _ in ps]))
            bodyids.extend([0 for _ in ps])

        frame.configuration.box = [
            3*max(np.abs(positions[:,0].max()), np.abs(positions[:,0].min())+1),
            3*max(np.abs(positions[:,1].max()), np.abs(positions[:,1].min())+1),
            3*max(np.abs(positions[:,2].max()), np.abs(positions[:,2].min())+1),
            0.0,
            0.0,
            0.0
        ]
        frame.particles.N = positions.shape[0]
        frame.particles.types = types
        frame.particles.typeid = typeids
        frame.particles.position = positions
        frame.particles.orientation = orientations
        frame.particles.mass = masses
        frame.particles.moment_inertia = mois
        frame.particles.body = bodyids

        return hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )

    def to_gsd(
        self,
        filename: os.PathLike,
        type_shapes: dict | None = None
    ):
        """Export the body to GSD.
        
        The exported GSD file has a single frame with the body centered on
        the origin.

        .. _shapes module: https://coxeter.readthedocs.io/en/latest/package-shapes.html

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the GSD file.
        type_shapes: dict[str, coxeter.shapes], optional
            If provided, encodes geometry for provided particle types. Specify
            a geometry using Coxeter's `shapes module`_.
        """
        snapshot = self.to_hoomd_snapshot()

        frame = util.snapshot_to_frame(snapshot)

        if type_shapes:
            gsd_shape_specs = []
            for t in frame.particles.types:
                s = type_shapes.get(t)
                accepted_types = (
                    coxeter.shapes.Sphere,
                    coxeter.shapes.Ellipsoid,
                    coxeter.shapes.Polygon,
                    coxeter.shapes.Polyhedron,
                )
                if isinstance(s, (accepted_types)):
                    gsd_shape_specs.append(s.gsd_shape_spec)
                else:
                    gsd_shape_specs.append({})

            frame.particles.type_shapes = gsd_shape_specs

        with gsd.hoomd.open(filename, "w") as f:
            f.append(frame)

    def to_json(
        self,
        filename: os.PathLike,
        json_path: str = "p4.body",
        indent: str | int | None = None
    ):
        """Export the body to JSON.
        
        If ``filename`` points to an existing file, a JSON path may be provided
        to ensure the body data does not clash with existing data in the file.

        A JSON path that looks like ``'a.b.c'`` represents the following
        location:

        .. code-block::

            <root>
            └─ a
               └─ b
                  └─ c
                     └─ <data will go here>

        If the path specifies a location that already contains data, the
        contents of that location may be overwritten.

        .. note::
            The internal data structures for the ``Body`` class have native
            JSON analogues, so the JSON representation is simply
            ``Body.__dict__``.
                     
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str or None, default='p4.body'
            The location within the JSON file to put the body's representation
            in. Only used if ``filename`` already exists. If ``'.'`` is
            provided, then the representation is placed at the root level.
        indent : str or int, optional
            The string or number of spaces to use when indenting newlines in the
            JSON file. If not provided, there are no newlines.
        """
        path = Path(filename)
        data = self._to_json_dict()

        if not path.exists():
            path.touch()
            existing_data = {}
        else:
            with open(path, "r") as f:
                existing_data = json.load(f)
            
        if json_path == ".":
            for k, v in data.items():
                existing_data[k] = v
        
        else:
            names = json_path.split(".")
            current_container = existing_data
            for i, name in enumerate(names):
                if name not in current_container:
                    current_container[name] = {}
                if i < (len(names) - 1):
                    current_container = current_container[name]
                else:
                    # try to write alongside existing data if possible...
                    if isinstance(current_container[name], dict):
                        current_container[name].update(data)
                    elif isinstance(current_container[name], list):
                        current_container[name].append(data)
                    # ... and insert or overwrite if not
                    else:
                        current_container[name] = data

        with open(path, "w") as f:
            json.dump(existing_data, f, indent=indent)

    def _to_json_dict(self) -> dict:
        """Return a JSON-compliant dictionary representing this body."""
        return dict(
            primary_type=self.primary_type,
            secondary_types=self.secondary_types,
            positions_by_type=self.positions_by_type,
            orientations_by_type=self.orientations_by_type,
            mass_by_type=self.mass_by_type,
            moi_by_type=self.moi_by_type
        )

    # -------------------------------- PLOTTING --------------------------------

    def plot(
        self,
        type_shapes: dict[str, coxeter.shapes.Polyhedron] | None = None,
        type_styles: dict[str, dict] | None = None,
        ignore_types: list[str] | None = None,
        slice: dict[str, float] | None = None,
        schematic_slice: bool = False,
        **kwargs
    ) -> tuple[plotly.graph_objects.Figure, list]:
        """Interactively plot the body using `Plotly`_.

        Slicing is supported along the X, Y, and Z axes via the ``slice``
        parameter. A slice along one axis (``slice={"x": 1}``) is 2D, while a
        slice along two axes (``slice={"x": 1, "y": 1}``) is 1D.

        Shapes and styles may be specified for specific types. A shape must be
        specified as a `Coxeter Polyhedron`_. A style must specified as a
        dictionary which may have the following keys and values:

        * ``color`` [``str``] - The symbol's color. Plotly accepts color strings
          in `standard HTML/CSS formats`_ (for example, `rgb`_), as well as
          `many named colors`_.
        
        * ``opacity`` [``float`` 0 to 1] - The symbol's opacity.
        
        * ``size`` [``float`` > 0]- The symbol's size. When the type is
          represented by a point, this corresponds to the symbol's ``size``
          attribute (see Plotly docs for `2D`_ and `3D`_). When the type is
          represented by a line, this setting corresponds to the line's
          ``width`` `attribute`_. When the type is represented by a polygon or
          polyhedron, this setting is ignored.

        .. _Plotly: https://plotly.com/
        .. _Coxeter Polyhedron: https://coxeter.readthedocs.io/en/latest/package-shapes.html#coxeter.shapes.Polyhedron
        .. _standard HTML/CSS formats: https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/color
        .. _rgb: https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/color_value/rgb
        .. _many named colors: https://plotly.com/python/css-colors/
        .. _2d: https://plotly.com/python/reference/scatter/#scatter-marker-size
        .. _3d: https://plotly.com/python/reference/scatter3d/#scatter3d-marker-size
        .. _attribute: https://plotly.com/python/reference/scatter/#scatter-line-width

        Parameters
        ----------
        type_shapes : dict, optional
            A mapping from particle type name [``str``] to shape
            [``coxeter.shapes.Polyhedron``]. If no shape is provided for a type,
            it will be plotted as a sphere.
        type_styles : dict, optional
            A mapping from particle type name to style, where style is given as
            a dictionary which may have the keys 'color', 'opacity', and 'size'.
            See above for more information.
        ignore_types : list[str], optional
            The names of the particle types to exclude from the plot.
        slice : dict, optional
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        schematic_slice : bool, default=False
            If True, the slice is shown schematically in a 3D view. A 2D slice
            appears like a plane intersecting with the body, while a 1D slice
            appears like a line intersecting with the body.
        **kwargs
            Other keyword arguments are passed to the following functions:

            * ``p4.util.plot_layout()`` - TODO: add link

            * ``p4.util.snapshot_schematic_slice_trace()``
        """
        # Set defaults
        if not type_shapes:
            type_shapes = {}
        if not type_styles:
            type_styles = {}
        if not ignore_types:
            ignore_types = []
        if not slice:
            slice = {}

        default_colors = util.WONG_COLORS

        # Calculate snapshot
        snapshot = self.to_hoomd_snapshot()

        # Construct the figure type-by-type
        figure = plotly.graph_objects.Figure()
        
        if len(slice) == 0 or schematic_slice:
            traces = util.snapshot_3D_traces(
                snapshot=snapshot,
                type_shapes=type_shapes,
                type_styles=type_styles,
                ignore_types=ignore_types,
                default_colors=default_colors
            )

            if schematic_slice:
                allowed_kwarg_names = (
                    signature(util.snapshot_schematic_slice_trace)
                        .parameters
                        .keys()
                )
                schematic_slice_kwargs = {
                    k: v for k, v in kwargs.items() if k in allowed_kwarg_names
                }
                traces.append(
                    util.snapshot_schematic_slice_trace(
                        snapshot=snapshot,
                        slice=slice,
                        **schematic_slice_kwargs
                    )
                )
        
        elif len(slice) == 1 and not schematic_slice:
            traces = util.snapshot_2D_traces(
                snapshot=snapshot,
                slice=slice,
                type_shapes=type_shapes,
                type_styles=type_styles,
                ignore_types=ignore_types,
                default_colors=default_colors
            )
        
        elif len(slice) == 2 and not schematic_slice:
            traces = util.snapshot_1D_traces(
                snapshot=snapshot,
                slice=slice,
                type_shapes=type_shapes,
                type_styles=type_styles,
                ignore_types=ignore_types,
                default_colors=default_colors
            )

        for trace in traces:
            if trace is not None:
                figure.add_trace(trace)

        # Style the plot
        allowed_kwarg_names = signature(util.plot_layout).parameters.keys()
        layout_kwargs = {
            k: v for k, v in kwargs.items() if k in allowed_kwarg_names
        }
        if "show_grid" not in layout_kwargs:
            layout_kwargs["show_grid"] = True
        layout = util.plot_layout(slice=slice, **layout_kwargs)
        figure.update_layout(layout)

        return figure, traces

    # --------------------------------- OTHER ----------------------------------

    def _is_rigid(self, interactions: list["Interaction"]) -> bool:
        """Whether the body must represent a rigid body for some interactions.
        
        If any of the interactions specify a non-zero ``r_cut`` for any of the
        body's secondary types, then the body must be rigid.

        Parameters
        ----------
        interactions : list[Interaction]
            The interactions to check over.
        """
        common_single_types = any(
            t in self.secondary_types
            for interaction in interactions
            for t in interaction.interacting_types("single")
        )
        common_pair_types = any(
            p[0] in self.secondary_types or p[1] in self.secondary_types
            for interaction in interactions
            for p in interaction.interacting_types("pair")
        )
        nonzero_default_r_cut = any(
            (
                (
                    len(self.secondary_types) > 0
                    and interaction.default_params["r_cut"] > 0
                ) or interaction.initial_args.get("default_r_cut", 0) > 0
            )
            for interaction in interactions
        )
        return common_single_types or common_pair_types or nonzero_default_r_cut

    def __eq__(self, other) -> bool:
        """Bodies are equal if their properties are equal or equivalent."""
        primary_same = self.primary_type == other.primary_type
        if not primary_same:
            return False
        
        secondary_same = self.secondary_types == other.secondary_types
        if not secondary_same:
            return False
        
        positions_same = self.positions_by_type == other.positions_by_type
        if not positions_same:
            return False

        orientations_same = (
            self.orientations_by_type == other.orientations_by_type
        )

        orientations_missing_from_self = (
            set(other.orientations_by_type) - set(self.orientations_by_type)
        )
        orientations_missing_from_other = (
            set(self.orientations_by_type) - set(other.orientations_by_type)
        )
        common_types = set(self.orientations_by_type).intersection(
            set(other.orientations_by_type)
        )
        orientations_equivalent = (
            all(
                self.orientations_by_type[t] == other.orientations_by_type[t]
                for t in common_types
            )
            and
            all(
                all(list(o) == [1,0,0,0] for o in other.orientations_by_type[t])
                for t in orientations_missing_from_self
            )
            and
            all(
                all(list(o) == [1,0,0,0] for o in self.orientations_by_type[t])
                for t in orientations_missing_from_other
            )
        )

        masses_same = (
            self.mass_by_type == other.mass_by_type
        )
        masses_missing_from_self = (
            set(other.mass_by_type) - set(self.mass_by_type)
        )
        masses_missing_from_other = (
            set(self.mass_by_type) - set(other.mass_by_type)
        )
        common_types = set(self.mass_by_type).intersection(
            set(other.mass_by_type)
        )
        masses_equivalent = (
            all(
                self.mass_by_type[t] == other.mass_by_type[t]
                for t in common_types
            )
            and all(
                other.mass_by_type[t] == 1.0 for t in masses_missing_from_self
            )
            and all(
                self.mass_by_type[t] == 1.0 for t in masses_missing_from_other
            )
        )

        moi_same = (
            self.moi_by_type == other.moi_by_type
        )
        moi_missing_from_self = (
            set(other.moi_by_type) - set(self.moi_by_type)
        )
        moi_missing_from_other = (
            set(self.moi_by_type) - set(other.moi_by_type)
        )
        common_types = set(self.moi_by_type).intersection(
            set(other.moi_by_type)
        )
        moi_equivalent = (
            all(
                self.moi_by_type[t] == other.moi_by_type[t]
                for t in common_types
            )
            and
            all(
                other.moi_by_type[t] == [1, 1, 1]
                for t in moi_missing_from_self
            )
            and all(
                self.moi_by_type[t] == [1, 1, 1]
                for t in moi_missing_from_other
            )
        )

        return (
            primary_same and secondary_same and positions_same
            and (orientations_same or orientations_equivalent)
            and (masses_same or masses_equivalent)
            and (moi_same or moi_equivalent)
        )
    
    def __repr__(self) -> str:
        return (
            "Body ("
            + f"\n\tprimary_type='{self.primary_type}',"
            + f"\n\tsecondary_types={self.secondary_types},"
            + f"\n\tpositions_by_type={self.positions_by_type},"
            + f"\n\torientations_by_type={self.orientations_by_type},"
            + f"\n\tmass_by_type={self.mass_by_type},"
            + f"\n\tmoi_by_type={self.moi_by_type},"
            + "\n)"
        )
