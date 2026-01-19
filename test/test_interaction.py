from copy import deepcopy
import itertools
import hoomd
import inspect
from types import ModuleType
import sys
import pkgutil
import importlib
from typing import Literal, Iterable
import numpy as np
from p4 import Interaction
import pytest


# 1. Get all hoomd md pair classes for testing


REQUIRED_PARENT_CLASS = hoomd.md.pair.pair.Pair
EXCLUDED_TYPE_STRINGS = [
    "hoomd.md.pair.pair.Pair",
    "hoomd.md.pair.aniso.AnisotropicPair",
    "hoomd.md.pair.aniso.Patchy",
    "hoomd.md.pair.friction.FrictionalPair"
]

def cls_to_str(cls):
    """Return a string representation of the class' path, including its name."""
    return cls.__module__ + "." + cls.__name__

def get_all_local_classes(module):
    """Return a list of all classes in the current module."""
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes.append(obj)
    return classes

def get_all_submodules(module: ModuleType) -> dict[str, ModuleType]:
    """ Import all submodules of a module recursively, and return them as objects.

    Ref: https://stackoverflow.com/a/25083161/15426433

    Parameters
    ----------
    module : ModuleType
        The package or module to search.
    
    Returns
    -------
    submodules
    """
    module = sys.modules[module.__name__]
    return [
        importlib.import_module(module.__name__ + '.' + name)
        for loader, name, is_pkg in pkgutil.walk_packages(module.__path__)
    ]

def get_all_classes_recursively(module):
    """Return a list of all classes in and under a module, searching recursively."""
    if hasattr(module, "__path__"):    
        classes = []
        for submodule in get_all_submodules(module):
            classes.extend(get_all_local_classes(submodule))
        return classes
    else:
        return get_all_local_classes(module)

def get_all_classes_to_test():
    """Return a list of the types that can be converted into p4 Interactions.
    
    The criteria for inclusion are:
    1. Must be a member of hoomd.md.pair or any of its submodules (recursively).
    2. Must be a subclass of hoomd.md.pair.Pair.
    3. Must not be any of the following types:
        - hoomd.md.pair.Pair
        - hoomd.md.pair.aniso.AnisotropicPair
        - hoomd.md.pair.aniso.Patchy
    
    Note: this list changes depending on what version of hoomd is used.
    """
    all_classes = get_all_classes_recursively(hoomd.md.pair)    #1

    filtered_classes = []
    for cls in all_classes:
        if issubclass(cls, REQUIRED_PARENT_CLASS):              #2
            if cls_to_str(cls) not in EXCLUDED_TYPE_STRINGS:    #3
                filtered_classes.append(cls)

    # Ensure there are no duplicates
    assert len(filtered_classes) == len(set(filtered_classes))
    return filtered_classes

CLASSES_TO_TEST = get_all_classes_to_test()


# 2. Verify that
#   1. there are no initial args we don't know how to handle
#   2. there are no initial arg clashes that we haven't approved


NLIST = hoomd.md.nlist.Cell(2)
INITIAL_ARGS_REQUIRED = dict(
    kT=1
)
INITIAL_ARGS_OPTIONAL = dict(   # all different from API's default values except for `mode`
    default_r_cut=2,
    default_r_on=1,
    mode="none",               # NOTE: 'none' is the only mode supported by all classes, and it is required for applying the tail_correction
    tail_correction=True
)

APPROVED_INITIAL_ARG_CLASHES = [
    "nlist",
    "default_r_cut",
    "mode",
    "default_r_on",
    "kT"
]

def parse_initial_arg_names(hoomd_class, required_or_optional):
    """Return the names of required and optional args for a Pair constructor."""
    names = []

    class_params = inspect.signature(hoomd_class.__init__).parameters
    for k, v in class_params.items():
        if k in ["self", "nlist"]:
            continue
        if v.default is inspect._empty:
            if required_or_optional in ["required", "all"]:
                names.append(k)
        elif required_or_optional in ["optional", "all"]:
            names.append(k)
    
    return names

def test_no_unexpected_initial_args():
    """Ensure args for constructors are all included in constants defined above.
    
    This test is required because required and optional initial args are all
    pulled from `INITIAL_ARGS_REQUIRED` and `INITIAL_ARGS_OPTIONAL`, which would
    cause errors on instantiation if they are not included.
    """
    for cls in CLASSES_TO_TEST:
        required = parse_initial_arg_names(cls, "required")
        optional = parse_initial_arg_names(cls, "optional")
        for arg in required:
            if arg not in INITIAL_ARGS_REQUIRED:
                raise Exception(
                    f"Initial arg `{arg}` is required by `{cls}` but is not "
                    f"provided in `INITIAL_ARGS_REQUIRED`."
                )
        for arg in optional:
            if arg not in INITIAL_ARGS_OPTIONAL:
                raise Exception(
                    f"Initial arg `{arg}` is accepted by `{cls}` but is not "
                    f"provided in `INITIAL_ARGS_OPTIONAL`."
                )

def test_no_unexpected_initial_arg_clashes():
    """Ensure args for constructors are all unique unless explicitly approved.
    
    This test is required because required and optional initial args are all
    pulled from `INITIAL_ARGS_REQUIRED` and `INITIAL_ARGS_OPTIONAL`, which could
    cause errors on instantiation if different constructors have args with the
    same name but they expect different kinds of values.
    """
    classes_by_arg = {}

    for cls in CLASSES_TO_TEST:
        required = parse_initial_arg_names(cls, "required")
        optional = parse_initial_arg_names(cls, "optional")
        args = required + optional

        for arg in args:
            if arg in classes_by_arg:
                if arg not in APPROVED_INITIAL_ARG_CLASHES:
                    raise Exception(
                        f"Unapproved initial arg clash detected: `{arg}` is "
                        f"accepted by both `{cls}` and `{classes_by_arg[arg]}`."
                    )
                else:
                    classes_by_arg[arg].append(cls)
            else:
                classes_by_arg[arg] = [cls]


# 3. Manually set non-interacting attribute values and verify that
#   1. there are no attribute names that we don't know how to handle
#   2. there are no attribute types that we don't know how to handle (no 3+ tuples)
#   3. there are no attribute clashes that we haven't approved


class Required:
    pass

def flatten(array):
    """Recursively flatten arbitrarily nested arrays, ignoring strings and bytes."""
    for item in array:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item

def is_required(tp_default_item):
    """Return True if input is a required arg or 1+ arbitrarily nested arrays thereof."""
    return (
        (tp_default_item is hoomd.data.typeconverter.RequiredArg)
        or (
            isinstance(tp_default_item, Iterable)
            and any([
                subitem is hoomd.data.typeconverter.RequiredArg
                for subitem in flatten(tp_default_item)
            ])
        )
    )

def parse_typeparam_default_item_dict(d, required_or_optional):
    """Return dicts of values for typeparams.
     
    Values for required typeparams are the Required marker type. Values for
    optional typeparams are the defaults.
    
    This goes two layers deep.
    """  
    subdict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            subsubdict = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    raise Exception("typeparam dict recursion depth exceeded.")
                
                if is_required(sv):
                    if required_or_optional in ["required", "all"]:
                        subsubdict[sk] = Required
                else:
                    if required_or_optional in ["optional", "all"]:
                        subsubdict[sk] = sv
            
            if len(subsubdict.keys()) > 0:
                subdict[k] = subsubdict
        
        else:
            if is_required(v):
                if required_or_optional in ["required", "all"]:
                    subdict[k] = Required
            else:
                if required_or_optional in ["optional", "all"]:
                    subdict[k] = v
        
    return subdict

def parse_attributes(hoomd_class, single_or_pair, required_or_optional):
    """Return an attribute dictionary for a given constructor."""
    attributes = {}
    required_args = parse_initial_arg_names(hoomd_class, "required")
    initial_args = {arg: INITIAL_ARGS_REQUIRED[arg] for arg in required_args}
    tpd = hoomd_class(nlist=NLIST, **initial_args)._typeparam_dict

    for name, typeparam in tpd.items():
        something_to_add = False    # TODO: this flag indicates refactoring needed

        # If the typeparam is a dictionary mapping names to some subtypeparams,
        # then each of those subtypeparams must also be checked. This process
        # is not recursive - i.e., it does not perform this dictionary check on
        #  each of those subtypeparams.
        if type(typeparam.default) is dict:
            value = parse_typeparam_default_item_dict(typeparam.default, required_or_optional)
            
            if len(value.keys()) > 0:
                something_to_add = True

        # If the typeparam is not a dictionary, everything's simple
        else:
            if is_required(typeparam.default):
                if required_or_optional in ["required", "all"]:
                    value = Required
                    something_to_add = True
            else:
                if required_or_optional in ["optional", "all"]:
                    value = typeparam.default
                    something_to_add = True
        
        # Add the attribute value to the container
        if something_to_add:
            if typeparam._indexer.len_key == 1:
                if single_or_pair in ["single", "all"]:
                        attributes[name] = value
            
            elif typeparam._indexer.len_key == 2:
                if single_or_pair in ["pair", "all"]:
                        attributes[name] = value

            else:
                raise Exception(
                    f"Encountered an unknown typearam type: '{name}' has "
                    f"{typeparam._indexer.len_key=}, but only 1 and 2 are allowed."
                )
    
    return attributes

def get_all_attributes(classes, single_or_pair, required_or_optional):
    """Return a dictionary of all attributes for the provided classes."""
    attributes = [
        parse_attributes(cls, single_or_pair, required_or_optional)
        for cls in classes
    ]

    merged_attributes = {}
    for attribute in attributes:
        for k, v in attribute.items():
            if k not in merged_attributes:
                merged_attributes[k] = v
            else:
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if sk not in merged_attributes[k]:
                            merged_attributes[k][sk] = sv    # Note the potential for a clash between 'no' values here
                        else:
                            if isinstance(sv, dict):
                                for ssk, ssv in sv.items():
                                    if ssk not in merged_attributes[k][sk]:
                                        merged_attributes[k][sk][ssk] = ssv    # Note the potential for a clash between 'no' values here

    return merged_attributes

NO_ATTRIBUTES_REQUIRED = dict(
    # Single-typed
    shape=dict(
        vertices=[],
        faces=[],
    ),
    mu=(1,1,1),
    directors=[(1,0,0), (0,1,0), (0,0,1)],

    # Pair-typed
    r_cut=0,
    params=dict(
        epsilon=0,
        sigma_i=0.1,
        sigma_j=0.1,
        alpha=1,    # int for ALJ, float for Ewald, Morse, TWF, Zetterling
        A=1,
        kappa=1,
        lperp=1,
        lpar=1,
        pair_params=dict(
            epsilon=1,
            sigma=1,
            delta=1,
            kappa=1,
            m=6,    # cannot equal n
            n=12
        ),
        envelope_params=dict(
            alpha=1,
            omega=1
        ),
        eps=1,
        phi=1,
        beta=1,
        rmin=1,
        twozeta=1,
        rho=1,
        C=1,
        Z=1,
        a1=1,
        a2=1,
        gamma=1,
        sigma=1,
        delta=1,
        m=6,        # cannot equal n
        n=12,
        a=(1,1,1),
        b=(1,1,1),
        r0=1,
        qi=1,
        qj=2,
        aF=1,
        D0=1,
        C1=1,
        C2=1,
        eta1=1,
        eta2=1,
        k=1,
        eps_rf=1,
        r_min=2,
        U=np.array([1,1,1,1]),
        F=np.array([1,1,1,1]),
        R=1,
        mu=1,
        nu=1,
        kf=1,
        B=1,
    )
)
NO_ATTRIBUTES_OPTIONAL = dict( # different from API's default values
    # Single-typed
    shape=dict(
        rounding_radii=(1,1,1)
    ),

    # Pair-typed
    r_on=0.5,
    params=dict(
        contact_ratio_i=0.2,
        contact_ratio_j=0.2,
        average_simplices=False,
        alpha=2,
        use_charge=True,
    )
)

def get_types_with_attribute(classes, attribute, single_or_pair, required_or_optional):
    """Return a list of types with a given attribute.
    
    The attribute may be a string or a list. If it is a listt, it refers to an
    attribute (at the end of a list) that is nested successively inside other
    attributes (the rest of the list). For example, ["a", "b", "c"] refers to
    the attribute structure

    ```
    a
    ├─ b
    │  ├─ c
    ```
    """
    if isinstance(attribute, list) and len(attribute) == 1:
        attribute = attribute[0]

    if isinstance(attribute, str):
        if required_or_optional == "all":
            required_results = [
                cls
                for cls in classes
                if attribute in parse_attributes(cls, single_or_pair, "required")
            ]
            optional_results = [
                cls
                for cls in classes
                if attribute in parse_attributes(cls, single_or_pair, "optional")
            ]
            positive_results = list(set(required_results + optional_results))
        else:
            positive_results = [
                cls
                for cls in classes
                if attribute in parse_attributes(cls, single_or_pair, required_or_optional)
            ]
        return positive_results
    
    elif isinstance(attribute, list):
        positive_results = []
        for cls in classes:
            levels_searched = 0
            current_level_dict = parse_attributes(cls, single_or_pair, required_or_optional)
            for level_name in attribute:
                if level_name in current_level_dict:
                    current_level_dict = current_level_dict[level_name]
                    levels_searched += 1
            if levels_searched == len(attribute):
                positive_results.append(cls)
        return positive_results

APPROVED_ATTRIBUTE_CLASHES = dict(
    r_cut=get_types_with_attribute(CLASSES_TO_TEST, "r_cut", "all", "all"),         # always positive float
    r_on=get_types_with_attribute(CLASSES_TO_TEST, "r_on", "all", "all"),           # always positive float
    directors=get_types_with_attribute(CLASSES_TO_TEST, "directors", "all", "all"), # always N-list of 3-tuples of floats
    mu=[
        hoomd.md.pair.aniso.Dipole, # 3-tuple of floats
        hoomd.md.pair.aniso.YLZ,    # 3-tuple of floats
    ],
    params=dict(
        epsilon=get_types_with_attribute(CLASSES_TO_TEST, ["params", "epsilon"], "all", "all"), # always float
        sigma=get_types_with_attribute(CLASSES_TO_TEST, ["params", "sigma"], "all", "all"),     # always positive non-zero float
        delta=get_types_with_attribute(CLASSES_TO_TEST, ["params", "delta"], "all", "all"),     # always positive float
        alpha=[
            hoomd.md.pair.aniso.ALJ,    # int 0-3
            hoomd.md.pair.Morse,        # positive non-zero float
            hoomd.md.pair.TWF,          # positive non-zero float
            hoomd.md.pair.Zetterling,   # positive float
            hoomd.md.pair.Ewald,        # positive float
        ],
        gamma=[
            hoomd.md.pair.DPD,  # positive float
            hoomd.md.pair.DPDLJ,# positive float
        ],
        kappa=[
            hoomd.md.pair.aniso.Dipole, # positive non-zero float
            hoomd.md.pair.DLVO,    # positive non-zero float
            hoomd.md.pair.Ewald,   # positive non-zero float
            hoomd.md.pair.Yukawa,  # float
        ],
        phi=[
            hoomd.md.pair.aniso.YLZ,    # float
            hoomd.md.pair.pair.OPP,     # float
        ],
        A=[
            hoomd.md.pair.aniso.Dipole,     # float
            hoomd.md.pair.Buckingham,       # float
            hoomd.md.pair.DLVO,             # float
            hoomd.md.pair.DPD,              # float
            hoomd.md.pair.DPDConservative,  # float
            hoomd.md.pair.Zetterling,       # float
        ],
        m=[
            hoomd.md.pair.ExpandedMie,  # float, cannot equal n
            hoomd.md.pair.Mie,          # float, cannot equal n
        ],
        n=[
            hoomd.md.pair.ExpandedMie,  # float, cannot equal m
            hoomd.md.pair.Mie,          # float, cannot equal m
            hoomd.md.pair.Zetterling,   # float
        ],
        r0=[
            hoomd.md.pair.pair.LJGauss, # float
            hoomd.md.pair.pair.Morse,   # float
        ],
        qi=get_types_with_attribute(CLASSES_TO_TEST, ["params", "qi"], "all", "all"),   # always float
        qj=get_types_with_attribute(CLASSES_TO_TEST, ["params", "qj"], "all", "all"),   # always float
        aF=get_types_with_attribute(CLASSES_TO_TEST, ["params", "aF"], "all", "all"),   # always float
        envelope_params=dict(
            alpha=get_types_with_attribute(CLASSES_TO_TEST, ["params", "envelope_params", "alpha"], "pair", "required"),    # inhereted from Patchy, only used in Patchy's subclasses
            omega=get_types_with_attribute(CLASSES_TO_TEST, ["params", "envelope_params", "omega"], "pair", "required"),    # inhereted from Patchy, only used in Patchy's subclasses
        ),
        pair_params=dict(
            epsilon=get_types_with_attribute(CLASSES_TO_TEST, ["params", "pair_params", "epsilon"], "pair", "required"),# always float 
            sigma=get_types_with_attribute(CLASSES_TO_TEST, ["params", "pair_params", "sigma"], "pair", "required"),    # always float
            delta=get_types_with_attribute(CLASSES_TO_TEST, ["params", "pair_params", "delta"], "pair", "required"),    # always float
            m=[
                hoomd.md.pair.aniso.PatchyExpandedMie,  # float, cannot equal n
                hoomd.md.pair.aniso.PatchyMie,          # float, cannot equal n
            ],
            n=[
                hoomd.md.pair.aniso.PatchyExpandedMie,  # float, cannot equal m
                hoomd.md.pair.aniso.PatchyMie,          # float, cannot equal m
            ],
        ),
    )
)

def test_no_unexpected_attribute_paths():
    """Ensure all attributes parsed from classes are provided in constants.
    
    Only 'no' constants are tested (not the 'yes' constants), because if
    setting an attribute works for no values then it should also work for
    yes values.
    """
    for t in ["single", "pair"]:
        for r in ["required", "optional"]:
            if r == "required":
                const = NO_ATTRIBUTES_REQUIRED
                const_name = "NO_ATTRIBUTES_REQUIRED"
            elif r == "optional":
                const = NO_ATTRIBUTES_OPTIONAL
                const_name = "NO_ATTRIBUTES_OPTIONAL"

            attributes = get_all_attributes(CLASSES_TO_TEST, t, r)
            for k, v in attributes.items():
                reason = f"Attribute with name '{k}' is not given in {const_name}."
                assert k in const, reason

                if isinstance(v, dict):
                    for sk, sv in v.items():
                        reason = f"Attribute with name '{sk}' is not given in {const_name}['{k}']."
                        assert sk in const[k], reason

                        if isinstance(sv, dict):
                            for ssk, _ in sv.items():
                                reason = f"Attribute with name '{ssk}' is not given in {const_name}['{k}']['{sk}']."
                                assert ssk in const[k][sk], reason

def test_no_unexpected_attribute_types():
    """Ensure all attributes are set with 1 or 2-tuples of particle types.
    
    This test is required because parsing only knows how to handle those cases.
    """
    for cls in CLASSES_TO_TEST:
        required_args = parse_initial_arg_names(cls, "required")
        initial_args = {arg: INITIAL_ARGS_REQUIRED[arg] for arg in required_args}
        tpd = cls(nlist=NLIST, **initial_args)._typeparam_dict

        for typeparam in tpd.values():
            reason = "All typed attributes must be assignable to single types or pairs of types."
            assert typeparam._indexer.len_key in (1, 2), reason

def test_no_unexpected_attribute_clashes():
    """Ensure attributes for classes are all unique unless explicitly approved.
    
    This test is required because required and optional attributes are all
    pulled from `NO_ATTRIBUTES_REQUIRED` and `NO_ATTRIBUTES_OPTIONAL`,
    which could cause errors on instantiation if different constructors have
    args with the same name but they expect different kinds of values.
    """
    attributes = get_all_attributes(CLASSES_TO_TEST, "all", "all")
    
    for k, v in attributes.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    for ssk, _ in sv.items():
                        classes_with_attribute = get_types_with_attribute(CLASSES_TO_TEST, [k, sk, ssk], "all", "all")
                        if len(classes_with_attribute) > 1:
                            reason = f"Attribute with path '{k}.{sk}.{ssk}' has clashes but is not given in `APPROVED_ATTRIBUTE_CLASHES`."
                            assert ssk in APPROVED_ATTRIBUTE_CLASHES[k][sk], reason

                            unapproved_clashes = [cls for cls in classes_with_attribute if cls not in APPROVED_ATTRIBUTE_CLASHES[k][sk][ssk]]
                            reason = f"Attribute with path '{k}.{sk}.{ssk}' has the following unapproved clashes: {unapproved_clashes}."
                            assert len(unapproved_clashes) == 0, reason

                else:
                    classes_with_attribute = get_types_with_attribute(CLASSES_TO_TEST, [k, sk], "all", "all")
                    if len(classes_with_attribute) > 1:
                        reason = f"Attribute with path '{k}.{sk}' has clashes but is not given in `APPROVED_ATTRIBUTE_CLASHES`."
                        assert sk in APPROVED_ATTRIBUTE_CLASHES[k], reason
                        
                        unapproved_clashes = [cls for cls in classes_with_attribute if cls not in APPROVED_ATTRIBUTE_CLASHES[k][sk]]
                        reason = f"Attribute with path '{k}.{sk}' has the following unapproved clashes: {unapproved_clashes}."
                        assert len(unapproved_clashes) == 0, reason
        else:
            classes_with_attribute = get_types_with_attribute(CLASSES_TO_TEST, k, "all", "all")
            if len(classes_with_attribute) > 1:
                reason = f"Attribute with path '{k}' has clashes but is not given in `APPROVED_ATTRIBUTE_CLASHES`."
                assert k in APPROVED_ATTRIBUTE_CLASHES, reason
                
                unapproved_clashes = [cls for cls in classes_with_attribute if cls not in APPROVED_ATTRIBUTE_CLASHES[k]]
                reason = f"Attribute with path '{k}' has the following unapproved clashes: {unapproved_clashes}."
                assert len(unapproved_clashes) == 0, reason


# 4. Manually set yes attribute values, and verify that
#   1. all yes attributes are included in the no attributes


def get_cube_vertices(side_length):
    s = side_length
    return [
        [-s/2, -s/2, -s/2],
        [-s/2, -s/2,  s/2],
        [-s/2,  s/2, -s/2],
        [-s/2,  s/2,  s/2],
        [ s/2, -s/2, -s/2],
        [ s/2, -s/2,  s/2],
        [ s/2,  s/2, -s/2],
        [ s/2,  s/2,  s/2]
    ]

def get_cube_faces():
    return [
        [0, 2, 6, 4],
        [0, 4, 5, 1],
        [4, 6, 7, 5],
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [1, 5, 7, 3],
    ]

YES_ATTRIBUTES_REQUIRED = dict(
    # Single-typed
    shape=dict(
        vertices=get_cube_vertices(1),
        faces=get_cube_faces(),
    ),
    mu=(1,2,3),
    directors=[(1,1,0), (0,1,1), (1,0,1)],

    # Pair-typed
    r_cut=5,
    params=dict(
        epsilon=1.0,
        sigma_i=0.2,
        sigma_j=0.2,
        alpha=2,    # int for ALJ, float for Ewald, Morse, TWF, Zetterling
        A=2,
        kappa=2,
        lperp=2,
        lpar=2,
        pair_params=dict(
            epsilon=2,
            sigma=2,
            delta=2,
            kappa=2,
            m=8,    # cannot equal n
            n=10
        ),
        envelope_params=dict(
            alpha=2,
            omega=2
        ),
        eps=2,
        phi=2,
        beta=2,
        rmin=2,
        twozeta=2,
        rho=2,
        C=2,
        Z=2,
        a1=2,
        a2=2,
        gamma=2,
        sigma=2,
        delta=2,
        m=8,        # cannot equal n
        n=10,
        a=(2,2,2),
        b=(2,2,2),
        r0=2,
        qi=2,
        qj=3,
        aF=2,
        D0=2,
        C1=2,
        C2=2,
        eta1=2,
        eta2=2,
        k=2,
        eps_rf=2,
        r_min=3,
        U=np.array([1,2,3,4]),
        F=np.array([1,4,6,8]),
        R=2,
        mu=2,
        nu=2,
        kf=2,
        B=2,
    )
)
YES_ATTRIBUTES_OPTIONAL = dict( # different from API's default values
    # Single-typed
    shape=dict(
        rounding_radii=(2,2,2)
    ),

    # Pair-typed
    r_on=1,
    params=dict(
        contact_ratio_i=0.3,
        contact_ratio_j=0.3,
        average_simplices=False,
        alpha=3,
        use_charge=True,
    )
)

def test_all_yes_attributes_also_in_nos():
    """Ensure that all 'yes' attributes are also given in 'no' constants."""
    for r in ["required", "optional"]:
        if r == "required":
            yes_const = YES_ATTRIBUTES_REQUIRED
            no_const = NO_ATTRIBUTES_REQUIRED
        elif r == "optional":
            yes_const = YES_ATTRIBUTES_OPTIONAL
            no_const = NO_ATTRIBUTES_OPTIONAL
    
        for k, v in yes_const.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        for ssk, _ in sv.items():
                            reason = f"Attribute with path '{k}.{sk}.{ssk}' is given in the 'yes' constant but not in the 'no' constant."
                            assert ssk in no_const[k][sk], reason

                    else:
                        reason = f"Attribute with path '{k}.{sk}' is given in the 'yes' constant but not in the 'no' constant."
                        assert sk in no_const[k], reason
            else:
                reason = f"Attribute with path '{k}' is given in the 'yes' constant but not in the 'no' constant."
                assert k in no_const, reason


# 5. Begin actual tests. Verify that
#   1. instantiation works given valid args for every hoomd class
#   2. instantiation fails expectedly given various kinds of invalid args (not needed for every class)
#   3. 'to' methods produce expected outputs
#   4. 'from' methods work given valid args
#   5. 'from' methods fail expectedly given various kinds of invalid args

def get_typed_attributes(cls, no_or_yes, single_or_pair, required_or_all):
    """Return the appropriate attribute dict with values set from defaults and constants."""
    if no_or_yes == "no":
        const_required = NO_ATTRIBUTES_REQUIRED
        const_optional = NO_ATTRIBUTES_OPTIONAL
    elif no_or_yes == "yes":
        const_required = YES_ATTRIBUTES_REQUIRED
        const_optional = YES_ATTRIBUTES_OPTIONAL

    parsed_attributes = parse_attributes(cls, single_or_pair, required_or_all)
    final_attributes = {}
    for k, v in parsed_attributes.items():
        if isinstance(v, dict):
            if k not in final_attributes:
                final_attributes[k] = {}
            
            for sk, sv in parsed_attributes[k].items():
                if isinstance(sv, dict):
                    if sk not in final_attributes[k]:
                        final_attributes[k][sk] = {}
                    
                    for ssk, _ in parsed_attributes[k][sk].items():
                        if k in const_required and sk in const_required[k] and ssk in const_required[k][sk]:
                            final_attributes[k][sk][ssk] = const_required[k][sk][ssk]
                        elif required_or_all == "all" and k in const_optional and sk in const_optional[k] and ssk in const_optional[k][sk]:
                            final_attributes[k][sk][ssk] = const_optional[k][sk][ssk]

                else:
                    if k in const_required and sk in const_required[k]:
                        final_attributes[k][sk] = const_required[k][sk]
                    elif required_or_all == "all" and k in const_optional and sk in const_optional[k]:
                        final_attributes[k][sk] = const_optional[k][sk]
        
        else:
            if k in const_required:
                final_attributes[k] = const_required[k]
            elif required_or_all == "all" and k in const_optional:
                final_attributes[k] = const_optional[k]

    return final_attributes

def get_kwargs(cls, required_or_all):
    """Return just required or all possible kwargs for the provided constructor."""
    initial_args = {}
    for name in parse_initial_arg_names(cls, required_or_all):
        if name in INITIAL_ARGS_REQUIRED:
            initial_args[name] = INITIAL_ARGS_REQUIRED[name]
        elif required_or_all == "all" and name in INITIAL_ARGS_OPTIONAL:
            initial_args[name] = INITIAL_ARGS_OPTIONAL[name]

    no_single_typed_attributes = get_typed_attributes(cls, "no", "single", required_or_all)
    yes_single_typed_attributes = get_typed_attributes(cls, "yes", "single", required_or_all)
    no_pair_typed_attributes = get_typed_attributes(cls, "no", "pair", required_or_all)
    yes_pair_typed_attributes = get_typed_attributes(cls, "yes", "pair", required_or_all)

    kwargs = dict(
        hoomd_class=cls,
        initial_args=initial_args,
        no_single_typed_attributes=no_single_typed_attributes,
        no_pair_typed_attributes=no_pair_typed_attributes,
        yes_types=["A", "B"],
        yes_single_typed_attributes=yes_single_typed_attributes,
        yes_pair_typed_attributes=yes_pair_typed_attributes
    )

    return kwargs

def typeparam_dicts_are_equivalent(one, other):
    """Return True if all items in the typeparam dicts are equivalent."""
    same_keys = all(k in other for k in one)
    
    same_values = []
    for k in one:
        if isinstance(one[k], Iterable):
            same_length = len(one[k])
            same_items = all(i == j for i, j in zip(one[k], other[k]))
            same_values.append(same_length and same_items)
        else:
            same_values.append(one[k] == other[k])
    
    return same_keys and all(same_values)

def pairs_are_equivalent(one, other):
    """Return True if hooomd.md.pair.Pair instances are equivalent."""
    same_types = type(one) is type(other)
    try:
        same_typeparam_dicts = one._typeparam_dict == other._typeparam_dict     # TODO: I don't know why this sometimes works when the other branch doesn't
    except ValueError:
        same_typeparam_dicts = typeparam_dicts_are_equivalent(
            one._typeparam_dict,
            other._typeparam_dict
        )
    return same_types and same_typeparam_dicts

@pytest.mark.parametrize("cls", CLASSES_TO_TEST)
@pytest.mark.parametrize("required_or_all", ["required", "all"])
def test_instantiation_valid(cls, required_or_all):
    """Ensure every covered hoomd class can be instantiated and exported to hoomd instances with valid parameters."""    
    kwargs = get_kwargs(cls, required_or_all)
    _ = Interaction(**kwargs)

INVALID_PARAMETERS = [
    # Initial args
    dict(   # missing required names
        hoomd_class=hoomd.md.pair.DPD,
        initial_args=dict(),
        no_single_typed_attributes=dict(),
        no_pair_typed_attributes=dict(
            params=dict(A=0, gamma=1),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(),
        yes_pair_typed_attributes=dict(
            params=dict(A=1, gamma=1),
            r_cut=1
        )
    ),
    dict(   # unexpected names
        hoomd_class=hoomd.md.pair.DPD,
        initial_args=dict(kT=1, wrong=None),
        no_single_typed_attributes=dict(),
        no_pair_typed_attributes=dict(
            params=dict(A=0, gamma=1),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(),
        yes_pair_typed_attributes=dict(
            params=dict(A=1, gamma=1),
            r_cut=1
        )
    ),
    dict(   # wrong values
        hoomd_class=hoomd.md.pair.DPD,
        initial_args=dict(kT=None),
        no_single_typed_attributes=dict(),
        no_pair_typed_attributes=dict(
            params=dict(A=0, gamma=1),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(),
        yes_pair_typed_attributes=dict(
            params=dict(A=1, gamma=1),
            r_cut=1
        )
    ),

    # Yes types
    dict(   # empty
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=[],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),

    # 'no' single-typed attributes
    dict(   # missing required names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # unexpected names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[], wrong=None)
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # wrong values
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=None)
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),

    # 'no' pair-typed attributes
    dict(   # missing required names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # unexpected names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0, wrong=None),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # wrong values
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=None),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),

    # 'yes' single-typed attributes
    dict(   # missing required names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1))
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # unexpected names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces(), wrong=None)
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),
    dict(   # wrong values
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=None)
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1),
            r_cut=1
        )
    ),

    # 'yes' pair-typed attributes
    dict(   # missing required names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2),
            r_cut=1
        )
    ),
    dict(   # unexpected names
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=1, wrong=None),
            r_cut=1
        )
    ),
    dict(   # wrong values
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(nlist=NLIST),
        no_single_typed_attributes=dict(
            shape=dict(vertices=[], faces=[])
        ),
        no_pair_typed_attributes=dict(
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            r_cut=0
        ),
        yes_types=["A", "B"],
        yes_single_typed_attributes=dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        ),
        yes_pair_typed_attributes=dict(
            params=dict(epsilon=1, sigma_i=0.2, sigma_j=0.2, alpha=None),
            r_cut=1
        )
    ),
]

@pytest.mark.parametrize("params", INVALID_PARAMETERS)
def test_instantiation_invalid(params):
    """Ensure instantiation fails predictably with invalid parameters.
    
    Only one hoomd class is tested.
    """
    with pytest.raises((TypeError, ValueError)):
        _ = Interaction(**params)

@pytest.mark.parametrize("cls", CLASSES_TO_TEST)
@pytest.mark.parametrize("required_or_all", ["required", "all"])
def test_to_hoomd_instance(cls, required_or_all):
    """Ensure for every coverted hoomd class an Interaction can be converted to an unparameterized hoomd instance."""
    kwargs = get_kwargs(cls, required_or_all)
    interaction = Interaction(**kwargs)

    pair = cls(nlist=NLIST, **kwargs["initial_args"])
    assert pairs_are_equivalent(pair, interaction.to_hoomd_instance(NLIST))

@pytest.mark.parametrize("cls", CLASSES_TO_TEST)
@pytest.mark.parametrize("required_or_all", ["required", "all"])
def test_to_parameterized_hoomd_instance(cls, required_or_all):
    """Ensure for every coverted hoomd class an Interaction can be converted to a parameterized hoomd instance."""
    kwargs = get_kwargs(cls, required_or_all)
    interaction = Interaction(**kwargs)

    pair = cls(nlist=NLIST, **kwargs["initial_args"])

    all_types = ["A", "B", "C"]
    all_type_pairs = list(itertools.combinations(all_types, 2))
    all_type_pairs.extend((t, t) for t in all_types)
    yes_type_pairs = []

    for p in all_type_pairs:
        if (p[0] in kwargs["yes_types"] and p[1] in kwargs["yes_types"] and p[0] != p[1]):
            yes_type_pairs.append(p)

    # No single-typed
    for a_t in all_types:
        for k, v in kwargs["no_single_typed_attributes"].items():
            getattr(pair, k)[a_t] = v
    
    # No pair-typed
    for a_p in all_type_pairs:
        for k, v in kwargs["no_pair_typed_attributes"].items():
            getattr(pair, k)[a_p] = v
    
    # Yes single-typed
    for y_t in kwargs["yes_types"]:
        for k, v in kwargs["yes_single_typed_attributes"].items():
            getattr(pair, k)[y_t] = v

    # Yes pair-typed
    for y_p in yes_type_pairs:
        for k, v in kwargs["yes_pair_typed_attributes"].items():
            getattr(pair, k)[y_p] = v
    
    assert pairs_are_equivalent(pair, interaction.to_parameterized_hoomd_instance(NLIST, all_types))

@pytest.mark.parametrize("cls", [hoomd.md.pair.LJ, hoomd.md.pair.aniso.ALJ])
def test_from_hoomd_pair_valid(cls):
    """Ensure every covered hoomd class can be parsed into an Interaction."""
    kwargs = get_kwargs(cls, "all")
    interaction = Interaction(**kwargs)
    pair = interaction.to_parameterized_hoomd_instance(nlist=hoomd.md.nlist.Cell(0), all_types=kwargs["yes_types"])
    breakpoint()
    other = Interaction.from_hoomd_pair(pair)
    assert interaction == other

def get_invalid_pairs(cls):
    """Return a list of hoomd pairs corresponding to a type that are wrongly parameterized in several ways.
    
    1. Missing a top-level typeparam
    2. Missing a particle type (or type pair) for a typeparam
    3. Include a typeparam that shouldn't be there
    4. Include a typeparam with a wrong value
    """
    initial_args = {}
    for name in parse_initial_arg_names(cls, "required"):
        if name in INITIAL_ARGS_REQUIRED:
            initial_args[name] = INITIAL_ARGS_REQUIRED[name]
    unparameterized_pair = cls(**initial_args)

    pairs = []

    # Missing a top-level typeparam
    parameterized_pair = deepcopy(unparameterized_pair)

    
    # Missing a particle type (or type pair) for a typeparam
    parameterized_pair = deepcopy(unparameterized_pair)

    # Include a typeparam that shouldn't be there
    parameterized_pair = deepcopy(unparameterized_pair)
    
    # Include a typeparam with a wrong value
    parameterized_pair = deepcopy(unparameterized_pair)

    return pairs

# INVALID_PAIRS = get_invalid_pairs(hoomd.md.pair.LJ)

# @pytest.mark.parametrize("invalid_pair", INVALID_PAIRS)
# def test_from_hoomd_pair_invalid(invalid_pair):
#     """Ensure pair parsing fails expectedly when the hoomd instance is not properly parameterized."""
#     with pytest.raises(ValueError):
#         _ = Interaction.from_hoomd_pair(invalid_pair)

@pytest.mark.parametrize("cls", [hoomd.md.pair.LJ, hoomd.md.pair.aniso.ALJ])
@pytest.mark.parametrize("multiple_forces", [False, True])
def test_from_hoomd_integrator_valid(cls, multiple_forces):
    """Ensure a hoomd integrator can be parsed into one or more Interactions."""
    integrator = hoomd.md.Integrator(dt=0.1)
    forces = []

    kwargs = get_kwargs(cls, "required")
    interaction = Interaction(**kwargs)
    pair = interaction.to_parameterized_hoomd_instance(NLIST)
    
    forces.append(pair)

    if multiple_forces:
        kwargs = get_kwargs(hoomd.md.pair.DPD, "required")
        interaction = Interaction(**kwargs)
        pair = Interaction.to_parameterized_hoomd_instance(NLIST)
        forces.append(pair)
    
    integrator.forces = forces
    assert forces == Interaction.from_hoomd_integrator(integrator)


INVALID_INTEGRATORS = [
    # invalid pairs
    # no forces
    hoomd.md.Integrator(dt=0.1)
]

# def test_from_hoomd_integrator_invalid(invalid_integrator):
#     """Ensure integrator parsing fails expectedly."""
#     # TODO: test both expected failure modes
#     with pytest.raises(ValueError):
#         _ = Interaction.from_hoomd_integrator(invalid_integrator)

# def test_from_hoomd_simulation_valid(simulation, expected):
#     """Ensure a hoomd simulation can be parsed into one or more Interactions."""
#     interaction = Interaction.from_hoomd_integrator(simulation)
#     assert interaction == expected

INVALID_SIMULATIONS = [
    # invalid pairs
    # integrator with no forces
    # no integrator
]

# def test_from_hoomd_simulation_invalid(invalid_simulation):
#     """Ensure hoomd simulation parsing fails expectedly."""
#     # TODO: test all three expected failure modes
#     with pytest.raises(ValueError):
#         _ = Interaction.from_hoomd_simulation(invalid_simulation)
