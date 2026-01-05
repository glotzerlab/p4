import inspect
from typing import Any, Callable

def get_interaction_constructor_params_from_constructor(
    constructor: Callable
) -> dict[str, Any]:
    """Return a dict interaction instantiation parameters from constructor."""
    # get constructor defaults
    parameters_without_defaults = []
    kw_parameters_and_defaults = {}
    signature = inspect.signature(constructor)
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            kw_parameters_and_defaults[name] = parameter.default
        else:
            parameters_without_defaults.append(name)
    
    if parameters_without_defaults != ["nlist"]:
        raise TypeError(
            f"Constructor {constructor} has parameters without defaults that "
            f"are not handled: {parameters_without_defaults}."
        )

    # calculate the parameters for Interaction()
    hoomd_class = constructor
    initial_inputs = dict(
        nlist=hoomd.md.nlist.Cell(2),
        **kw_parameters_and_defaults
    )
    # default_single_typed_attributes   # TODO: return here

# cover all hoomd.md.pair classes
def test_interaction_valid_construction():
    pass

# wrong initial inputs
# wrong typed attributes
def test_interaction_invalid_construction():
    pass

# with secondary types
# without secondary types
def test_particlemodel_valid_construction():
    pass

# no primary type
# no secondary types but callable
# secondary types but no callable
def test_particlemodel_invalid_construction():
    pass

# with secondary types
# without secondary types
def test_particlemodel_methods():
    pass

def test_system_valid_construction():
    pass

def test_system_invalid_construction():
    pass

def test_system_interaction_getters():
    pass

def test_system_probe_potential():
    pass

def test_field_valid_construction():
    pass

def test_field_invalid_construction():
    pass

def test_field_processing():
    pass

def test_field_saving():
    pass

def test_field_plotting():
    pass