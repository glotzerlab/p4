import hoomd
import p4

def detect_error(particle_types, pair):
    sim = hoomd.util.make_example_simulation(particle_types=particle_types)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.1)
    sim.operations.integrator.forces.append(pair)
    sim.run(0)

def make_alj():
    nlist = hoomd.md.nlist.Cell(2)
    alj = hoomd.md.pair.aniso.ALJ(nlist, default_r_cut=1.0)

    alj.shape["A"] = dict(
        vertices=[],
        faces=[]
    )
    alj.shape["B"] = dict(
        vertices=[[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]],
        faces=[[0,1,2,3]]
    )
    alj.shape["C"] = dict(
        vertices=[[1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0]],
        faces=[[0,1,2,3]]
    )

    alj.params[("A", "A")] = dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("A", "A")] = 0
    alj.params[("B", "B")] = dict(epsilon=1, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("B", "B")] = 1
    alj.params[("C", "C")] = dict(epsilon=2, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("C", "C")] = 2

    alj.params[("A", "B")] = dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("A", "B")] = 0
    alj.params[("A", "C")] = dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("A", "C")] = 0

    alj.params[("B", "C")] = dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0)
    alj.r_cut[("B", "C")] = 0

    _ = detect_error(["A", "B", "C"], alj)

    return alj

def unique(items):
    searched = []
    for i in items:
        if i not in searched:
            searched.append(i)
    return searched

def get_particle_types(typeparam_dict, yes_or_all):
    """Return a list of particle type names in a typeparameter dictionary.
    
    'yes' types are defined as having non-zero r_cut values.
    
    """
    if yes_or_all == "yes":
        particle_types = []
        for k, v in typeparam_dict["r_cut"].items():
            if v != 0:
                particle_types.extend(i for i in k if i not in particle_types)
        return particle_types

    particle_types = []
    for name, typeparam in typeparam_dict.items():
        for t_k, t_v in typeparam.items():
            if isinstance(t_k, tuple):
                for i in t_k:
                    if i not in particle_types:
                        particle_types.append(i)
            elif isinstance(t_k, str):
                if t_k not in particle_types:
                    particle_types.append(i)
            else:
                raise ValueError(
                    f"Malformed typeparam dict: the value for key '{name}' must "
                    + "be a dict with tuples or strings for keys, but it has a "
                    + f"key '{t_k}'."
                )
    return particle_types
                
def parse_pair(hoomd_instance):
    # Ensure pair is fully parameterized
    tpd = {k: v.to_base() for k, v in hoomd_instance._typeparam_dict.items()}

    sim = hoomd.util.make_example_simulation(
        particle_types=get_particle_types(tpd, "all")
    )
    sim.operations.integrator = hoomd.md.Integrator(dt=0.1)
    sim.operations.integrator.forces.append(hoomd_instance)
    try:
        sim.run(0)
    except RuntimeError as e:
        raise ValueError("pair is not fully parameterized") from e

    # Constructor
    hoomd_class = type(hoomd_instance)
    
    # Initial args
    initial_args = hoomd_instance._param_dict.to_base()   # TODO: default_r_cut, default_r_on??
    
    # Delete unnecessary initial args
    del initial_args["nlist"]
    if "mode" in initial_args and initial_args["mode"] == "none":
        del initial_args["mode"]
    
    # Attributes
    yes_types = get_particle_types(tpd, "yes")

    no_single_typed_attributes = {}
    no_pair_typed_attributes = {}
    yes_single_typed_attributes = {}
    yes_pair_typed_attributes = {}

    for name, typeparam in tpd.items():
        for t_k, t_v in typeparam.items():
            if isinstance(t_k, tuple):
                if all(t in yes_types for t in t_k):
                    if t_k not in yes_pair_typed_attributes:
                        yes_pair_typed_attributes[t_k] = {}
                    if name not in yes_pair_typed_attributes[t_k]:
                        yes_pair_typed_attributes[t_k][name] = {}
                    yes_pair_typed_attributes[t_k][name] = t_v  # NOTE the order of keys # NOTE: cannot currently have yes (B, B), yes (C, C), but no (B, C)
                else:
                    if name not in no_pair_typed_attributes:
                        no_pair_typed_attributes[name] = {}
                    no_pair_typed_attributes[name] = t_v    # NOTE: this can overwrite itself
            elif isinstance(t_k, str):
                if t_k in yes_types:
                    if t_k not in yes_single_typed_attributes:
                        yes_single_typed_attributes[t_k] = {}
                    if name not in yes_single_typed_attributes[t_k]:
                        yes_single_typed_attributes[t_k][name] = {}
                    yes_single_typed_attributes[t_k][name] = t_v  # NOTE the order of keys
                else:
                    if name not in no_single_typed_attributes:
                        no_single_typed_attributes[name] = {}
                    no_single_typed_attributes[name] = t_v  # NOTE: this can overwrite itself

    kwargs=dict(
        hoomd_class=hoomd_class,
        initial_args=initial_args,
        no_single_typed_attributes=no_single_typed_attributes,
        no_pair_typed_attributes=no_pair_typed_attributes,
        yes_types=yes_types,
        yes_single_typed_attributes=yes_single_typed_attributes,
        yes_pair_typed_attributes=yes_pair_typed_attributes,
    )

    return kwargs

i = p4.Interaction(**parse_pair(make_alj()))
