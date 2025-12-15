from typing import Tuple


class Probe:
    def __init__(
        self,
        position_resolutions: list[float],
        orientation_resolutions: list[float],
        orientation_symmetries: list[int],
    ):
        pass

    def probe(
        self,
        analyte: Analyte,
        interaction_model: dict[str, Interaction],
        outside_cutoff_distance: float, # depends to the interaction model
        inside_cutoff_distance: float,  # same as previous
        cutoff_shape,                   # same as previous
        nlist,                          # depends on the analyte - TODO: can this be calculated automatically?
        filename,
        save_gsd,
        safety_factor,
        n_processes
    ):
        # Note that all interactions in interaction model are included
        # Note that validation has to happen here because no system is
        #   instantiated
        pass


class Analyte:
    def __init__(self):
        self.frame = None
        pass

    def from_frame(frame):
        pass

    def from_particle_model(particle_model):
        pass


class Interaction:
    pass

class ParticleModel:
    pass


def parse_frame(
    frame,
    probe_type,
    analyte_type=None      # if None, whole frame is used
) -> Tuple[Probe, Analyte]:
    pass

def parse_simulation(
    simulation,
    probe_type,         # if corresponds to body type, rigid is scanned
    analyte_type=None   # if None, whole frame is used
) -> Tuple[Probe, Analyte, dict[str, Interaction]]:
    pass

def predict_optimal_n_threads():
    pass