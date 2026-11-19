# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Functions for processing and sanitizing data."""

import csv
from io import StringIO

import numpy as np


# --------------------------------- VALIDATION ---------------------------------


def sanitize(d: dict):
    """Traverse a dictionary, converting numpy types to native python analogues.
    
    Parameters
    ----------
    d : dict
        The dictionary to traverse.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            sanitize(v)
        else:
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.number):
                d[k] = v.item()
            elif isinstance(v, (list, tuple)):
                if v:
                    new_v = None
                    if isinstance(v[0], np.ndarray):
                        new_v = [i.tolist() for i in v]
                    elif isinstance(v[0], np.floating):
                        new_v = [float(i) for i in v]
                    elif isinstance(v[0], np.integer):
                        new_v = [int(i) for i in v]
                    
                    # Convert back to tuple if necessary
                    if new_v and isinstance(v, tuple):
                        new_v = tuple(new_v)
                    
                    if new_v:
                        d[k] = new_v
                    else:
                        d[k] = v
        
    return d
