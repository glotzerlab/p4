# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Functions for processing and sanitizing data."""

import csv
from io import StringIO

import numpy as np


def merge_tables(table_csvs: list[StringIO]):
    """Combine an array of tables stored in string buffers.

    Parameters
    ----------
    table_csvs : list[StringIO]
        The tables to merge. The tables should be in CSV format and stored
        in string buffers.

    Returns
    -------
    table
        The merged table.
    """
    merged_table = StringIO()
    writer = csv.writer(merged_table)

    for i, table in enumerate(table_csvs):
        table.seek(0)
        
        # skip header for all tables after the first
        if i != 0:
            next(table)

        reader = csv.reader(table)

        for row in reader:
            writer.writerow(row)
        
    return merged_table

def clean_header(table: StringIO):
    """Simplify the header of a table string buffer.

    This function strips whitespace from column names and if a column named 
    'md.compute.ThermodynamicQuantities.potential_energy' is present, it is
    rebnamed to 'U'.

    Parameters
    ----------
    table : StringIO
        The CSV-formatted string buffer.

    Returns
    -------
    cleaned_table
    """
    # Calculate the clean column names
    table.seek(0)
    raw_header = table.readline()

    raw_columns = raw_header.split(",")
    clean_columns = []
    for c in raw_columns:
        if c.strip() == "md.compute.ThermodynamicQuantities.potential_energy":
            clean_columns.append("U")
        else:
            clean_columns.append(c.strip())

    # Build the clean table
    cleaned_table = StringIO()
    writer = csv.writer(cleaned_table)
    writer.writerow(clean_columns)

    table.seek(0)
    next(table)

    for row in csv.reader(table):
        writer.writerow(row)
    
    return cleaned_table


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
