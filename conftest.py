from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser
import sys

# Check if source/data exists - if it doesn't, then the data generation script
# must be run.
import importlib
from pathlib import Path
if not (Path(__file__).parent / "doc/source/data").exists():
    sys.path.append(str(Path(__file__).parent / "doc"))
    _ = importlib.import_module("generate_data_and_plots")

# Use sybil to ensure none of the examples error
pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()],
    patterns=['*.rst', '*.py'],
).pytest()