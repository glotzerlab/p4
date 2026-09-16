from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser
import sys


# Use sybil to ensure none of the examples error
pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()],
    patterns=['*.rst', '*.py'],
).pytest()
