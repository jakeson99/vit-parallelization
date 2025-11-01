import sys
from pathlib import Path

"""
This script adds the 'src' directory to the Python path to facilitate module imports during testing.
"""

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
