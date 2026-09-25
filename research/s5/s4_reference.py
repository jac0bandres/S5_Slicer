"""Compatibility import for the archived S5 scripts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.s4.s4_reference import *
