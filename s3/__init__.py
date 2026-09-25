"""Compatibility package for the S³ modules at the repository root."""
from pathlib import Path

# Keep existing `s3.*` imports and `python -m s3` working after moving the
# implementation to the repository root.
__path__.append(str(Path(__file__).resolve().parents[1]))

from .mesh import TetMesh, read_tet, write_tet
from .pipeline import PaperConfig, Objectives, run_paper

__all__ = ['TetMesh', 'read_tet', 'write_tet', 'PaperConfig', 'Objectives', 'run_paper']
