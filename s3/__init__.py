"""Python port of S³ numerical stages. See PORT_STATUS.md for fidelity scope."""
from .mesh import TetMesh, read_tet, write_tet
from .pipeline import PaperConfig, Objectives, run_paper

__all__ = ['TetMesh', 'read_tet', 'write_tet','PaperConfig','Objectives','run_paper']
