import tempfile
from typing import NamedTuple
from pathlib import Path


def handle_mesh_name(mesh_name: str | Path = "") -> Path:
    if mesh_name == "":
        fd, mesh_name = tempfile.mkstemp(suffix=".msh")
    return Path(mesh_name).with_suffix(".msh")


class GmshOutput(NamedTuple):
    path: Path
    logs: list[str]
