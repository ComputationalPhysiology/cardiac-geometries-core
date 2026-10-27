import tempfile
from pathlib import Path

import gmsh


def handle_mesh_name(mesh_name: str | Path = "") -> Path:
    if mesh_name == "":
        fd, mesh_name = tempfile.mkstemp(suffix=".msh")
    return Path(mesh_name).with_suffix(".msh")


def set_order(order: int) -> None:
    """Set the order of the mesh in the current gmsh model.

    The nodes added for orders higher than one are placed on the geometry, so
    curved boundaries are represented by curved elements. Elements that this
    distorts too much are then untangled by the high order optimizer.

    Parameters
    ----------
    order : int
        Order of the mesh elements
    """
    if order < 1:
        raise ValueError(f"The order of the mesh must be at least 1, got {order}")
    if order == 1:
        return

    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.optimize("HighOrder")
