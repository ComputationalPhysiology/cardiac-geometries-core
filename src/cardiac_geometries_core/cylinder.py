from pathlib import Path
import gmsh

from . import utils


def cylinder(
    mesh_name: str | Path = "",
    inner_radius: float = 10.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    char_length: float = 10.0,
    verbose: bool = True,
):
    """Create a thick cylindrical shell (hollow cylinder) mesh using GMSH

    Parameters
    ----------
    mesh_name : str, optional
        Name of the mesh, by default ""
    inner_radius : float
        Inner radius of the cylinder, default is 10.0
    outer_radius : float
        Outer radius of the cylinder, default is 20.0
    height : float
        Height of the cylinder, default is 40.0
    char_length : float
        Characteristic length of the mesh, default is 10.0
    verbose : bool
        If True, GMSH will print messages to the console, default is True
    """
    path = utils.handle_mesh_name(mesh_name=mesh_name)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # Create two concentric cylinders
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)
    inner_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)

    # Boolean subtraction to get the shell
    outer = [(3, outer_cylinder)]
    inner = [(3, inner_cylinder)]
    shell, id = gmsh.model.occ.cut(outer, inner, removeTool=True)

    gmsh.model.occ.synchronize()

    # Get all surfaces to identify them
    surfaces = gmsh.model.occ.getEntities(dim=2)

    gmsh.model.addPhysicalGroup(
        dim=surfaces[0][0],
        tags=[surfaces[0][1]],
        tag=1,
        name="INSIDE",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[1][0],
        tags=[surfaces[1][1]],
        tag=2,
        name="OUTSIDE",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[2][0],
        tags=[surfaces[2][1]],
        tag=3,
        name="TOP",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[3][0],
        tags=[surfaces[3][1]],
        tag=4,
        name="BOTTOM",
    )

    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in shell], tag=5, name="VOLUME")

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    # gmsh.option.setNumber("Mesh.SaveAll", 1)

    # Write mesh to file
    gmsh.write(path.as_posix())

    # Finalize GMSH
    gmsh.finalize()

    print(f"Cylindrical shell mesh generated and saved to {mesh_name}")

    return path
