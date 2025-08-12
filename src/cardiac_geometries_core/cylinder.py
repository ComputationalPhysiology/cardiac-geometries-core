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


def cylinder_flat_sides(
    mesh_name: str | Path = "cylinder_flat_sides.msh",
    inner_radius: float = 10.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    flat_distance: float = 8.0,
    char_length: float = 3.0,
    verbose: bool = True,
):
    """
    Create a thick cylindrical shell mesh with two flat inner sides using GMSH.

    The flat sides are parallel to the YZ-plane, located at x = +/- flat_distance.

    Parameters
    ----------
    mesh_name : str or Path, optional
        Name of the mesh file to be created.
    inner_radius : float
        The radius of the inner cylindrical surfaces.
    outer_radius : float
        The radius of the outer cylindrical surface.
    height : float
        The height of the cylinder.
    flat_distance : float
        The distance from the center (x=0) to the flat inner surfaces.
        This value must be positive and less than inner_radius.
    char_length : float
        The characteristic length of the mesh elements.
    verbose : bool
        If True, GMSH will print messages to the console.
    """
    path = utils.handle_mesh_name(mesh_name=mesh_name)

    # Validate the input to ensure the geometry is valid
    if not 0 < flat_distance < inner_radius:
        raise ValueError("The 'flat_distance' must be positive and smaller than 'inner_radius'.")

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # Use the filename as the model name
    gmsh.model.add(path.stem)

    # --- 1. Define the Geometry ---

    # Create the main outer cylinder
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)

    # Create the base for the inner cutting tool
    inner_tool_base = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)

    # Create two large boxes that will be used to slice the inner cylinder.
    # We make them slightly larger than the inner cylinder to ensure a clean cut.
    box_dx = outer_radius  # A sufficiently large dimension
    box_dy = 2 * outer_radius
    box_dz = height

    # Box to cut the positive-x side
    cutter1 = gmsh.model.occ.addBox(flat_distance, -outer_radius, 0, box_dx, box_dy, box_dz)
    # Box to cut the negative-x side
    cutter2 = gmsh.model.occ.addBox(
        -flat_distance - box_dx, -outer_radius, 0, box_dx, box_dy, box_dz
    )

    # --- 2. Perform Boolean Operations ---

    # Cut the inner cylinder with the two boxes to create the final cutting tool
    # The result of the cut is the first element in the returned 'out' list
    tool, _ = gmsh.model.occ.cut(
        [(3, inner_tool_base)], [(3, cutter1), (3, cutter2)], removeObject=True, removeTool=True
    )

    # Cut the final tool from the outer cylinder to get the desired shell
    # The 'shell_map' is crucial as it tells us which new surfaces correspond
    # to the boundaries of the original shapes.
    shell, shell_map = gmsh.model.occ.cut(
        [(3, outer_cylinder)], tool, removeObject=True, removeTool=True
    )

    gmsh.model.occ.synchronize()

    # --- 3. Assign Physical Groups ---

    # The final volume
    gmsh.model.addPhysicalGroup(3, [v[1] for v in shell], tag=101, name="VOLUME")

    surfaces = gmsh.model.occ.getEntities(dim=2)

    gmsh.model.addPhysicalGroup(
        dim=surfaces[0][0],
        tags=[surfaces[0][1]],
        tag=1,
        name="INSIDE_CURVED1",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[1][0],
        tags=[surfaces[1][1]],
        tag=2,
        name="INSIDE_FLAT1",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[2][0],
        tags=[surfaces[2][1]],
        tag=3,
        name="INSIDE_FLAT2",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[3][0],
        tags=[surfaces[3][1]],
        tag=4,
        name="INSIDE_CURVED2",
    )

    gmsh.model.addPhysicalGroup(
        dim=surfaces[4][0],
        tags=[surfaces[4][1]],
        tag=5,
        name="OUTSIDE",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[5][0],
        tags=[surfaces[5][1]],
        tag=6,
        name="TOP",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[6][0],
        tags=[surfaces[6][1]],
        tag=7,
        name="BOTTOM",
    )
    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in shell], tag=8, name="VOLUME")

    # --- 4. Generate Mesh and Save ---

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    print(f"Cylinder mesh with flat inner sides generated and saved to '{path}'")
    return path


def cylinder_D_shaped(
    mesh_name: str | Path = "cylinder-d-shaped.msh",
    inner_radius: float = 10.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 5.0,
    outer_flat_face_distance: float = 15.0,
    char_length: float = 10.0,
    verbose: bool = True,
):
    """Create a thick D-shaped cylindrical shell mesh using GMSH.

    The D-shape has a single flat surface on the inside.

    Parameters
    ----------
    mesh_name : str or Path, optional
        Name of the mesh file, by default "cylinder-d-shaped.msh".
    inner_radius : float
        The radius of the curved part of the inner surface, default is 10.0.
    outer_radius : float
        Outer radius of the cylinder, default is 20.0.
    height : float
        Height of the cylinder, default is 40.0.
    inner_flat_face_distance : float
        The distance of the inner flat face from the center (along the x-axis).
        This value must be less than inner_radius. Default is 5.0.
    outer_flat_face_distance : float
        The distance of the outer flat face from the center (along the x-axis).
        This value must be less than outer_radius. Default is 15.0.
    char_length : float
        Characteristic length of the mesh, default is 10.0.
    verbose : bool
        If True, GMSH will print messages to the console, default is True.
    """
    if inner_flat_face_distance >= inner_radius:
        raise ValueError("The 'inner_flat_face_distance' must be less than the 'inner_radius'.")
    if outer_flat_face_distance >= outer_radius:
        raise ValueError("The 'outer_flat_face_distance' must be less than the 'outer_radius'.")

    path = utils.handle_mesh_name(mesh_name=mesh_name)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # --- Geometry Creation ---

    # 1. Create the outer D-shaped cylinder.
    #    Start with a full cylinder, then cut it with a box.
    outer_cylinder_full = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)
    outer_cutter_box = gmsh.model.occ.addBox(
        outer_flat_face_distance,
        -outer_radius,
        -height * 0.1,
        outer_radius,
        2 * outer_radius,
        height * 1.2,
    )
    outer_d_shape, _ = gmsh.model.occ.cut(
        [(3, outer_cylinder_full)], [(3, outer_cutter_box)], removeTool=True
    )

    # 2. Create the inner D-shaped volume that will be subtracted.
    inner_cylinder_tool = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)
    inner_cutter_box = gmsh.model.occ.addBox(
        inner_flat_face_distance,
        -inner_radius,
        -height * 0.1,
        inner_radius,
        2 * inner_radius,
        height * 1.2,
    )
    inner_d_shape_tool, _ = gmsh.model.occ.cut(
        [(3, inner_cylinder_tool)], [(3, inner_cutter_box)], removeTool=True
    )

    # 3. Subtract the inner D-shaped volume from the outer D-shaped volume.
    final_shell, _ = gmsh.model.occ.cut(outer_d_shape, inner_d_shape_tool, removeTool=True)

    gmsh.model.occ.synchronize()

    # --- Physical Group Assignment ---
    # This section identifies each surface by its geometric properties (location/shape)
    # which is more reliable than assuming a fixed order.
    surfaces = gmsh.model.occ.getEntities(dim=2)

    gmsh.model.addPhysicalGroup(
        dim=surfaces[0][0],
        tags=[surfaces[0][1]],
        tag=1,
        name="INSIDE_CURVED",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[1][0],
        tags=[surfaces[1][1]],
        tag=2,
        name="INSIDE_FLAT",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[2][0],
        tags=[surfaces[2][1]],
        tag=3,
        name="OUTSIDE_CURVED",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[3][0],
        tags=[surfaces[3][1]],
        tag=4,
        name="TOP",
    )

    gmsh.model.addPhysicalGroup(
        dim=surfaces[4][0],
        tags=[surfaces[4][1]],
        tag=5,
        name="OUTSIDE_FLAT",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[5][0],
        tags=[surfaces[5][1]],
        tag=6,
        name="BOTTOM",
    )

    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in final_shell], tag=7, name="VOLUME")

    # --- Meshing ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")  # Optional: optimize mesh quality

    # --- Save and Finalize ---
    gmsh.write(path.as_posix())
    gmsh.finalize()

    print(f"D-shaped cylindrical shell mesh generated and saved to {path.as_posix()}")
    return path
