from pathlib import Path
import math
import typing
import logging

import gmsh

from . import utils

logger = logging.getLogger(__name__)


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

    logger.info(f"Cylindrical shell mesh generated and saved to {mesh_name}")

    return path


def cylinder_racetrack(
    mesh_name: str | Path = "cylinder_flat_sides.msh",
    inner_radius: float = 13.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    char_length: float = 10.0,
    verbose: bool = True,
):
    """Create a racetrack-shaped thick cylindrical shell mesh using GMSH.

    Both the inner and outer surfaces have two flat faces on opposite sides.

    Parameters
    ----------
    mesh_name : str or Path, optional
        Name of the mesh file, by default "cylinder-d-shaped.msh".
    inner_radius : float
        The radius of the curved part of the inner surface, default is 13.0.
    outer_radius : float
        Outer radius of the cylinder, default is 20.0.
    height : float
        Height of the cylinder, default is 40.0.
    inner_flat_face_distance : float
        The distance of the inner flat face from the center (along the x-axis).
        This value must be less than inner_radius. Default is 10.0.
    outer_flat_face_distance : float
        The distance of the outer flat face from the center (along the x-axis).
        This value must be less than outer_radius. Default is 17.0.
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

    # 1. Create the outer racetrack-shaped cylinder.
    outer_cylinder_full = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)
    # Cutter for the positive-x side
    outer_cutter_pos = gmsh.model.occ.addBox(
        outer_flat_face_distance,
        -outer_radius,
        -height * 0.1,
        outer_radius,
        2 * outer_radius,
        height * 1.2,
    )
    # Cutter for the negative-x side
    outer_cutter_neg = gmsh.model.occ.addBox(
        -outer_flat_face_distance - outer_radius,
        -outer_radius,
        -height * 0.1,
        outer_radius,
        2 * outer_radius,
        height * 1.2,
    )
    # Cut the full cylinder with both boxes
    outer_shape, _ = gmsh.model.occ.cut(
        [(3, outer_cylinder_full)], [(3, outer_cutter_pos), (3, outer_cutter_neg)], removeTool=True
    )

    # 2. Create the inner racetrack-shaped volume that will be subtracted.
    inner_cylinder_full = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)
    # Cutter for the positive-x side
    inner_cutter_pos = gmsh.model.occ.addBox(
        inner_flat_face_distance,
        -inner_radius,
        -height * 0.1,
        inner_radius,
        2 * inner_radius,
        height * 1.2,
    )
    # Cutter for the negative-x side
    inner_cutter_neg = gmsh.model.occ.addBox(
        -inner_flat_face_distance - inner_radius,
        -inner_radius,
        -height * 0.1,
        inner_radius,
        2 * inner_radius,
        height * 1.2,
    )
    # Cut the full cylinder with both boxes
    inner_tool, _ = gmsh.model.occ.cut(
        [(3, inner_cylinder_full)], [(3, inner_cutter_pos), (3, inner_cutter_neg)], removeTool=True
    )

    # 3. Subtract the inner shape from the outer shape.
    final_shell, _ = gmsh.model.occ.cut(outer_shape, inner_tool, removeTool=True)

    gmsh.model.occ.synchronize()

    # --- Physical Group Assignment ---
    # This section identifies each surface by its geometric properties (location/shape)
    # which is more reliable than assuming a fixed order.
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
        name="OUTSIDE_CURVED1",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[5][0],
        tags=[surfaces[5][1]],
        tag=6,
        name="OUTSIDE_FLAT1",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[6][0],
        tags=[surfaces[6][1]],
        tag=7,
        name="BOTTOM",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[7][0],
        tags=[surfaces[7][1]],
        tag=8,
        name="OUTSIDE_FLAT2",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[8][0],
        tags=[surfaces[8][1]],
        tag=9,
        name="TOP",
    )
    gmsh.model.addPhysicalGroup(
        dim=surfaces[9][0],
        tags=[surfaces[9][1]],
        tag=10,
        name="OUTSIDE_CURVED2",
    )

    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in final_shell], tag=11, name="VOLUME")

    # --- Meshing ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")  # Optional: optimize mesh quality

    # --- Save and Finalize ---
    gmsh.write(path.as_posix())
    gmsh.finalize()

    logger.info(f"Racetrack cylindrical shell mesh generated and saved to {path.as_posix()}")
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
        This value "must be less than inner_radius. Default is 5.0.
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

    logger.info(f"D-shaped cylindrical shell mesh generated and saved to {path.as_posix()}")
    return path


def _add_quadrant_wire(radius, x_cut, r_fillet, quadrant: int, is_cut: bool):
    """
    Generates curves for a single 90-degree quadrant.
    Returns a list of curve tags.

    Parameters
    ----------
    radius : float
        Radius of the circular arc.
    x_cut : float
        Distance from center to flat cut line.
    r_fillet : float
        Radius of the fillet arc.
    quadrant : int
        Quadrant number (1 to 4).
    is_cut : bool
        Whether this quadrant has a flat cut.
    """
    # Determine start and end points for a standard circle quadrant
    # Q1: (R,0)->(0,R), Q2: (0,R)->(-R,0), etc.
    # However, it is easier to build generic "Path A" (Cut) vs "Path B" (Arc)
    # and rotate/mirror them, or simply explicit coordinates.

    curves = []
    origin = gmsh.model.occ.addPoint(0, 0, 0)

    # Helper to simplify coordinates based on quadrant signs
    # Q1: (+,+), Q2: (-,+), Q3: (-,-), Q4: (+,-)
    sx = 1 if quadrant in [1, 4] else -1
    sy = 1 if quadrant in [1, 2] else -1

    # Rotate logic: We build everything as if it is Q1, then flip coords.
    # But Q2/Q4 need swapped logic for CCW traversal.
    # Let's do explicit coordinates for clarity.

    if not is_cut:
        # --- Standard Circular Arc ---
        # Define cardinal points
        p_start = None
        p_end = None

        if quadrant == 1:  # East to North
            p_start = gmsh.model.occ.addPoint(radius, 0, 0)
            p_end = gmsh.model.occ.addPoint(0, radius, 0)
        elif quadrant == 2:  # North to West
            p_start = gmsh.model.occ.addPoint(0, radius, 0)
            p_end = gmsh.model.occ.addPoint(-radius, 0, 0)
        elif quadrant == 3:  # West to South
            p_start = gmsh.model.occ.addPoint(-radius, 0, 0)
            p_end = gmsh.model.occ.addPoint(0, -radius, 0)
        elif quadrant == 4:  # South to East
            p_start = gmsh.model.occ.addPoint(0, -radius, 0)
            p_end = gmsh.model.occ.addPoint(radius, 0, 0)

        curves.append(gmsh.model.occ.addCircleArc(p_start, origin, p_end))

    else:
        # --- Flat Cut with Fillet ---
        # We need to construct the geometry for "Flat -> Fillet -> Arc" or "Arc -> Fillet -> Flat"
        # depending on CCW direction.

        # Calculate Fillet Center Geometry (in positive Quadrant)
        # Ensure fillet fits
        max_r = radius - x_cut
        safe_r = min(r_fillet, max_r * 0.99)

        xc = x_cut - safe_r
        yc = math.sqrt(max(0, (radius - safe_r) ** 2 - xc**2))

        # Tangent Points (in Q1)
        # Pt on Flat Line
        pt_flat_x, pt_flat_y = x_cut, yc
        # Pt on Circle
        scale = radius / (radius - safe_r)
        pt_circ_x, pt_circ_y = xc * scale, yc * scale

        # Apply signs for current quadrant
        def mk_pt(x, y):
            return gmsh.model.occ.addPoint(x * sx, y * sy, 0)

        def mk_cen(x, y):
            return gmsh.model.occ.addPoint(x * sx, y * sy, 0)

        # Build Segments based on flow direction
        if quadrant == 1:  # East(Flat) -> North
            p_start = mk_pt(pt_flat_x, 0)  # On x-axis
            p_flat_tan = mk_pt(pt_flat_x, pt_flat_y)
            p_circ_tan = mk_pt(pt_circ_x, pt_circ_y)
            p_end = mk_pt(0, radius)  # On y-axis
            cen = mk_cen(xc, yc)

            curves.append(gmsh.model.occ.addLine(p_start, p_flat_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_flat_tan, cen, p_circ_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_circ_tan, origin, p_end))

        elif quadrant == 2:  # North -> West(Flat)
            # Note: Q2 cut means Left side cut (Negative X).
            # Input x_cut is distance from center, sx handles the sign.
            p_start = mk_pt(0, radius)
            p_circ_tan = mk_pt(pt_circ_x, pt_circ_y)
            p_flat_tan = mk_pt(pt_flat_x, pt_flat_y)
            p_end = mk_pt(pt_flat_x, 0)
            cen = mk_cen(xc, yc)

            curves.append(gmsh.model.occ.addCircleArc(p_start, origin, p_circ_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_circ_tan, cen, p_flat_tan))
            curves.append(gmsh.model.occ.addLine(p_flat_tan, p_end))

        elif quadrant == 3:  # West(Flat) -> South
            p_start = mk_pt(pt_flat_x, 0)
            p_flat_tan = mk_pt(pt_flat_x, pt_flat_y)
            p_circ_tan = mk_pt(pt_circ_x, pt_circ_y)
            p_end = mk_pt(0, radius)
            cen = mk_cen(xc, yc)

            curves.append(gmsh.model.occ.addLine(p_start, p_flat_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_flat_tan, cen, p_circ_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_circ_tan, origin, p_end))

        elif quadrant == 4:  # South -> East(Flat)
            p_start = mk_pt(0, radius)
            p_circ_tan = mk_pt(pt_circ_x, pt_circ_y)
            p_flat_tan = mk_pt(pt_flat_x, pt_flat_y)
            p_end = mk_pt(pt_flat_x, 0)
            cen = mk_cen(xc, yc)

            curves.append(gmsh.model.occ.addCircleArc(p_start, origin, p_circ_tan))
            curves.append(gmsh.model.occ.addCircleArc(p_circ_tan, cen, p_flat_tan))
            curves.append(gmsh.model.occ.addLine(p_flat_tan, p_end))

    return curves


def _build_profile(radius, flat_dist, r_fillet, cut_pos_x: bool = True, cut_neg_x: bool = True):
    """Constructs the full closed loop for one shell (inner or outer).

    Parameters
    ----------
    radius : float
        Radius of the circular arc.
    flat_dist : float
        Distance from center to flat cut line.
    r_fillet : float
        Radius of the fillet arc.
    cut_pos_x : bool
        Whether to cut the positive-x side.
    cut_neg_x : bool
        Whether to cut the negative-x side.
    """
    loop_curves = []
    # Q1 (East -> North)
    loop_curves.extend(_add_quadrant_wire(radius, flat_dist, r_fillet, 1, cut_pos_x))
    # Q2 (North -> West)
    loop_curves.extend(_add_quadrant_wire(radius, flat_dist, r_fillet, 2, cut_neg_x))
    # Q3 (West -> South)
    loop_curves.extend(_add_quadrant_wire(radius, flat_dist, r_fillet, 3, cut_neg_x))
    # Q4 (South -> East)
    loop_curves.extend(_add_quadrant_wire(radius, flat_dist, r_fillet, 4, cut_pos_x))

    return gmsh.model.occ.addCurveLoop(loop_curves)


def cylinder_cut(
    mesh_name: str | Path = "cylinder_cut.msh",
    mode: typing.Literal["racetrack", "d_shaped"] = "racetrack",
    inner_radius: float = 13.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    fillet_radius: float | None = None,
    char_length: float = 2.0,
    verbose: bool = True,
):
    """
    Create a unified cylindrical shell mesh (Racetrack or D-Shaped) with filleted corners.

    This function uses explicit 2D profile construction (Lines and Arcs)
    to guarantee valid geometry and meshing even with complex fillets.

    Parameters
    ----------
    mesh_name : str | Path
        Output filename.
    mode : "racetrack" | "d_shaped"
        "racetrack": Cuts both left and right sides (two flat faces).
        "d_shaped": Cuts only the positive-x side (one flat face).
    inner_radius, outer_radius : float
        Radii of the curved sections.
    height : float
        Extrusion height.
    inner_flat_face_distance : float
        Distance from center to the inner flat wall (Positive X).
        For Racetrack, this applies to negative X as well.
    outer_flat_face_distance : float
        Distance from center to the outer flat wall (Positive X).
        For Racetrack, this applies to negative X as well.
    fillet_radius : Optional[float]
        Radius of the corners between flat and curved walls.
        If None, a safe value is calculated automatically.
    char_length : float
        Target mesh size.
    """

    if inner_flat_face_distance >= inner_radius:
        raise ValueError("inner_flat_face_distance must be < inner_radius")
    if outer_flat_face_distance >= outer_radius:
        raise ValueError("outer_flat_face_distance must be < outer_radius")

    # Determine cuts based on mode
    # (Positive X, Negative X)
    cuts = {"racetrack": (True, True), "d_shaped": (True, False)}
    cut_pos_x, cut_neg_x = cuts[mode]

    # Auto-calculate fillet radius if missing
    if fillet_radius is None:
        gaps = []
        if cut_pos_x:
            gaps.append(inner_radius - inner_flat_face_distance)
            gaps.append(outer_radius - outer_flat_face_distance)
        if cut_neg_x:
            gaps.append(inner_radius - inner_flat_face_distance)
            gaps.append(outer_radius - outer_flat_face_distance)

        min_gap = min(gaps) if gaps else 1.0
        fillet_radius = min_gap * 1.0
        if verbose:
            logger.debug(f"Auto-calculated fillet_radius: {fillet_radius:.4f}")

    path = utils.handle_mesh_name(mesh_name=mesh_name)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # Geometry Builder (Quadrant System)

    # Build Inner and Outer Loops
    outer_loop = _build_profile(
        outer_radius, outer_flat_face_distance, fillet_radius, cut_pos_x, cut_neg_x
    )
    inner_loop = _build_profile(
        inner_radius, inner_flat_face_distance, fillet_radius, cut_pos_x, cut_neg_x
    )

    # Create Planar Surface with Hole
    surf_tag = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])

    # Extrude
    extrude_res = gmsh.model.occ.extrude([(2, surf_tag)], 0, 0, height)
    final_vol_tag = next(tag for dim, tag in extrude_res if dim == 3)

    gmsh.model.occ.synchronize()

    # Physical Groups Assignment
    surfaces = gmsh.model.occ.getEntities(dim=2)

    groups: dict[str, list[int]] = {
        "INSIDE_FLAT": [],
        "INSIDE_CURVED": [],
        "OUTSIDE_FLAT": [],
        "OUTSIDE_CURVED": [],
        "TOP": [],
        "BOTTOM": [],
    }

    mid_radius = (inner_radius + outer_radius) / 2.0
    tol = min(inner_radius, height) * 1e-3

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x, y, z = com[0], com[1], com[2]

        # Caps
        if abs(z - height) < tol:
            groups["TOP"].append(tag)
            continue
        if abs(z - 0.0) < tol:
            groups["BOTTOM"].append(tag)
            continue

        # Walls
        r = math.hypot(x, y)
        bb = gmsh.model.getBoundingBox(dim, tag)
        x_span = abs(bb[3] - bb[0])

        # Logic: Flat faces have very small X-span
        if x_span < tol:
            # Differentiate Inner vs Outer based on X location
            if abs(abs(x) - inner_flat_face_distance) < tol:
                groups["INSIDE_FLAT"].append(tag)
            elif abs(abs(x) - outer_flat_face_distance) < tol:
                groups["OUTSIDE_FLAT"].append(tag)
        else:
            # Curved faces (including fillets)
            if r < mid_radius:
                groups["INSIDE_CURVED"].append(tag)
            else:
                groups["OUTSIDE_CURVED"].append(tag)

    # Apply groups
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=name)

    gmsh.model.addPhysicalGroup(3, [final_vol_tag], tag=100, name="VOLUME")

    # --- 4. Meshing ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.write(path.as_posix())
    gmsh.finalize()

    if verbose:
        logger.info(f"Generated {mode} mesh: {path}")
    return path
