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
    outer_radius: float = 15.0,
    height: float = 40.0,
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    char_length: float = 5.0,
    verbose: bool = True,
):
    """Create a thick cylindrical shell (hollow cylinder) mesh using GMSH,
    with optional flat caps on the top and bottom.

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
    floor_thickness : float
        Thickness of the bottom cap, default is 10.0
    roof_thickness : float
        Thickness of the top cap, default is 10.0
    char_length : float
        Characteristic length of the mesh, default is 10.0
    verbose : bool
        If True, GMSH will print messages to the console, default is True
    """
    path = utils.handle_mesh_name(mesh_name=mesh_name)

    logger.info("--- Generating Cylinder ---")
    logger.info(
        f"Parameters: inner_radius={inner_radius}, outer_radius={outer_radius}, height={height}",
    )
    logger.info(
        f"floor_thickness={floor_thickness}, "
        f"roof_thickness={roof_thickness}, "
        f"char_length={char_length}"
    )

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    inner_z_start = floor_thickness
    inner_height = height - floor_thickness - roof_thickness

    logger.info(
        f"Inner cavity geometry: z_start={inner_z_start}, z_end={inner_z_start + inner_height}"
    )

    if inner_height < 0:
        raise ValueError(
            "Floor and roof thickness combined must be less than the total cylinder height."
        )

    # 1. Create the solid exterior cylinder
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)

    # 2. Create the inner cavity to subtract
    if inner_height > 0 and inner_radius > 0:
        inner_cylinder = gmsh.model.occ.addCylinder(
            0, 0, inner_z_start, 0, 0, inner_height, inner_radius
        )
        shell, id = gmsh.model.occ.cut(
            [(3, outer_cylinder)], [(3, inner_cylinder)], removeTool=True
        )
    else:
        # If no inner cavity, just keep the solid cylinder
        shell = [(3, outer_cylinder)]

    gmsh.model.occ.synchronize()

    # --- Robust Physical Group Assignment ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    # Keeping the original names expected by standard FEniCS workflows
    groups: dict[str, list[int]] = {"INSIDE": [], "OUTSIDE": [], "TOP": [], "BOTTOM": []}

    tol_z = height * 1e-3
    threshold_radius = (inner_radius + outer_radius) / 2.0

    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min, z_max = bb[2], bb[5]
        z_center = (z_min + z_max) / 2.0

        # Check if the surface is a flat horizontal cap
        if abs(z_max - z_min) < tol_z:
            if abs(z_center - height) < tol_z:
                groups["TOP"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> TOP")
            elif abs(z_center - 0.0) < tol_z:
                groups["BOTTOM"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> BOTTOM")
            else:
                # Any other flat cap belongs to the inner cavity bounds (floor/ceiling)
                groups["INSIDE"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> INSIDE (Inner Cap)")
        else:
            # Curved vertical walls
            max_extent_x = max(abs(bb[0]), abs(bb[3]))
            max_extent_y = max(abs(bb[1]), abs(bb[4]))
            max_radial_extent = max(max_extent_x, max_extent_y)

            if max_radial_extent < threshold_radius:
                groups["INSIDE"].append(tag)
                logger.info(
                    f"Surface {tag} (Radius={max_radial_extent:.2f}) mapped -> INSIDE (Curved)"
                )
            else:
                groups["OUTSIDE"].append(tag)
                logger.info(
                    f"Surface {tag} (Radius={max_radial_extent:.2f}) mapped -> OUTSIDE (Curved)"
                )

    # Assign mapped groups to standard fixed Tags
    fixed_tags = {"INSIDE": 1, "OUTSIDE": 2, "TOP": 3, "BOTTOM": 4}
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, tag=fixed_tags[name], name=name)

    # Finalize Volume
    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in shell], tag=5, name="VOLUME")

    # Meshing configuration
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    # Generate & optimize mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    if verbose:
        logger.info(f"Closed cylindrical shell mesh generated and saved to {path.as_posix()}")

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
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    char_length: float = 10.0,
    verbose: bool = True,
):
    """Create a thick D-shaped cylindrical shell mesh using GMSH,
    optionally closed with a roof and floor.

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
    floor_thickness : float
        Thickness of the bottom cap (floor), default is 0.0 (open).
    roof_thickness : float
        Thickness of the top cap (roof), default is 0.0 (open).
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

    logger.info("--- Generating D-Shaped Cylinder ---")
    logger.info(
        f"Parameters: inner_radius={inner_radius}, outer_radius={outer_radius}, height={height}"
    )
    logger.info(
        f"floor_thickness={floor_thickness}, "
        f"roof_thickness={roof_thickness}, char_length={char_length}"
    )

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # Calculate inner cavity dimensions
    inner_z_start = floor_thickness
    inner_height = height - floor_thickness - roof_thickness

    if inner_height < 0:
        raise ValueError(
            "Floor and roof thickness combined must be less than the total cylinder height."
        )

    # 1. Create the outer D-shaped cylinder
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

    # 2. Create the inner D-shaped volume that will be subtracted
    if inner_height > 0 and inner_radius > 0:
        inner_cylinder_tool = gmsh.model.occ.addCylinder(
            0, 0, inner_z_start, 0, 0, inner_height, inner_radius
        )
        inner_cutter_box = gmsh.model.occ.addBox(
            inner_flat_face_distance,
            -inner_radius,
            inner_z_start - height * 0.1,
            inner_radius,
            2 * inner_radius,
            inner_height + height * 0.2,
        )
        inner_d_shape_tool, _ = gmsh.model.occ.cut(
            [(3, inner_cylinder_tool)], [(3, inner_cutter_box)], removeTool=True
        )

        # 3. Subtract the inner cavity from the outer shape
        final_shell, _ = gmsh.model.occ.cut(outer_d_shape, inner_d_shape_tool, removeTool=True)
    else:
        # If the cavity has negative or zero dimensions, treat as solid
        final_shell = outer_d_shape

    gmsh.model.occ.synchronize()

    # --- Robust Physical Group Assignment ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    groups: dict[str, list[int]] = {
        "INSIDE_CURVED": [],
        "INSIDE_FLAT": [],
        "OUTSIDE_CURVED": [],
        "OUTSIDE_FLAT": [],
        "TOP": [],
        "BOTTOM": [],
        "INSIDE_TOP": [],
        "INSIDE_BOTTOM": [],
    }

    tol_z = height * 1e-3
    threshold_radius = (inner_radius + outer_radius) / 2.0

    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min, z_max = bb[2], bb[5]
        z_center = (z_min + z_max) / 2.0

        # Check for horizontal caps
        if abs(z_max - z_min) < tol_z:
            if abs(z_center - height) < tol_z:
                groups["TOP"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> TOP")
            elif abs(z_center - 0.0) < tol_z:
                groups["BOTTOM"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> BOTTOM")
            elif abs(z_center - (height - roof_thickness)) < tol_z:
                groups["INSIDE_TOP"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> INSIDE_TOP")
            elif abs(z_center - floor_thickness) < tol_z:
                groups["INSIDE_BOTTOM"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> INSIDE_BOTTOM")
            else:
                # Fallback for minor floating point discrepancies
                if z_center > height / 2.0:
                    groups["INSIDE_TOP"].append(tag)
                else:
                    groups["INSIDE_BOTTOM"].append(tag)
        else:
            # Check for vertical walls
            x_min, x_max = bb[0], bb[3]
            x_span = abs(x_max - x_min)

            # If span in X is nearly 0, it's a flat wall
            if x_span < tol_z:
                x_center = (x_min + x_max) / 2.0
                if abs(x_center - inner_flat_face_distance) < tol_z:
                    groups["INSIDE_FLAT"].append(tag)
                    logger.info(f"Surface {tag} (X={x_center:.2f}) mapped -> INSIDE_FLAT")
                else:
                    groups["OUTSIDE_FLAT"].append(tag)
                    logger.info(f"Surface {tag} (X={x_center:.2f}) mapped -> OUTSIDE_FLAT")
            else:
                # It's a curved wall
                max_extent_x = max(abs(bb[0]), abs(bb[3]))
                max_extent_y = max(abs(bb[1]), abs(bb[4]))
                max_radial_extent = max(max_extent_x, max_extent_y)

                if max_radial_extent < threshold_radius:
                    groups["INSIDE_CURVED"].append(tag)
                    logger.info(
                        f"Surface {tag} (Radius={max_radial_extent:.2f}) mapped -> INSIDE_CURVED"
                    )
                else:
                    groups["OUTSIDE_CURVED"].append(tag)
                    logger.info(
                        f"Surface {tag} (Radius={max_radial_extent:.2f}) mapped -> OUTSIDE_CURVED"
                    )

    # Add dynamically mapped physical groups
    tag_idx = 1
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, tag=tag_idx, name=name)
            tag_idx += 1

    # Add final Volume group
    gmsh.model.addPhysicalGroup(dim=3, tags=[t[1] for t in final_shell], tag=tag_idx, name="VOLUME")

    # Meshing configuration
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    # Generate & optimize mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    if verbose:
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
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    fillet_radius: float | None = None,
    char_length: float = 2.0,
    verbose: bool = True,
):
    """
    Create a unified cylindrical shell mesh (Racetrack or D-Shaped) with filleted corners.
    Can be optionally closed with a floor and roof.

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
    outer_flat_face_distance : float
        Distance from center to the outer flat wall (Positive X).
    floor_thickness : float
        Thickness of the bottom solid floor. Default is 0.0 (open).
    roof_thickness : float | None
        Thickness of the top solid roof. Default is None (mirrors floor).
    fillet_radius : Optional[float]
        Radius of the corners between flat and curved walls.
    char_length : float
        Target mesh size.
    """

    if inner_flat_face_distance >= inner_radius:
        raise ValueError("inner_flat_face_distance must be < inner_radius")
    if outer_flat_face_distance >= outer_radius:
        raise ValueError("outer_flat_face_distance must be < outer_radius")

    cuts = {"racetrack": (True, True), "d_shaped": (True, False)}
    cut_pos_x, cut_neg_x = cuts[mode]

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

    path = utils.handle_mesh_name(mesh_name=mesh_name)

    if verbose:
        logger.info(f"--- Generating {mode.title()} Cut Cylinder ---")
        logger.info(f"floor_thickness={floor_thickness}, roof_thickness={roof_thickness}")

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    inner_z_start = floor_thickness
    inner_height = height - floor_thickness - roof_thickness

    if inner_height < 0:
        raise ValueError(
            "Floor and roof thickness combined must be less than the total cylinder height."
        )

    # 1. Build Solid Outer Shape
    outer_loop = _build_profile(
        outer_radius, outer_flat_face_distance, fillet_radius, cut_pos_x, cut_neg_x
    )
    outer_surf = gmsh.model.occ.addPlaneSurface([outer_loop])
    outer_extrude = gmsh.model.occ.extrude([(2, outer_surf)], 0, 0, height)
    outer_vol = next(tag for dim, tag in outer_extrude if dim == 3)

    # 2. Build Solid Inner Tool and Cut
    if inner_height > 0 and inner_radius > 0:
        inner_loop = _build_profile(
            inner_radius, inner_flat_face_distance, fillet_radius, cut_pos_x, cut_neg_x
        )
        inner_surf = gmsh.model.occ.addPlaneSurface([inner_loop])

        # Move the inner surface up to the floor thickness before extruding
        gmsh.model.occ.translate([(2, inner_surf)], 0, 0, inner_z_start)
        inner_extrude = gmsh.model.occ.extrude([(2, inner_surf)], 0, 0, inner_height)
        inner_vol = next(tag for dim, tag in inner_extrude if dim == 3)

        final_shell, _ = gmsh.model.occ.cut([(3, outer_vol)], [(3, inner_vol)], removeTool=True)
    else:
        final_shell = [(3, outer_vol)]

    gmsh.model.occ.synchronize()

    # --- Physical Groups Assignment ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    groups: dict[str, list[int]] = {
        "INSIDE_FLAT": [],
        "INSIDE_CURVED": [],
        "OUTSIDE_FLAT": [],
        "OUTSIDE_CURVED": [],
        "TOP": [],
        "BOTTOM": [],
        "INSIDE_TOP": [],
        "INSIDE_BOTTOM": [],
    }

    mid_radius = (inner_radius + outer_radius) / 2.0
    tol = min(inner_radius, height) * 1e-3
    tol_z = height * 1e-3

    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min, z_max = bb[2], bb[5]
        z_center = (z_min + z_max) / 2.0
        x_min, x_max = bb[0], bb[3]
        x_span = abs(x_max - x_min)

        # Check for Horizontal Caps
        if abs(z_max - z_min) < tol_z:
            if abs(z_center - height) < tol_z:
                groups["TOP"].append(tag)
            elif abs(z_center - 0.0) < tol_z:
                groups["BOTTOM"].append(tag)
            elif abs(z_center - (height - roof_thickness)) < tol_z:
                groups["INSIDE_TOP"].append(tag)
            elif abs(z_center - floor_thickness) < tol_z:
                groups["INSIDE_BOTTOM"].append(tag)
            else:
                if z_center > height / 2.0:
                    groups["INSIDE_TOP"].append(tag)
                else:
                    groups["INSIDE_BOTTOM"].append(tag)
            continue

        # Check for Vertical Walls
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x = com[0]

        # Flat faces have very small X-span
        if x_span < tol:
            if abs(abs(x) - inner_flat_face_distance) < tol:
                groups["INSIDE_FLAT"].append(tag)
            elif abs(abs(x) - outer_flat_face_distance) < tol:
                groups["OUTSIDE_FLAT"].append(tag)
        else:
            # Curved faces (including fillets)
            r = math.hypot(com[0], com[1])
            if r < mid_radius:
                groups["INSIDE_CURVED"].append(tag)
            else:
                groups["OUTSIDE_CURVED"].append(tag)

    # Apply physical groups dynamically
    tag_idx = 1
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, tag=tag_idx, name=name)
            tag_idx += 1

    gmsh.model.addPhysicalGroup(3, [t[1] for t in final_shell], tag=tag_idx, name="VOLUME")

    # Meshing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    if verbose:
        logger.info(f"Generated {mode} mesh with caps: {path}")

    return path


def cylinder_elliptical(
    mesh_name: str | Path = "",
    inner_radius_x: float = 10.0,
    outer_radius_x: float = 20.0,
    inner_radius_y: float | None = None,  # If None, defaults to inner_radius_x (Circle)
    outer_radius_y: float | None = None,  # If None, defaults to outer_radius_x (Circle)
    height: float = 40.0,
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    char_length: float = 10.0,
    verbose: bool = True,
):
    """
    Create a thick cylindrical shell which can be circular or elliptical.

    Parameters
    ----------
    inner_radius_x : float
        Inner radius along the X-axis.
    outer_radius_x : float
        Outer radius along the X-axis.
    inner_radius_y : float | None
        Inner radius along the Y-axis. If None, circle is created.
    outer_radius_y : float | None
        Outer radius along the Y-axis. If None, circle is created.
    height : float
        Height of the cylinder.
    floor_thickness : float
        Thickness of the bottom cap (floor), default is 0.0 (open).
    roof_thickness : float
        Thickness of the top cap (roof), default is 0.0 (open).
    char_length : float
        Characteristic length of the mesh, default is 10.0.
    verbose : bool
        If True, GMSH will print messages to the console, default is True.
    """

    # --- 1. Handle Defaults for Ellipse vs Circle ---
    # If Y-radii are missing, we assume a standard circular cylinder
    if inner_radius_y is None:
        inner_radius_y = inner_radius_x
    if outer_radius_y is None:
        outer_radius_y = outer_radius_x

    path = utils.handle_mesh_name(mesh_name=mesh_name)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    logger.info("--- Generating Elliptical Cylinder ---")
    logger.info(
        f"Inner radii: ({inner_radius_x}, {inner_radius_y}), "
        f"Outer radii: ({outer_radius_x}, {outer_radius_y})"
    )
    logger.info(
        f"height={height}, floor_thickness={floor_thickness}, roof_thickness={roof_thickness}"
    )
    # Calculate inner cavity dimensions
    inner_z_start = floor_thickness
    inner_height = height - floor_thickness - roof_thickness

    if inner_height < 0:
        raise ValueError(
            "Floor and roof thickness combined must be less than the total cylinder height."
        )

    # --- 2. Create Geometry using 3D Boolean Cut ---

    # Create the Outer Solid Volume (extrude a 2D disk)
    outer_disk = gmsh.model.occ.addDisk(0, 0, 0, outer_radius_x, outer_radius_y)
    outer_extrude = gmsh.model.occ.extrude([(2, outer_disk)], 0, 0, height)
    outer_vol = next(tag for dim, tag in outer_extrude if dim == 3)

    # Create the Inner Solid Volume and Subtract
    if inner_height > 0 and max(inner_radius_x, inner_radius_y) > 0:
        inner_disk = gmsh.model.occ.addDisk(0, 0, 0, inner_radius_x, inner_radius_y)
        # Move the 2D sketch up to the floor thickness before extruding
        gmsh.model.occ.translate([(2, inner_disk)], 0, 0, inner_z_start)
        inner_extrude = gmsh.model.occ.extrude([(2, inner_disk)], 0, 0, inner_height)
        inner_vol = next(tag for dim, tag in inner_extrude if dim == 3)

        # Cut the cavity
        shell, _ = gmsh.model.occ.cut([(3, outer_vol)], [(3, inner_vol)], removeTool=True)
    else:
        shell = [(3, outer_vol)]

    gmsh.model.occ.synchronize()

    # --- 3. Robust Physical Group Assignment ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    groups: dict[str, list[int]] = {
        "INSIDE": [],
        "OUTSIDE": [],
        "TOP": [],
        "BOTTOM": [],
        "INSIDE_TOP": [],
        "INSIDE_BOTTOM": [],
    }

    major_inner = max(inner_radius_x, inner_radius_y)
    major_outer = max(outer_radius_x, outer_radius_y)
    threshold_radius = (major_inner + major_outer) / 2.0

    tol_z = height * 1e-3

    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min, z_max = bb[2], bb[5]
        z_center = (z_min + z_max) / 2.0

        # Check for Horizontal Caps
        if abs(z_max - z_min) < tol_z:
            if abs(z_center - height) < tol_z:
                groups["TOP"].append(tag)
                logger.debug(f"Surface {tag} (Z={z_center:.2f}) mapped -> TOP")
            elif abs(z_center - 0.0) < tol_z:
                groups["BOTTOM"].append(tag)
                logger.debug(f"Surface {tag} (Z={z_center:.2f}) mapped -> BOTTOM")
            elif abs(z_center - (height - roof_thickness)) < tol_z:
                groups["INSIDE_TOP"].append(tag)
                logger.debug(f"Surface {tag} (Z={z_center:.2f}) mapped -> INSIDE_TOP")
            elif abs(z_center - floor_thickness) < tol_z:
                groups["INSIDE_BOTTOM"].append(tag)
                logger.debug(f"Surface {tag} (Z={z_center:.2f}) mapped -> INSIDE_BOTTOM")
            else:
                # Fallback
                if z_center > height / 2.0:
                    groups["INSIDE_TOP"].append(tag)
                else:
                    groups["INSIDE_BOTTOM"].append(tag)
        else:
            # Vertical Walls (Inner vs Outer)
            max_extent_x = max(abs(bb[0]), abs(bb[3]))
            max_extent_y = max(abs(bb[1]), abs(bb[4]))
            max_radial_extent = max(max_extent_x, max_extent_y)

            if max_radial_extent < threshold_radius:
                groups["INSIDE"].append(tag)
                logger.info(f"Surface {tag} (MaxExt={max_radial_extent:.2f}) mapped -> INSIDE")
            else:
                groups["OUTSIDE"].append(tag)
                logger.info(f"Surface {tag} (MaxExt={max_radial_extent:.2f}) mapped -> OUTSIDE")

    # Add dynamically mapped physical groups
    tag_idx = 1
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, tag=tag_idx, name=name)
            tag_idx += 1

    gmsh.model.addPhysicalGroup(3, [t[1] for t in shell], tag=tag_idx, name="VOLUME")

    # --- 4. Meshing ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    logger.info(f"Elliptical shell mesh generated: {path.as_posix()}")

    return path
