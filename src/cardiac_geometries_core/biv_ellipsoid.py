import logging
import gmsh

import math
from pathlib import Path

from . import utils

logger = logging.getLogger(__name__)


def within_bounding_box(
    box1: tuple[float, float, float, float, float, float],
    box2: tuple[float, float, float, float, float, float],
) -> bool:
    """Check if box1 is within box2. Each box is defined by (xmin, ymin, zmin, xmax, ymax, zmax)."""
    return (
        box1[0] >= box2[0]
        and box1[1] >= box2[1]
        and box1[2] >= box2[2]
        and box1[3] <= box2[3]
        and box1[4] <= box2[4]
        and box1[5] <= box2[5]
    )


def biv_ellipsoid(
    mesh_name: str | Path = "",
    char_length: float = 0.4,  # cm
    base_cut_z: float = 1.0,
    box_size: float = 20.0,  # Size of the cutting box
    rv_wall_thickness: float = 0.4,  # cm
    lv_wall_thickness: float = 0.5,  # cm
    rv_offset_x: float = 1.0,
    lv_radius_x: float = 2.2,
    lv_radius_y: float = 2.2,
    lv_radius_z: float = 4.5,
    rv_radius_x: float = 3.2,
    rv_radius_y: float = 2.3,
    rv_radius_z: float = 4.3,
    verbose: bool = False,
):
    """Create an idealized BiV geometry

    Parameters
    ----------
    mesh_name : str | Path, optional
        Path to the mesh, by default ""
    char_length : float, optional
        Characteristic length for mesh generation, by default 0.4
    box_size : float, optional
        Size of the cutting box, by default 20.0
    base_cut_z : float, optional
        Z-coordinate at which to cut the base, by default 1.0
    rv_wall_thickness : float, optional
        Thickness of the right ventricular wall, by default 0.4
    lv_wall_thickness : float, optional
        Thickness of the left ventricular wall, by default 0.5
    rv_offset_x : float, optional
        X-offset of the right ventricle center from the origin, by default 1.0
    lv_radius_x : float, optional
        Radius of the left ventricle in the x-direction, by default 2.2
    lv_radius_y : float, optional
        Radius of the left ventricle in the y-direction, by default 2.2
    lv_radius_z : float, optional
        Radius of the left ventricle in the z-direction, by default 4.5
    rv_radius_x : float, optional
        Radius of the right ventricle in the x-direction, by default 3.2
    rv_radius_y : float, optional
        Radius of the right ventricle in the y-direction, by default 2.3
    rv_radius_z : float, optional
        Radius of the right ventricle in the z-direction, by default 4.3
    verbose : bool, optional
        Whether to print verbose output from gmsh, by default False

    Returns
    -------
    Path
        Path to the generated mesh file.

    Raises
    ------
    RuntimeError
        If mesh generation fails.
    """
    path = utils.handle_mesh_name(mesh_name=mesh_name)
    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("biv")

    rv_center = (rv_offset_x, 0, 0)
    lv_center = (0, 0, 0)

    occ = gmsh.model.occ

    lv_r_inner = (
        lv_radius_x - lv_wall_thickness,
        lv_radius_y - lv_wall_thickness,
        lv_radius_z - lv_wall_thickness,
    )
    lv_r_outer = (lv_radius_x, lv_radius_y, lv_radius_z)
    rv_r_inner = (
        rv_radius_x - rv_wall_thickness,
        rv_radius_y - rv_wall_thickness,
        rv_radius_z - rv_wall_thickness,
    )
    rv_r_outer = (
        rv_radius_x,
        rv_radius_y,
        rv_radius_z,
    )

    # --- 1. Create the Solid Outer Shell ---
    lv_outer = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate([(3, lv_outer)], *lv_center, *lv_r_outer)
    lv_outer_bounding_box = occ.getBoundingBox(3, lv_outer)

    rv_outer = occ.addSphere(rv_center[0], rv_center[1], rv_center[2], 1)
    occ.dilate([(3, rv_outer)], *rv_center, *rv_r_outer)
    rv_outer_bounding_box = occ.getBoundingBox(3, rv_outer)

    # Deep intersection fuse (very robust in OCC)
    outer_shell, _ = occ.fuse([(3, lv_outer)], [(3, rv_outer)], removeTool=True, removeObject=True)

    # --- 2. Create the Inner Cavities ---
    lv_inner = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate([(3, lv_inner)], *lv_center, *lv_r_inner)
    lv_inner_bounding_box = occ.getBoundingBox(3, lv_inner)

    rv_inner = occ.addSphere(rv_center[0], rv_center[1], rv_center[2], 1)
    occ.dilate([(3, rv_inner)], *rv_center, *rv_r_inner)
    rv_inner_bounding_box = occ.getBoundingBox(3, rv_inner)

    # To ensure the RV cavity doesn't carve into the LV wall (the septum),
    # we trim the RV cavity using an independent LV outer profile.
    lv_carver = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate([(3, lv_carver)], *lv_center, *lv_r_outer)

    rv_cavity, _ = occ.cut([(3, rv_inner)], [(3, lv_carver)], removeTool=True, removeObject=True)

    # --- 3. Hollow Out the Myocardium ---
    # Cut both cavities (LV inner and the trimmed RV cavity) from the fused outer shell
    myocardium, _ = occ.cut(
        outer_shell, [(3, lv_inner)] + rv_cavity, removeTool=True, removeObject=True
    )

    # --- 4. Truncate the Base ---
    trunc_box = occ.addBox(-box_size / 2, -box_size / 2, base_cut_z, box_size, box_size, box_size)
    final_model, _ = occ.cut(myocardium, [(3, trunc_box)], removeTool=True, removeObject=True)

    occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)

    labels: dict[str, list[int]] = {
        "LV_EPICARDIUM": [],
        "RV_EPICARDIUM": [],
        "LV_ENDOCARDIUM": [],
        "RV_ENDOCARDIUM": [],
        "BASE": [],
    }

    for s in surfaces:
        if s[1] == 9:
            # This is the final LV endocardial surface that we cut off
            continue

        center_of_mass = occ.getCenterOfMass(s[0], s[1])
        bounding_box = occ.getBoundingBox(s[0], s[1])
        logger.debug("\n ---------------------------- ")
        logger.debug(s)
        logger.debug(center_of_mass)
        logger.debug(bounding_box)

        if math.isclose(center_of_mass[2], base_cut_z):
            logger.debug("  -> Base surface")
            labels["BASE"].append(s[1])
            continue

        if within_bounding_box(bounding_box, lv_inner_bounding_box):
            labels["LV_ENDOCARDIUM"].append(s[1])
            logger.debug("  -> LV free wall endocardium")

        elif within_bounding_box(bounding_box, rv_inner_bounding_box):
            labels["RV_ENDOCARDIUM"].append(s[1])
            logger.debug("  -> RV free wall endocardium")

        elif within_bounding_box(bounding_box, lv_outer_bounding_box):
            if center_of_mass[0] > rv_center[0]:
                labels["RV_ENDOCARDIUM"].append(s[1])
                logger.debug("  -> RV septal endocardium")
            else:
                labels["LV_EPICARDIUM"].append(s[1])
                logger.debug("  -> LV epicardium")
        elif within_bounding_box(bounding_box, rv_outer_bounding_box):
            labels["RV_EPICARDIUM"].append(s[1])
            logger.debug("  -> RV epicardium")
        else:
            raise RuntimeError("Surface does not fit any known category")

    # Define Physical Groups for different surfaces
    epi = gmsh.model.addPhysicalGroup(2, labels["LV_EPICARDIUM"] + labels["RV_EPICARDIUM"])
    gmsh.model.setPhysicalName(2, epi, "EPI")
    base = gmsh.model.addPhysicalGroup(2, labels["BASE"])
    gmsh.model.setPhysicalName(2, base, "BASE")
    lv_endo_fw = gmsh.model.addPhysicalGroup(2, labels["LV_ENDOCARDIUM"])
    gmsh.model.setPhysicalName(2, lv_endo_fw, "LV")
    rv_endo_fw = gmsh.model.addPhysicalGroup(2, labels["RV_ENDOCARDIUM"])
    gmsh.model.setPhysicalName(2, rv_endo_fw, "RV")

    myocardium_group = gmsh.model.add_physical_group(dim=3, tags=[t[1] for t in final_model], tag=1)
    gmsh.model.setPhysicalName(3, myocardium_group, "Myocardium")

    # Set mesh options
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    # Generate the 3D volumetric mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(path.as_posix())

    # if "close" not in sys.argv:
    #     logger.debug("Opening Gmsh GUI. Close window to exit.")
    #     gmsh.fltk.run()

    # Final
    gmsh.finalize()
    return path
