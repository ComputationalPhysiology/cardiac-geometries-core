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
    base_cut_z: float = 2.5,
    box_size: float = 15.0,  # Size of the cutting box
    rv_wall_thickness: float = 0.4,  # cm
    lv_wall_thickness: float = 0.5,  # cm
    rv_offset_x: float = 3.0,
    lv_radius_x: float = 2.0,
    lv_radius_y: float = 1.8,
    lv_radius_z: float = 3.25,
    rv_radius_x: float = 1.9,
    rv_radius_y: float = 2.5,
    rv_radius_z: float = 3.0,
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
        Size of the cutting box, by default 15.0
    lv_radius_x : float, optional
        Radius of the left ventricle in the x-direction, by default 2.0
    lv_radius_y : float, optional
        Radius of the left ventricle in the y-direction, by default 1.8
    lv_radius_z : float, optional
        Radius of the left ventricle in the z-direction, by default 3.25
    rv_radius_x : float, optional
        Radius of the right ventricle in the x-direction, by default 1.9
    rv_radius_y : float, optional
        Radius of the right ventricle in the y-direction, by default 2.5
    rv_radius_z : float, optional
        Radius of the right ventricle in the z-direction, by default 3.0
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

    lv_outer_s = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate(
        [(3, lv_outer_s)],
        lv_center[0],
        lv_center[1],
        lv_center[2],
        lv_radius_x + lv_wall_thickness,
        lv_radius_y + lv_wall_thickness,
        lv_radius_z + lv_wall_thickness,
    )
    lv_outer_center_of_mass = occ.getCenterOfMass(3, lv_outer_s)
    lv_outer_bounding_box = occ.getBoundingBox(3, lv_outer_s)
    logger.debug(f"LV outer: \n{lv_outer_center_of_mass}, \n{lv_outer_bounding_box}")

    lv_inner_s = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate(
        [(3, lv_inner_s)],
        lv_center[0],
        lv_center[1],
        lv_center[2],
        lv_radius_x,
        lv_radius_y,
        lv_radius_z,
    )
    lv_inner_center_of_mass = gmsh.model.occ.getCenterOfMass(3, lv_inner_s)
    lv_inner_bounding_box = gmsh.model.occ.getBoundingBox(3, lv_inner_s)
    logger.debug(f"LV inner: \n{lv_inner_center_of_mass}, \n{lv_inner_bounding_box}")

    rv_outer_s = occ.addSphere(rv_center[0], rv_center[1], rv_center[2], 1)
    occ.dilate(
        [(3, rv_outer_s)],
        rv_center[0],
        rv_center[1],
        rv_center[2],
        rv_radius_x + rv_wall_thickness,
        rv_radius_y + rv_wall_thickness,
        rv_radius_z + rv_wall_thickness,
    )
    rv_outer_center_of_mass = gmsh.model.occ.getCenterOfMass(3, rv_outer_s)
    rv_outer_bounding_box = gmsh.model.occ.getBoundingBox(3, rv_outer_s)
    logger.debug(f"RV outer: \n{rv_outer_center_of_mass}, \n{rv_outer_bounding_box}")

    rv_inner_s = occ.addSphere(rv_center[0], rv_center[1], rv_center[2], 1)
    occ.dilate(
        [(3, rv_inner_s)],
        rv_center[0],
        rv_center[1],
        rv_center[2],
        rv_radius_x,
        rv_radius_y,
        rv_radius_z,
    )
    rv_inner_center_of_mass = gmsh.model.occ.getCenterOfMass(3, rv_inner_s)
    rv_inner_bounding_box = gmsh.model.occ.getBoundingBox(3, rv_inner_s)
    logger.debug(f"RV inner: \n{rv_inner_center_of_mass}, \n{rv_inner_bounding_box}")

    lv_wall, _ = occ.cut([(3, lv_outer_s)], [(3, lv_inner_s)])
    rv_wall, _ = occ.cut([(3, rv_outer_s)], [(3, rv_inner_s)])

    # Create the myocardium by fusing the LV and RV walls
    # However this will also contain extra bits inside the cavities, so we will cut those out next
    myocardium1, _ = occ.fuse(
        [(3, lv_wall[0][1])],
        [(3, rv_wall[0][1])],
        removeTool=True,
        removeObject=True,
    )

    # Now cut out the inner cavities of the LV to get the final myocardium with a solid septum
    lv_inner_s = occ.addSphere(lv_center[0], lv_center[1], lv_center[2], 1)
    occ.dilate(
        [(3, lv_inner_s)],
        lv_center[0],
        lv_center[1],
        lv_center[2],
        lv_radius_x,
        lv_radius_y,
        lv_radius_z,
    )
    # Here we remove the cells inside the LV cavity only
    myocardium, _ = occ.cut(
        [(3, myocardium1[0][1])], [(3, lv_inner_s)], removeTool=False, removeObject=True
    )

    # Truncate the top to create a flat base
    trunc_box = occ.addBox(-box_size / 2, -box_size / 2, base_cut_z, box_size, box_size, box_size)
    final_model, _ = occ.cut([(3, myocardium[0][1])], [(3, trunc_box)])

    occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)

    labels: dict[str, list[int]] = {
        "LV_EPICARDIUM": [],
        "RV_EPICARDIUM": [],
        "LV_ENDOCARDIUM_FW": [],
        "LV_ENDOCARDIUM_SEPTUM": [],
        "RV_ENDOCARDIUM_FW": [],
        "RV_ENDOCARDIUM_SEPTUM": [],
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
            if center_of_mass[0] < 0:
                labels["LV_ENDOCARDIUM_FW"].append(s[1])
                logger.debug("  -> LV free wall endocardium")
            else:
                labels["LV_ENDOCARDIUM_SEPTUM"].append(s[1])
                logger.debug("  -> LV septal endocardium")

        elif within_bounding_box(bounding_box, rv_inner_bounding_box):
            if center_of_mass[0] > rv_center[0]:
                labels["RV_ENDOCARDIUM_FW"].append(s[1])
                logger.debug("  -> RV free wall endocardium")
            else:
                labels["RV_ENDOCARDIUM_SEPTUM"].append(s[1])
                logger.debug("  -> RV septal endocardium")

        elif within_bounding_box(bounding_box, lv_outer_bounding_box):
            labels["LV_EPICARDIUM"].append(s[1])
            logger.debug("  -> LV epicardium")
        elif within_bounding_box(bounding_box, rv_outer_bounding_box):
            labels["RV_EPICARDIUM"].append(s[1])
            logger.debug("  -> RV epicardium")
        else:
            raise RuntimeError("Surface does not fit any known category")

    # Define Physical Groups for different surfaces
    lv_epi = gmsh.model.addPhysicalGroup(2, labels["LV_EPICARDIUM"])
    gmsh.model.setPhysicalName(2, lv_epi, "LV_EPI_FW")
    rv_epi = gmsh.model.addPhysicalGroup(2, labels["RV_EPICARDIUM"])
    gmsh.model.setPhysicalName(2, rv_epi, "RV_EPI_FW")
    base = gmsh.model.addPhysicalGroup(2, labels["BASE"])
    gmsh.model.setPhysicalName(2, base, "BASE")
    lv_endo_fw = gmsh.model.addPhysicalGroup(2, labels["LV_ENDOCARDIUM_FW"])
    gmsh.model.setPhysicalName(2, lv_endo_fw, "LV_ENDO_FW")
    rv_endo_fw = gmsh.model.addPhysicalGroup(2, labels["RV_ENDOCARDIUM_FW"])
    gmsh.model.setPhysicalName(2, rv_endo_fw, "RV_ENDO_FW")
    rv_septum = gmsh.model.addPhysicalGroup(2, labels["RV_ENDOCARDIUM_SEPTUM"])
    gmsh.model.setPhysicalName(2, rv_septum, "RV_SEPTUM")
    lv_septum = gmsh.model.addPhysicalGroup(2, labels["LV_ENDOCARDIUM_SEPTUM"])
    gmsh.model.setPhysicalName(2, lv_septum, "LV_SEPTUM")

    # myocardium_group = gmsh.model.addPhysicalGroup(3, [volumes[0][1]])
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
