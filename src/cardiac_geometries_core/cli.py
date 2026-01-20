import math
import typing
from importlib.metadata import metadata
from pathlib import Path

import rich_click as click


meta = metadata("cardiac-geometries-core")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["License"]


@click.group()
@click.version_option(__version__, prog_name="cardiac_geometries_code")
def app():
    """
    Cardiac Geometries - A library for creating meshes of
    cardiac geometries
    """
    pass


@click.command(help="Create LV ellipsoidal geometry")
@click.argument(
    "outname",
    default="lv_ellipsoid.msh",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--r-short-endo",
    default=7.0,
    type=float,
    help="Shortest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-short-epi",
    default=10.0,
    type=float,
    help="Shortest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--r-long-endo",
    default=17.0,
    type=float,
    help="Longest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-long-epi",
    default=20.0,
    type=float,
    help="Longest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--psize-ref",
    default=3.0,
    type=float,
    help="The reference point size (smaller values yield as finer mesh",
    show_default=True,
)
@click.option(
    "--mu-apex-endo",
    default=-math.pi,
    type=float,
    help="Angle for the endocardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-endo",
    default=-math.acos(5 / 17),
    type=float,
    help="Angle for the endocardial base",
    show_default=True,
)
@click.option(
    "--mu-apex-epi",
    default=-math.pi,
    type=float,
    help="Angle for the epicardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-epi",
    default=-math.acos(5 / 20),
    type=float,
    help="Angle for the epicardial base",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def lv_ellipsoid(
    outname: Path,
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    psize_ref: float = 3,
    mu_apex_endo: float = -math.pi,
    mu_base_endo: float = -math.acos(5 / 17),
    mu_apex_epi: float = -math.pi,
    mu_base_epi: float = -math.acos(5 / 20),
    verbose: bool = False,
):
    from .lv_ellipsoid import lv_ellipsoid

    lv_ellipsoid(
        mesh_name=outname,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        mu_base_endo=mu_base_endo,
        mu_base_epi=mu_base_epi,
        mu_apex_endo=mu_apex_endo,
        mu_apex_epi=mu_apex_epi,
        psize_ref=psize_ref,
        verbose=verbose,
    )


@click.command(help="Create 2D axisymmetric LV ellipsoidal geometry")
@click.argument(
    "outname",
    default="lv_ellipsoid.msh",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--r-short-endo",
    default=7.0,
    type=float,
    help="Shortest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-short-epi",
    default=10.0,
    type=float,
    help="Shortest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--r-long-endo",
    default=17.0,
    type=float,
    help="Longest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-long-epi",
    default=20.0,
    type=float,
    help="Longest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--psize-ref",
    default=1.0,
    type=float,
    help="The reference point size (smaller values yield as finer mesh",
    show_default=True,
)
@click.option(
    "--mu-apex-endo",
    default=-math.pi,
    type=float,
    help="Angle for the endocardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-endo",
    default=-math.acos(5 / 17),
    type=float,
    help="Angle for the endocardial base",
    show_default=True,
)
@click.option(
    "--mu-apex-epi",
    default=-math.pi,
    type=float,
    help="Angle for the epicardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-epi",
    default=-math.acos(5 / 20),
    type=float,
    help="Angle for the epicardial base",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def lv_ellipsoid_2D(
    outname: Path,
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    psize_ref: float = 1,
    mu_apex_endo: float = -math.pi,
    mu_base_endo: float = -math.acos(5 / 17),
    mu_apex_epi: float = -math.pi,
    mu_base_epi: float = -math.acos(5 / 20),
    verbose: bool = False,
):
    from .lv_ellipsoid import lv_ellipsoid_2D

    lv_ellipsoid_2D(
        mesh_name=outname,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        mu_base_endo=mu_base_endo,
        mu_base_epi=mu_base_epi,
        mu_apex_endo=mu_apex_endo,
        mu_apex_epi=mu_apex_epi,
        psize_ref=psize_ref,
        verbose=verbose,
    )


@click.command(help="Create BiV ellipsoidal geometry")
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--char-length",
    default=0.5,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--base-cut-z",
    default=2.5,
    type=float,
    help="Z-coordinate of the base cut",
    show_default=True,
)
@click.option(
    "--box-size",
    default=15.0,
    type=float,
    help="Size of the cutting box",
    show_default=True,
)
@click.option(
    "--rv-wall-thickness",
    default=0.4,
    type=float,
    help="Thickness of the right ventricle wall",
    show_default=True,
)
@click.option(
    "--lv-wall-thickness",
    default=0.5,
    type=float,
    help="Thickness of the left ventricle wall",
    show_default=True,
)
@click.option(
    "--rv-offset-x",
    default=3.0,
    type=float,
    help="X-offset of the right ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-x",
    default=2.0,
    type=float,
    help="X-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-y",
    default=1.8,
    type=float,
    help="Y-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-z",
    default=3.25,
    type=float,
    help="Z-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-x",
    default=1.9,
    type=float,
    help="X-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-y",
    default=2.5,
    type=float,
    help="Y-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-z",
    default=3.0,
    type=float,
    help="Z-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def biv_ellipsoid(
    outname: Path,
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
    from .biv_ellipsoid import biv_ellipsoid

    biv_ellipsoid(
        mesh_name=outname,
        char_length=char_length,
        base_cut_z=base_cut_z,
        box_size=box_size,
        rv_wall_thickness=rv_wall_thickness,
        lv_wall_thickness=lv_wall_thickness,
        rv_offset_x=rv_offset_x,
        lv_radius_x=lv_radius_x,
        lv_radius_y=lv_radius_y,
        lv_radius_z=lv_radius_z,
        rv_radius_x=rv_radius_x,
        rv_radius_y=rv_radius_y,
        rv_radius_z=rv_radius_z,
        verbose=verbose,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--lx",
    default=20.0,
    type=float,
    help="Length of slab in the x-direction",
    show_default=True,
)
@click.option(
    "--ly",
    default=7.0,
    type=float,
    help="Length of slab in the y-direction",
    show_default=True,
)
@click.option(
    "--lz",
    default=1.0,
    type=float,
    help="Length of slab in the z-direction",
    show_default=True,
)
@click.option(
    "--dx",
    default=1.0,
    type=float,
    help="Element size",
    show_default=True,
)
def slab(
    outname: Path,
    lx: float = 20.0,
    ly: float = 7.0,
    lz: float = 3.0,
    dx: float = 1.0,
):
    from .slab import slab

    slab(
        mesh_name=outname,
        lx=lx,
        ly=ly,
        lz=lz,
        dx=dx,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--lx",
    default=1.0,
    type=float,
    help="Length of slab in the x-direction",
    show_default=True,
)
@click.option(
    "--ly",
    default=0.01,
    type=float,
    help="Length of slab in the y-direction",
    show_default=True,
)
@click.option(
    "--lz",
    default=0.5,
    type=float,
    help="Length of slab in the z-direction",
    show_default=True,
)
@click.option(
    "--bx",
    default=0.0,
    type=float,
    help="Thickness of bath in the x-direction",
    show_default=True,
)
@click.option(
    "--by",
    default=0.0,
    type=float,
    help="Thickness of bath in the y-direction",
    show_default=True,
)
@click.option(
    "--bz",
    default=0.1,
    type=float,
    help="Thickness of bath in the z-direction",
    show_default=True,
)
@click.option(
    "--dx",
    default=0.01,
    type=float,
    help="Element size",
    show_default=True,
)
def slab_in_bath(
    outname: Path,
    lx: float = 1.0,
    ly: float = 0.01,
    lz: float = 0.5,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.1,
    dx: float = 0.01,
    verbose: bool = False,
):
    from .slab import slab_in_bath

    slab_in_bath(
        mesh_name=outname,
        lx=lx,
        ly=ly,
        lz=lz,
        bx=bx,
        by=by,
        bz=bz,
        dx=dx,
        verbose=verbose,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--ri",
    default=10.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--ro",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def cylinder(
    outname: Path,
    ri: float = 10.0,
    ro: float = 20.0,
    height: float = 40.0,
    char_length: float = 10.0,
    verbose: bool = False,
):
    from .cylinder import cylinder

    cylinder(
        mesh_name=outname,
        inner_radius=ri,
        outer_radius=ro,
        height=height,
        char_length=char_length,
        verbose=verbose,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--ri",
    default=13.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--ro",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--inner-flat-face-distance",
    "-if",
    default=10.0,
    type=float,
    help=(
        "The distance of the inner flat face from the center (along the x-axis)."
        "This value must be less than inner_radius. Default is 5.0."
    ),
    show_default=True,
)
@click.option(
    "--outer-flat-face-distance",
    "-of",
    default=17.0,
    type=float,
    help=(
        "The distance of the outer flat face from the center (along the x-axis)."
        "This value must be less than outer_radius. Default is 15.0."
    ),
    show_default=True,
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def cylinder_racetrack(
    outname: Path,
    ri: float = 13.0,
    ro: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    char_length: float = 10.0,
    verbose: bool = False,
):
    from .cylinder import cylinder_racetrack

    cylinder_racetrack(
        mesh_name=outname,
        inner_radius=ri,
        outer_radius=ro,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=char_length,
        verbose=verbose,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--ri",
    default=10.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--ro",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--inner-flat-face-distance",
    "-if",
    default=5.0,
    type=float,
    help=(
        "The distance of the inner flat face from the center (along the x-axis)."
        "This value must be less than inner_radius. Default is 5.0."
    ),
    show_default=True,
)
@click.option(
    "--outer-flat-face-distance",
    "-of",
    default=15.0,
    type=float,
    help=(
        "The distance of the outer flat face from the center (along the x-axis)."
        "This value must be less than outer_radius. Default is 15.0."
    ),
    show_default=True,
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def cylinder_D_shaped(
    outname: Path,
    ri: float = 10.0,
    ro: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 5.0,
    outer_flat_face_distance: float = 15.0,
    char_length: float = 10.0,
    verbose: bool = False,
):
    from .cylinder import cylinder_D_shaped

    cylinder_D_shaped(
        mesh_name=outname,
        inner_radius=ri,
        outer_radius=ro,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=char_length,
        verbose=verbose,
    )


@click.command()
@click.argument(
    "outname",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--mode",
    help="Mode of cut: 'd_shaped' or 'racetrack'",
    default="racetrack",
    type=click.Choice(["d_shaped", "racetrack"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--ri",
    default=13.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--ro",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--inner-flat-face-distance",
    "-if",
    default=10.0,
    type=float,
    help=(
        "The distance of the inner flat face from the center (along the x-axis)."
        "This value must be less than inner_radius. Default is 5.0."
    ),
    show_default=True,
)
@click.option(
    "--outer-flat-face-distance",
    "-of",
    default=17.0,
    type=float,
    help=(
        "The distance of the outer flat face from the center (along the x-axis)."
        "This value must be less than outer_radius. Default is 15.0."
    ),
    show_default=True,
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
@click.option(
    "--fillet-radius",
    default=None,
    type=float,
    help="Radius of fillet to apply to the cut edges",
    show_default=True,
)
def cylinder_cut(
    outname: Path,
    ri: float = 13.0,
    ro: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    char_length: float = 10.0,
    verbose: bool = False,
    mode: typing.Literal["racetrack", "d_shaped"] = "racetrack",
    fillet_radius: float | None = None,
):
    from .cylinder import cylinder_cut

    cylinder_cut(
        mesh_name=outname,
        inner_radius=ri,
        outer_radius=ro,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=char_length,
        verbose=verbose,
        mode=mode,
        fillet_radius=fillet_radius,
    )


app.add_command(lv_ellipsoid)
app.add_command(lv_ellipsoid_2D)
app.add_command(biv_ellipsoid)
app.add_command(slab)
app.add_command(slab_in_bath)
app.add_command(cylinder)
app.add_command(cylinder_racetrack)
app.add_command(cylinder_D_shaped)
app.add_command(cylinder_cut)
