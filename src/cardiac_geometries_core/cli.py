import math
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
    "--center-lv-x",
    default=0.0,
    type=float,
    help="X-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--center-lv-y",
    default=0.0,
    type=float,
    help="Y-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--center-lv-z",
    default=0.0,
    type=float,
    help="Z-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--a-endo-lv",
    default=2.5,
    type=float,
    help="Dilation of lv endo ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-endo-lv",
    default=1.0,
    type=float,
    help="Dilation of lv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-endo-lv",
    default=1.0,
    type=float,
    help="Dilation of lv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--a-epi-lv",
    default=3.0,
    type=float,
    help="Dilation of lv epi ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-epi-lv",
    default=1.5,
    type=float,
    help="Dilation of lv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-epi-lv",
    default=1.5,
    type=float,
    help="Dilation of lv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--center-rv-x",
    default=0.0,
    type=float,
    help="X-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--center-rv-y",
    default=0.5,
    type=float,
    help="Y-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--center-rv-z",
    default=0.0,
    type=float,
    help="Z-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--a-endo-rv",
    default=3.0,
    type=float,
    help="Dilation of rv endo ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-endo-rv",
    default=1.5,
    type=float,
    help="Dilation of rv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-endo-rv",
    default=1.5,
    type=float,
    help="Dilation of rv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--a-epi-rv",
    default=4.0,
    type=float,
    help="Dilation of rv epi ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-epi-rv",
    default=2.5,
    type=float,
    help="Dilation of rv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-epi-rv",
    default=2.0,
    type=float,
    help="Dilation of rv epi ellipsoid in the z-direction",
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
    char_length: float = 0.5,
    center_lv_x: float = 0.0,
    center_lv_y: float = 0.0,
    center_lv_z: float = 0.0,
    a_endo_lv: float = 2.5,
    b_endo_lv: float = 1.0,
    c_endo_lv: float = 1.0,
    a_epi_lv: float = 3.0,
    b_epi_lv: float = 1.5,
    c_epi_lv: float = 1.5,
    center_rv_x: float = 0.0,
    center_rv_y: float = 0.5,
    center_rv_z: float = 0.0,
    a_endo_rv: float = 3.0,
    b_endo_rv: float = 1.5,
    c_endo_rv: float = 1.5,
    a_epi_rv: float = 4.0,
    b_epi_rv: float = 2.5,
    c_epi_rv: float = 2.0,
    verbose: bool = False,
):
    from .biv_ellipsoid import biv_ellipsoid

    biv_ellipsoid(
        mesh_name=outname,
        char_length=char_length,
        center_lv_x=center_lv_x,
        center_lv_y=center_lv_y,
        center_lv_z=center_lv_z,
        a_endo_lv=a_endo_lv,
        b_endo_lv=b_endo_lv,
        c_endo_lv=c_endo_lv,
        a_epi_lv=a_epi_lv,
        b_epi_lv=b_epi_lv,
        c_epi_lv=c_epi_lv,
        center_rv_x=center_rv_x,
        center_rv_y=center_rv_y,
        center_rv_z=center_rv_z,
        a_endo_rv=a_endo_rv,
        b_endo_rv=b_endo_rv,
        c_endo_rv=c_endo_rv,
        a_epi_rv=a_epi_rv,
        b_epi_rv=b_epi_rv,
        c_epi_rv=c_epi_rv,
        verbose=verbose,
    )


@click.command(help="Create BiV ellipsoidal geometry embedded in a torso")
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
    "--heart-as-surface/--heart-as-volume",
    default=False,
    help="Whether the heart should be a surface of a volume inside the torso",
    show_default=True,
)
@click.option(
    "--torso-length",
    default=20,
    type=float,
    help="Length of torso in the x-direction",
    show_default=True,
)
@click.option(
    "--torso-width",
    default=20,
    type=float,
    help="Length of torso in the y-direction",
    show_default=True,
)
@click.option(
    "--torso-height",
    default=20,
    type=float,
    help="Length of torso in the z-direction",
    show_default=True,
)
@click.option(
    "--rotation-angle",
    default=math.pi / 6,
    type=float,
    help=(
        "Angle to rotate the torso in order to object realistic position of the heart in a torso"
    ),
    show_default=True,
)
@click.option(
    "--center-lv-x",
    default=0.0,
    type=float,
    help="X-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--center-lv-y",
    default=0.0,
    type=float,
    help="Y-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--center-lv-z",
    default=0.0,
    type=float,
    help="Z-coordinate for the center of the lv",
    show_default=True,
)
@click.option(
    "--a-endo-lv",
    default=2.5,
    type=float,
    help="Dilation of lv endo ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-endo-lv",
    default=1.0,
    type=float,
    help="Dilation of lv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-endo-lv",
    default=1.0,
    type=float,
    help="Dilation of lv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--a-epi-lv",
    default=3.0,
    type=float,
    help="Dilation of lv epi ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-epi-lv",
    default=1.5,
    type=float,
    help="Dilation of lv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-epi-lv",
    default=1.5,
    type=float,
    help="Dilation of lv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--center-rv-x",
    default=0.0,
    type=float,
    help="X-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--center-rv-y",
    default=0.5,
    type=float,
    help="Y-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--center-rv-z",
    default=0.0,
    type=float,
    help="Z-coordinate for the center of the rv",
    show_default=True,
)
@click.option(
    "--a-endo-rv",
    default=3.0,
    type=float,
    help="Dilation of rv endo ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-endo-rv",
    default=1.5,
    type=float,
    help="Dilation of rv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-endo-rv",
    default=1.5,
    type=float,
    help="Dilation of rv endo ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--a-epi-rv",
    default=4.0,
    type=float,
    help="Dilation of rv epi ellipsoid in the x-direction",
    show_default=True,
)
@click.option(
    "--b-epi-rv",
    default=2.5,
    type=float,
    help="Dilation of rv epi ellipsoid in the y-direction",
    show_default=True,
)
@click.option(
    "--c-epi-rv",
    default=2.0,
    type=float,
    help="Dilation of rv epi ellipsoid in the z-direction",
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Whether to print GMSH messages to the console",
    show_default=True,
)
def biv_ellipsoid_torso(
    outname: Path,
    char_length: float = 0.5,
    heart_as_surface: bool = True,
    torso_length: float = 20.0,
    torso_width: float = 20.0,
    torso_height: float = 20.0,
    rotation_angle: float = math.pi / 6,
    center_lv_x: float = 0.0,
    center_lv_y: float = 0.0,
    center_lv_z: float = 0.0,
    a_endo_lv: float = 2.5,
    b_endo_lv: float = 1.0,
    c_endo_lv: float = 1.0,
    a_epi_lv: float = 3.0,
    b_epi_lv: float = 1.5,
    c_epi_lv: float = 1.5,
    center_rv_x: float = 0.0,
    center_rv_y: float = 0.5,
    center_rv_z: float = 0.0,
    a_endo_rv: float = 3.0,
    b_endo_rv: float = 1.5,
    c_endo_rv: float = 1.5,
    a_epi_rv: float = 4.0,
    b_epi_rv: float = 2.5,
    c_epi_rv: float = 2.0,
    verbose: bool = False,
):
    from .biv_ellipsoid import biv_ellipsoid_torso

    biv_ellipsoid_torso(
        mesh_name=outname,
        char_length=char_length,
        heart_as_surface=heart_as_surface,
        torso_length=torso_length,
        torso_height=torso_height,
        torso_width=torso_width,
        rotation_angle=rotation_angle,
        center_lv_x=center_lv_x,
        center_lv_y=center_lv_y,
        center_lv_z=center_lv_z,
        a_endo_lv=a_endo_lv,
        b_endo_lv=b_endo_lv,
        c_endo_lv=c_endo_lv,
        a_epi_lv=a_epi_lv,
        b_epi_lv=b_epi_lv,
        c_epi_lv=c_epi_lv,
        center_rv_x=center_rv_x,
        center_rv_y=center_rv_y,
        center_rv_z=center_rv_z,
        a_endo_rv=a_endo_rv,
        b_endo_rv=b_endo_rv,
        c_endo_rv=c_endo_rv,
        a_epi_rv=a_epi_rv,
        b_epi_rv=b_epi_rv,
        c_epi_rv=c_epi_rv,
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


app.add_command(lv_ellipsoid)
app.add_command(lv_ellipsoid_2D)
app.add_command(biv_ellipsoid)
app.add_command(biv_ellipsoid_torso)
app.add_command(slab)
app.add_command(slab_in_bath)
app.add_command(cylinder)
app.add_command(cylinder_racetrack)
app.add_command(cylinder_D_shaped)
