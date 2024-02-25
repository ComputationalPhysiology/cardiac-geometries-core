from pathlib import Path

import cardiac_geometries_core as cgc


def test_lv_prolate_flat_base():
    mesh_path = Path("prolate_lv_ellipsoid_flat_base.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.prolate_lv_ellipsoid_flat_base(mesh_path)
    mesh_path.unlink(missing_ok=False)


def test_lv_prolate():
    mesh_path = Path("prolate_lv_ellipsoid.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.prolate_lv_ellipsoid(mesh_path)
    mesh_path.unlink(missing_ok=False)


def test_lv_flat_base():
    mesh_path = Path("lv_ellipsoid_flat_base.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.lv_ellipsoid_flat_base(mesh_path)
    mesh_path.unlink(missing_ok=False)


def test_lv_simple():
    mesh_path = Path("lv_ellipsoid.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.lv_ellipsoid(mesh_path, psize_ref=0.05)
    mesh_path.unlink(missing_ok=False)


def test_create_benchmark_geometry_land15():
    path = cgc.create_benchmark_geometry_land15()
    path.unlink(missing_ok=False)


def test_slab(tmp_path):
    mesh_name = tmp_path / "mesh.msh"
    path = cgc.slab(mesh_name)
    path.unlink(missing_ok=False)


def test_biv_ellipsoid():
    mesh_path = Path("biv_ellipsoid.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.biv_ellipsoid(mesh_name=mesh_path)
    mesh_path.unlink(missing_ok=False)


def test_biv_ellipsoid_torso():
    mesh_path = Path("biv_ellipsoid_torso.msh")
    mesh_path.unlink(missing_ok=True)
    cgc.biv_ellipsoid_torso(mesh_name=mesh_path)
    mesh_path.unlink(missing_ok=False)
