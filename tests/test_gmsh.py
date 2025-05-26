import cardiac_geometries_core as cgc


def test_lv_prolate_flat_base(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.prolate_lv_ellipsoid_flat_base(mesh_path)


def test_lv_2D(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.lv_ellipsoid_2D(mesh_path)


def test_lv_prolate(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.prolate_lv_ellipsoid(mesh_path)


def test_lv_flat_base(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.lv_ellipsoid_flat_base(mesh_path)


def test_lv_simple(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.lv_ellipsoid(mesh_path, psize_ref=0.05)


def test_create_benchmark_geometry_land15(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.create_benchmark_geometry_land15(mesh_name=mesh_path)


def test_slab(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.slab(mesh_name=mesh_path)


def test_biv_ellipsoid(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.biv_ellipsoid(mesh_name=mesh_path)


def test_biv_ellipsoid_torso(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.biv_ellipsoid_torso(mesh_name=mesh_path)


def test_cylinder(tmp_path):
    mesh_path = tmp_path.with_suffix(".msh")
    cgc.cylinder(mesh_name=mesh_path)
