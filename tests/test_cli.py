from pathlib import Path
import pytest
from cardiac_geometries_core import cli
from click.testing import CliRunner


@pytest.mark.parametrize(
    "script",
    [
        cli.cylinder,
        cli.cylinder_racetrack,
        cli.cylinder_D_shaped,
        cli.slab,
        cli.slab_in_bath,
        cli.lv_ellipsoid,
        cli.lv_ellipsoid_2D,
        cli.biv_ellipsoid,
    ],
    ids=[
        "cylinder",
        "cylinder_racetrack",
        "cylinder_D_shaped",
        "slab",
        "slab_in_bath",
        "lv_ellipsoid",
        "lv_ellipsoid_2D",
        "biv_ellipsoid",
    ],
)
def test_script(script, tmp_path: Path):
    runner = CliRunner()
    path = tmp_path.with_suffix(".msh")
    res = runner.invoke(script, [path.as_posix()])
    assert res.exit_code == 0
    assert path.is_file()


@pytest.mark.parametrize(
    "mode",
    [
        "d_shaped",
        "racetrack",
    ],
)
def test_cylinder_cut(mode: str, tmp_path: Path):
    runner = CliRunner()
    path = tmp_path.with_suffix(".msh")
    res = runner.invoke(cli.cylinder_cut, ["--mode", mode, path.as_posix()])
    assert res.exit_code == 0
    assert path.is_file()
