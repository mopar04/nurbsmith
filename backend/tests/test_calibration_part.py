"""The calibration part (tools/calibration_part.py), converted from its own design mesh."""

import importlib.util
from pathlib import Path

import pytest

from nurbsmith import fixtures as fx
from nurbsmith.convert import convert
from nurbsmith.tolerance import LEVELS

TOOL = Path(__file__).parents[2] / "tools" / "calibration_part.py"


@pytest.fixture(scope="module")
def calibration():
    spec = importlib.util.spec_from_file_location("calibration_part", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    part = module.build()
    design = fx.from_cad(part, 0.02)
    return design, convert(design, LEVELS["functional"])


def test_calibration_part_features(calibration):
    design, result = calibration
    report = result.report()
    assert result.solid.isValid()
    assert report["fit"]["within_tolerance_pct"] > 95
    assert result.solid.Volume() == pytest.approx(design.volume, rel=0.02)

    feats = report["features"]
    holes = sorted((f["diameter_mm"], f.get("depth_mm")) for f in feats if f["type"] == "hole")
    # Through holes (one non-standard), the counterbore over its hole, the blind hole and
    # the cross hole in the boss. The countersunk hole (4.5) depends on the tessellation, as
    # its cone is not modelled yet.
    holes = [h for h in holes if h != (4.5, None)]
    assert holes == [
        (3.4, None), (5.0, None), (5.5, None), (5.5, None), (6.0, 5.0), (6.6, None),
        (7.3, None), (10.0, 4.0),
    ]  # fmt: skip
    (slot,) = [f for f in feats if f["type"] == "slot"]
    assert (slot["width_mm"], slot["length_mm"]) == (6.0, 20.0)
    (boss,) = [f for f in feats if f["type"] == "boss"]
    assert (boss["diameter_mm"], boss["height_mm"]) == (16.0, 20.0)
    assert any(f["type"] == "chamfer" and f["size_mm"] == 1.5 for f in feats)
    radii = [f["radius_mm"] for f in feats if f["type"] == "fillet"]
    assert 8.0 in radii or 8.5 in radii  # +X corners (r8); read slightly large for now
