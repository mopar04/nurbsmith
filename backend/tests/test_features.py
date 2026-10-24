import numpy as np
import pytest
import trimesh

from nurbsmith import fixtures as fx
from nurbsmith.convert import convert, repair
from nurbsmith.features import STANDARD_HOLES, _snap_radius
from nurbsmith.step_io import face_type_counts
from nurbsmith.tolerance import LEVELS

# Part: (mesh, CAD truth, face types, features as (type, size) with size = diameter or radius)
PARTS = {
    "holed_plate": (fx.holed_plate, fx.holed_plate_cad, {"PLANE": 6, "CYLINDER": 4},
                    [("hole", 6.6)] * 4),
    "filleted_block": (fx.filleted_block, fx.filleted_block_cad, {"PLANE": 6, "CYLINDER": 4},
                       [("fillet", 5.0)] * 4),
    "boss_plate": (fx.boss_plate, fx.boss_plate_cad, {"PLANE": 7, "CYLINDER": 1},
                   [("boss", 16.0)]),
    "blind_hole_block": (fx.blind_hole_block, fx.blind_hole_block_cad,
                         {"PLANE": 7, "CYLINDER": 1}, [("hole", 10.0)]),
    "bracket": (fx.bracket, fx.bracket_cad, {"PLANE": 8, "CYLINDER": 3},
                [("fillet", 4.0), ("hole", 5.5), ("hole", 5.5)]),
}  # fmt: skip


def scanned(make, sigma: float, seed: int = 3):
    return fx.rotated(fx.add_noise(fx.subdivided(make(), 1.0), sigma, seed=seed), seed)


def sizes(features: list[dict]) -> list[tuple[str, float]]:
    return sorted((f["type"], f.get("diameter_mm", f.get("radius_mm"))) for f in features)


@pytest.mark.parametrize("name", PARTS)
def test_noisy_scan_becomes_cad_features(name):
    make, truth, faces, expected = PARTS[name]
    result = convert(scanned(make, 0.05), LEVELS["functional"])
    report = result.report()

    assert result.solid.isValid()
    assert face_type_counts(result.solid) == faces
    assert sizes(report["features"]) == sorted(expected)
    # Mean surface offset (volume error / area) stays far below the noise.
    cad = truth().val()
    assert abs(result.solid.Volume() - cad.Volume()) / cad.Area() < 0.02
    assert report["fit"]["within_tolerance_pct"] > 99


def test_hole_depths_and_standard_sizes():
    report = convert(scanned(fx.blind_hole_block, 0.05), LEVELS["functional"]).report()
    (hole,) = report["features"]
    assert hole["depth_mm"] == pytest.approx(12.0, abs=0.1)
    assert "through" not in hole

    report = convert(scanned(fx.holed_plate, 0.05), LEVELS["functional"]).report()
    assert {f["standard"] for f in report["features"]} == {"M6 clearance"}
    assert all(f["through"] for f in report["features"])


def test_rough_level_leaves_fillets_out():
    # At 1 mm a 5 mm fillet is within tolerance of a chamfer, so it may come out as one; it
    # must not come out as a fillet.
    result = convert(scanned(fx.filleted_block, 0.05), LEVELS["rough"])
    assert result.solid.isValid()
    assert set(face_type_counts(result.solid)) == {"PLANE"}
    assert all(f["type"] != "fillet" for f in result.report()["features"])


def test_snapping_stays_within_tolerance():
    assert _snap_radius(3.31, 0.3, "hole") == (3.3, "M6 clearance")
    # Too far from any standard size for a tight tolerance: kept as measured, rounded.
    radius, label = _snap_radius(3.71, 0.02, "hole")
    assert label is None and radius == pytest.approx(3.71, abs=0.01)
    assert _snap_radius(4.93, 0.3, "fillet")[0] == 5.0
    assert all(d > 0 for d, _ in STANDARD_HOLES)


def test_loose_pieces_are_dropped_and_reported():
    part = fx.box()
    specks = [trimesh.creation.icosphere(radius=0.5).apply_translation([60 + 3 * i, 0, 0])
              for i in range(3)]  # fmt: skip
    mesh = repair(trimesh.util.concatenate([part, *specks]))
    assert mesh.metadata["loose_pieces"] == 3
    assert np.allclose(mesh.extents, part.extents)

    result = convert(trimesh.util.concatenate([part, *specks]), LEVELS["functional"])
    assert result.scan.loose_pieces == 3
    assert "3 loose pieces" in result.scan.warning
