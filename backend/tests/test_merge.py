"""Merging a top and a bottom scan of one part, with no design to line them up against."""

import numpy as np
import pytest
import trimesh

from nurbsmith import fixtures as fx
from nurbsmith.convert import convert
from nurbsmith.merge import MergeError, merge_scans
from nurbsmith.tolerance import LEVELS


def scan(side: trimesh.Trimesh) -> trimesh.Trimesh:
    return fx.add_noise(fx.subdivided(side, 1.0), 0.03, seed=2)


def flipped(mesh: trimesh.Trimesh, seed: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    """The bottom scan as the scanner returns it: turned over and moved."""
    move = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    move[:3, 3] = [40.0, -25.0, 10.0]
    out = mesh.copy()
    out.apply_transform(move)
    return out, move


@pytest.mark.parametrize("make", [fx.boss_plate, fx.bracket, fx.blind_hole_block])
def test_bottom_scan_lands_where_it_belongs(make):
    part = make()
    top, bottom = (scan(s) for s in fx.sides(part))
    moved, _ = flipped(bottom, seed=5)
    result = merge_scans(top, moved)
    # Placed, the bottom scan lies on the part. (Not necessarily point for point where it
    # started: a symmetric part turned half round looks the same from below.)
    placed = trimesh.transform_points(moved.vertices, result.transform)
    _, d, _ = trimesh.proximity.closest_point(part, placed)
    assert d.mean() < 0.1
    assert np.percentile(d, 99) < 0.3
    assert result.air_pct < 2


def test_merged_scans_convert_like_the_whole_part():
    top, bottom = (scan(s) for s in fx.sides(fx.boss_plate()))
    moved, _ = flipped(bottom, seed=7)
    merged = merge_scans(top, moved).mesh
    report = convert(merged, LEVELS["functional"]).report()
    assert report["fit"]["within_tolerance_pct"] > 99
    assert [f["type"] for f in report["features"]] == ["boss"]


def test_scans_that_do_not_overlap_are_refused():
    # Two balls of very different sizes: no pose puts one surface on the other.
    small, big = fx.ball(radius=10.0, squash=1.0), fx.ball(radius=40.0, squash=1.0)
    with pytest.raises(MergeError):
        merge_scans(small, big)
