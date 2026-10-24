"""Synthetic test meshes with known ground truth, optionally with scanner-like noise."""

from __future__ import annotations

import numpy as np
import trimesh


def add_noise(mesh: trimesh.Trimesh, sigma_mm: float, seed: int = 0) -> trimesh.Trimesh:
    """Jitter every vertex by isotropic Gaussian noise (sigma per axis), like a scanner.

    Not along vertex normals: at a CAD edge those point sideways, which would slide edge
    vertices within the face and leave long runs of them perfectly coplanar.
    """
    rng = np.random.default_rng(seed)
    noisy = mesh.copy()
    noisy.vertices = noisy.vertices + rng.normal(0.0, sigma_mm, size=noisy.vertices.shape)
    return noisy


def box(size=(40.0, 30.0, 20.0), subdivisions: int = 3) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    return mesh


def cylinder(radius: float = 10.0, height: float = 30.0, sections: int = 96) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def ball(radius: float = 15.0, squash: float = 0.6) -> trimesh.Trimesh:
    """A squashed sphere: curved everywhere, and not a cylinder anywhere."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    mesh.apply_scale([1.0, 1.0, squash])
    return mesh


def plate_with_hole(
    size=(60.0, 40.0, 8.0), hole_radius: float = 6.0, sections: int = 96
) -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=size)
    tool = trimesh.creation.cylinder(radius=hole_radius, height=size[2] * 2, sections=sections)
    return plate.difference(tool, engine="manifold")


def l_bracket(base=(40.0, 20.0, 5.0), upright_height: float = 30.0) -> trimesh.Trimesh:
    """Non-convex: a base plate with an upright wall at one end."""
    bx, by, bz = base
    plate = trimesh.creation.box(extents=base)
    wall = trimesh.creation.box(
        extents=(bz, by, upright_height),
        transform=trimesh.transformations.translation_matrix(
            ((bx - bz) / 2, 0, (upright_height - bz) / 2)
        ),
    )
    return plate.union(wall, engine="manifold")


def stepped_block() -> trimesh.Trimesh:
    """Three steps: lots of concave and convex edges, and 4-plane corners."""
    parts = [
        trimesh.creation.box(
            extents=(60.0 - 20 * i, 30.0, 10.0),
            transform=trimesh.transformations.translation_matrix((10.0 * i, 0, 5.0 + 10 * i)),
        )
        for i in range(3)
    ]
    return trimesh.boolean.union(parts, engine="manifold")


def frame(outer=(50.0, 40.0, 6.0), window=(30.0, 20.0)) -> trimesh.Trimesh:
    """A plate with a rectangular through-window: faces with inner loops."""
    plate = trimesh.creation.box(extents=outer)
    cut = trimesh.creation.box(extents=(window[0], window[1], outer[2] * 2))
    return plate.difference(cut, engine="manifold")


def subdivided(mesh: trimesh.Trimesh, max_edge: float = 3.0) -> trimesh.Trimesh:
    """Split long triangles so noise looks like a real scan's dense sampling."""
    verts, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
    return trimesh.Trimesh(verts, faces)


def rotated(mesh: trimesh.Trimesh, seed: int = 1) -> trimesh.Trimesh:
    """Arbitrary orientation, so tests do not lean on world axes."""
    out = mesh.copy()
    out.apply_transform(
        trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    )
    return out


def sides(mesh: trimesh.Trimesh, up=(0.0, 0.0, 1.0)):
    """The part as a turntable scanner sees it from above and, turned over, from below: the
    faces a ray from 30 or 60 degrees above (or below) the turntable reaches unblocked. The
    two overlap only on the walls. Works on the faces as given, so pass the coarse CAD mesh
    and subdivide each side afterwards."""
    up = np.asarray(up, dtype=float)
    side = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    out = []
    for sign in (1.0, -1.0):
        seen = np.zeros(len(mesh.faces), dtype=bool)
        for elevation in np.radians([30.0, 60.0]):
            for azimuth in np.radians(np.arange(0.0, 360.0, 45.0)):
                ray = np.cos(elevation) * (np.cos(azimuth) * side + np.sin(azimuth) * other)
                ray = ray + sign * np.sin(elevation) * up
                facing = mesh.face_normals @ ray > 0.05
                idx = np.flatnonzero(facing & ~seen)
                if not len(idx):
                    continue
                start = mesh.triangles_center[idx] + 1e-3 * mesh.face_normals[idx]
                blocked = mesh.ray.intersects_any(start, np.tile(ray, (len(idx), 1)))
                seen[idx[~blocked]] = True
        part = mesh.copy()
        part.update_faces(seen)
        part.remove_unreferenced_vertices()
        out.append(part)
    return out[0], out[1]


# ---------------------------------------------------------------------------------------------
# Parts with curved features, built as real CAD and tessellated, so the ground truth is exact.
# ---------------------------------------------------------------------------------------------


def from_cad(shape, tolerance: float = 0.02) -> trimesh.Trimesh:
    """Tessellate a CadQuery shape or workplane into a mesh."""
    import cadquery as cq

    if isinstance(shape, cq.Workplane):
        shape = shape.val()
    verts, tris = shape.tessellate(tolerance, angularTolerance=0.1)
    mesh = trimesh.Trimesh(np.array([v.toTuple() for v in verts]), np.array(tris))
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def holed_plate_cad():
    """60 x 40 x 8 plate with four M6 clearance holes (6.6 mm)."""
    import cadquery as cq

    return (
        cq.Workplane("XY")
        .box(60, 40, 8)
        .faces(">Z")
        .workplane()
        .pushPoints([(-20, -10), (20, -10), (-20, 10), (20, 10)])
        .hole(6.6)
    )


def filleted_block_cad():
    """50 x 30 x 20 block with its four vertical edges filleted at 5 mm."""
    import cadquery as cq

    return cq.Workplane("XY").box(50, 30, 20).edges("|Z").fillet(5)


def boss_plate_cad():
    """60 x 40 x 6 plate with a 16 mm boss, 10 mm tall."""
    import cadquery as cq

    return cq.Workplane("XY").box(60, 40, 6).faces(">Z").workplane().circle(8).extrude(10)


def blind_hole_block_cad():
    """40 x 40 x 20 block with a 10 mm blind hole, 12 mm deep."""
    import cadquery as cq

    return cq.Workplane("XY").box(40, 40, 20).faces(">Z").workplane().hole(10, 12)


def bracket_cad():
    """L-bracket with a 4 mm inside fillet and two 5.5 mm holes in the base."""
    import cadquery as cq

    base = cq.Workplane("XY").box(60, 30, 6).translate((0, 0, 3))
    wall = cq.Workplane("XY").box(6, 30, 40).translate((27, 0, 20))
    body = base.union(wall).edges("|Y").edges(cq.selectors.NearestToPointSelector((24, 0, 6)))
    body = body.fillet(4)
    return (
        body.faces("<Z")
        .workplane(centerOption="CenterOfBoundBox")
        .pushPoints([(-15, 0), (5, 0)])
        .hole(5.5)
    )


def holed_plate() -> trimesh.Trimesh:
    return from_cad(holed_plate_cad())


def filleted_block() -> trimesh.Trimesh:
    return from_cad(filleted_block_cad())


def boss_plate() -> trimesh.Trimesh:
    return from_cad(boss_plate_cad())


def blind_hole_block() -> trimesh.Trimesh:
    return from_cad(blind_hole_block_cad())


def bracket() -> trimesh.Trimesh:
    return from_cad(bracket_cad())
