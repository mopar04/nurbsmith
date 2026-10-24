"""Snap fitted planes to clean geometry and rebuild a watertight B-rep solid.

Corners come from intersecting planes, never from mesh vertices. Each region's boundary in the
mesh tells us which neighbours it meets, and in what order; each change of neighbour along that
boundary is a corner where three planes meet. That keeps the output exact and noise-free even
when the scan's edges are ragged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.ShapeFix import ShapeFix_Solid
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from .segment import Plane, Region, face_labels, plane_offset, update_deviation
from .tolerance import ToleranceLevel

# Adjacent regions meeting at less than this (but not parallel) are a curved surface faceted
# into strips.
MIN_CREASE_DEG = 20.0
# A region whose plane-fit RMS exceeds this share of the tolerance is curved.
CURVED_TOL_RATIO = 0.5
# A face counts as flat but warped while its RMS bow stays under this share of its size
# (square root of its area): about 0.5 mm over a 100 mm face.
WARP_SHARE = 0.005
# Beyond this share of the surface, curved panels are too much to approximate with planes.
FREEFORM_SHARE = 0.15


# Neighbour id for a region boundary that runs along a gap in the scan.
GAP = -2


class UnsupportedGeometry(ValueError):
    """The mesh has geometry this version cannot rebuild yet."""


@dataclass
class _Cluster:
    direction: np.ndarray
    weight: float


# ---------------------------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------------------------


def snap_planes(mesh: trimesh.Trimesh, regions: list[Region], level: ToleranceLevel) -> None:
    """Make near-parallel planes parallel and near-perpendicular directions exactly 90 deg."""
    cos_par = np.cos(np.radians(level.angle_snap_deg))
    sin_perp = np.sin(np.radians(level.angle_snap_deg))

    clusters: list[_Cluster] = []
    assignment: dict[int, tuple[int, float]] = {}
    for r in sorted(regions, key=lambda r: -r.area):
        n = r.plane.normal
        for ci, c in enumerate(clusters):
            dot = float(n @ c.direction)
            if abs(dot) >= cos_par:
                sign = 1.0 if dot > 0 else -1.0
                total = c.weight + r.area
                merged = c.direction * c.weight + sign * n * r.area
                c.direction = merged / np.linalg.norm(merged)
                c.weight = total
                assignment[r.id] = (ci, sign)
                break
        else:
            clusters.append(_Cluster(n.copy(), r.area))
            assignment[r.id] = (len(clusters) - 1, 1.0)

    # Gram-Schmidt against heavier directions, but only where they are nearly perpendicular.
    order = sorted(range(len(clusters)), key=lambda i: -clusters[i].weight)
    fixed: list[np.ndarray] = []
    for i in order:
        d = clusters[i].direction
        for f in fixed:
            if abs(d @ f) <= sin_perp:
                d = d - (d @ f) * f
                d = d / np.linalg.norm(d)
        clusters[i].direction = d
        fixed.append(d)

    for r in regions:
        ci, sign = assignment[r.id]
        normal = sign * clusters[ci].direction
        r.plane = Plane(normal, plane_offset(mesh, r.faces, normal))
        r.cluster = ci  # type: ignore[attr-defined]


def merge_coplanar(
    mesh: trimesh.Trimesh, regions: list[Region], level: ToleranceLevel
) -> list[Region]:
    """Merge adjacent regions that ended up on the same plane after snapping."""
    parent = {r.id: r.id for r in regions}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_id = {r.id: r for r in regions}
    for r in regions:
        for nb in r.neighbours:
            o = by_id[nb]
            same_dir = r.plane.normal @ o.plane.normal > 0.9999
            # A step smaller than the smallest kept feature (embossed text, a mould line, a
            # warp in the scan) is not a feature: one face.
            step = max(level.fit_tolerance_mm, level.min_feature_mm / 2)
            if same_dir and abs(r.plane.offset - o.plane.offset) <= step:
                parent[find(r.id)] = find(o.id)

    groups: dict[int, list[Region]] = {}
    for r in regions:
        groups.setdefault(find(r.id), []).append(r)
    if all(len(g) == 1 for g in groups.values()):
        return regions

    merged: list[Region] = []
    for new_id, group in enumerate(groups.values()):
        faces = np.concatenate([g.faces for g in group])
        normal = group[0].plane.normal
        plane = Plane(normal, plane_offset(mesh, faces, normal))
        merged.append(Region(new_id, faces, plane, float(mesh.area_faces[faces].sum())))
    labels = face_labels(merged, len(mesh.faces))
    a = labels[mesh.face_adjacency[:, 0]]
    b = labels[mesh.face_adjacency[:, 1]]
    for x, y in zip(a, b, strict=True):
        if x != y and x >= 0 and y >= 0:
            merged[x].neighbours.add(int(y))
            merged[y].neighbours.add(int(x))
    update_deviation(mesh, merged)
    return merged


def check_supported(
    regions: list[Region], level: ToleranceLevel, unmodelled: float = 0.0, curved_total: float = 0.0
) -> str | None:
    """Refuse parts that are mostly freeform, rather than return a faceted approximation.

    Holes, bosses and fillets are already out of the way by now. Whatever is left that meets a
    neighbour at a shallow angle, or bows well beyond the tolerance, is part of a curved panel.
    A little of that (a rounded rim, a blended corner) is approximated with flat faces and
    noted; a lot of it is a sculpted part that needs freeform surfaces. A face that is only
    warped a little for its size (bent sheet metal, a moulding that sagged) stays one plane and
    just costs some fit score.

    unmodelled is the area of curved patches that became no feature; curved_total is the area
    of all curved patches, so the share is of the whole surface.
    """
    by_id = {r.id: r for r in regions}
    cos_min = np.cos(np.radians(MIN_CREASE_DEG))
    cos_parallel = np.cos(np.radians(level.angle_snap_deg))
    curved: set[int] = set()
    for r in regions:
        for nb in r.neighbours:
            if nb in by_id and cos_parallel > r.plane.normal @ by_id[nb].plane.normal > cos_min:
                # The smaller of the two is the facet; the bigger is usually a real face.
                curved.add(min((r, by_id[nb]), key=lambda x: x.area).id)
        if r.bow_mm > max(CURVED_TOL_RATIO * level.fit_tolerance_mm, WARP_SHARE * np.sqrt(r.area)):
            curved.add(r.id)
    if not curved and not unmodelled:
        return None
    area = sum(by_id[i].area for i in curved) + unmodelled
    total = sum(r.area for r in regions) + curved_total
    if area > FREEFORM_SHARE * total:
        raise UnsupportedGeometry(
            "This part is largely made of curved surfaces other than holes, round bosses and "
            "fillets (like a domed or sculpted panel). Freeform surfaces are the next "
            "milestone."
        )
    return (
        f"About {100 * area / total:.0f}% of the surface is gently curved and was approximated "
        "with flat faces."
    )


# ---------------------------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------------------------


def _directed_boundary(
    mesh: trimesh.Trimesh, labels: np.ndarray
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """For each region: start vertex -> [(end vertex, neighbour region)], in face winding order."""
    out: dict[int, dict[int, list[tuple[int, int]]]] = {}
    fa, fb = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    for (f, g), (v1, v2) in zip(np.stack([fa, fb], axis=1), mesh.face_adjacency_edges, strict=True):
        la, lb = labels[f], labels[g]
        if la == lb:
            continue
        for face, own, other in ((f, la, lb), (g, lb, la)):
            tri = list(mesh.faces[face])
            i = tri.index(v1)
            start, end = (v1, v2) if tri[(i + 1) % 3] == v2 else (v2, v1)
            out.setdefault(int(own), {}).setdefault(int(start), []).append((int(end), int(other)))
    # Edges of gaps in the scan: they close each region's loops, and the neighbour is GAP.
    directed = np.stack([mesh.faces, np.roll(mesh.faces, -1, axis=1)], axis=2).reshape(-1, 2)
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    for idx in np.flatnonzero(counts[inverse.ravel()] == 1):
        start, end = directed[idx]
        own = labels[idx // 3]
        out.setdefault(int(own), {}).setdefault(int(start), []).append((int(end), GAP))
    return out


def _trace_loops(edges: dict[int, list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Chain directed edges into closed loops of (vertex, neighbour) steps."""
    remaining = {k: list(v) for k, v in edges.items()}
    loops = []
    while any(remaining.values()):
        start = next(k for k, v in remaining.items() if v)
        loop, v = [], start
        while True:
            nxt = remaining.get(v)
            if not nxt:
                raise UnsupportedGeometry("Region boundary is not closed; the mesh may have holes.")
            end, nb = nxt.pop()
            loop.append((v, nb))
            v = end
            if v == start:
                break
        loops.append(loop)
    return loops


def _corner(planes: list[Plane], hint: np.ndarray) -> np.ndarray:
    """Point on all given planes, pulled toward hint when they are nearly degenerate."""
    n = np.array([p.normal for p in planes])
    d = np.array([p.offset for p in planes])
    lam = 1e-4
    a = np.vstack([n, lam * np.eye(3)])
    b = np.concatenate([d, lam * hint])
    return np.linalg.lstsq(a, b, rcond=None)[0]


def rebuild_solid(
    mesh: trimesh.Trimesh,
    regions: list[Region],
    tol: float,
    labels: np.ndarray | None = None,
    through: Callable[[int, int], int | None] | None = None,
) -> cq.Solid:
    """Solid bounded by the given planes, with corners from plane intersections.

    labels may also mark faces that belong to something other than these planes (a fillet, a
    hole); through(region, other) then says which plane to treat that neighbour as (the plane
    on the far side of a fillet), or None to ignore it (a hole, which leaves the face whole).
    """
    if labels is None:
        labels = face_labels(regions, len(mesh.faces))
    by_id = {r.id: r for r in regions}
    boundary = _directed_boundary(mesh, labels)

    corner_cache: dict[frozenset[int], np.ndarray] = {}
    face_loops: dict[int, list[list[frozenset[int]]]] = {}
    for rid, edges in boundary.items():
        if rid not in by_id:
            continue
        face_loops[rid] = []
        for loop in _trace_loops(edges):
            # Neighbour sequence with consecutive repeats collapsed, cyclically.
            seq: list[tuple[int, int]] = []
            for v, nb in loop:
                if nb not in by_id:
                    nb = through(rid, nb) if through else None
                    if nb is None:
                        continue
                if not seq or seq[-1][1] != nb:
                    seq.append((v, nb))
            if len(seq) > 1 and seq[0][1] == seq[-1][1]:
                seq[0] = seq.pop()
            if len(seq) < 2:
                # The rim of a hole, or a sliver of the face cut off along one edge (between
                # a fillet and a chamfer): the face is rebuilt without it.
                continue
            if len(seq) < 3:
                raise UnsupportedGeometry(
                    f"Flat region {rid} meets fewer than 3 neighbours along one boundary; "
                    "it is probably part of a curved surface."
                )
            keys = []
            for i, (v, nb) in enumerate(seq):
                prev_nb = seq[i - 1][1]
                key = frozenset((rid, prev_nb, nb))
                if key not in corner_cache:
                    planes = [by_id[k].plane for k in key]
                    corner_cache[key] = _corner(planes, mesh.vertices[v])
                keys.append(key)
            face_loops[rid].append(keys)

    points = _merge_close_corners(corner_cache, tol)

    faces = []
    for rid, loops in face_loops.items():
        wires = []
        for keys in loops:
            # Merged 4-plane corners may sit a hair off this face; keep each wire exactly planar
            # and let sewing close the gaps.
            pts = [_project(points[k], by_id[rid].plane) for k in keys]
            pts = [p for i, p in enumerate(pts) if np.linalg.norm(p - pts[i - 1]) > 1e-9]
            if len(pts) < 3:
                continue
            wires.append((_loop_area(pts, by_id[rid].plane.normal), pts))
        if not wires:
            continue
        wires.sort(key=lambda w: -abs(w[0]))
        outer = cq.Wire.makePolygon([tuple(p) for p in wires[0][1]], close=True)
        inner = [cq.Wire.makePolygon([tuple(p) for p in w[1]], close=True) for w in wires[1:]]
        faces.append(cq.Face.makeFromWires(outer, inner))

    return _sew_solid(faces, tol)


def _merge_close_corners(
    cache: dict[frozenset[int], np.ndarray], tol: float
) -> dict[frozenset[int], np.ndarray]:
    """Where 4+ planes meet, triples give nearly the same point; collapse them to one."""
    keys = list(cache)
    pts = np.array([cache[k] for k in keys])
    merged = {}
    used = np.zeros(len(keys), bool)
    for i in range(len(keys)):
        if used[i]:
            continue
        close = np.flatnonzero(~used & (np.linalg.norm(pts - pts[i], axis=1) <= tol))
        used[close] = True
        centre = pts[close].mean(axis=0)
        for j in close:
            merged[keys[j]] = centre
    return merged


def _project(p: np.ndarray, plane: Plane) -> np.ndarray:
    return p - plane.distance(p[None])[0] * plane.normal


def _loop_area(pts: list[np.ndarray], normal: np.ndarray) -> float:
    total = np.zeros(3)
    for i in range(len(pts)):
        total += np.cross(pts[i - 1], pts[i])
    return float(total @ normal) / 2


def _sew_solid(faces: list[cq.Face], tol: float) -> cq.Solid:
    sewing = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sewing.Add(f.wrapped)
    sewing.Perform()
    sewn = sewing.SewedShape()

    explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    if not explorer.More():
        raise UnsupportedGeometry("Rebuilt faces did not close into a shell.")
    maker = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(explorer.Current()))
    fixer = ShapeFix_Solid(maker.Solid())
    fixer.Perform()
    solid = cq.Solid(fixer.Solid())
    if not solid.isValid() or solid.Volume() <= 0:
        raise UnsupportedGeometry("Rebuilt solid is not valid.")
    return solid
