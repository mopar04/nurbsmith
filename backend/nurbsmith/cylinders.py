"""Find cylindrical surfaces (holes, bosses, fillets) among the planar regions.

Region growing with a plane tolerance breaks a round surface into narrow flat strips that meet
at shallow angles. Two neighbouring strips give a first guess at the cylinder they are chords
of; the cylinder then grows over every connected triangle that lies on it within tolerance and
faces the right way, refitting as it goes. Growing over triangles rather than whole strips
matters on real scans, where strips are ragged and often take in a bit of the next face.
Afterwards the planar regions are rebuilt from whatever triangles the cylinders left.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .segment import (
    Geo,
    Region,
    face_labels,
    fit_plane,
    oriented_plane,
    smoothed_normals,
    update_deviation,
)
from .tolerance import ToleranceLevel

# Strips of one cylinder all have normals perpendicular to its axis; allow for noise.
MAX_AXIS_TILT_DEG = 15.0
# A triangle joins a cylinder only if its (smoothed) normal is within this angle of the
# cylinder's normal there, so the cylinder does not creep onto a plane it meets.
MAX_NORMAL_DEG = 30.0
# Coverage around the axis (fraction of 10-degree sectors with scan points) that counts as a
# full hole or boss rather than a fillet.
FULL_COVERAGE = 0.85
COVERAGE_SECTORS = 36
# Fit residual at this percentile must be within tolerance.
FIT_PERCENTILE = 99
MAX_FIT_POINTS = 1500
GROW_ROUNDS = 4
# Fewest points a cylinder is fitted to or accepted from.
MIN_FIT_POINTS = 12
# The normals must turn through between these shares of the arc the points span.
NORMAL_TURN_SHARE = 0.5
NORMAL_TURN_MAX = 2.0
# Above this share of the part's size, a radius spanning under this coverage is a bowed wall.
WARP_RADIUS_SHARE = 0.25
WARP_MAX_COVERAGE = 0.15
# A part-way curve must have at least this share of its area facing the middle of its turn.
MIN_MIDDLE = 0.15
# On a cylinder each face's normal points straight out from the axis. A run of flat faces and
# tight corners (a pocket end, a cone, a corner blend) can lie near a cylinder while its normals
# lag behind; the steady lag, as a share of the arc the patch spans, measured in slices.
MAX_NORMAL_LAG = 0.12
LAG_SLICES = 6


@dataclass
class Cylinder:
    point: np.ndarray
    """A point on the axis."""
    axis: np.ndarray
    """Unit axis direction."""
    radius: float
    convex: bool = False

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from the surface: positive outside the cylinder."""
        rel = points - self.point
        radial = rel - np.outer(rel @ self.axis, self.axis)
        return np.linalg.norm(radial, axis=1) - self.radius

    def radial(self, points: np.ndarray) -> np.ndarray:
        """Unit vectors from the axis out to each point."""
        rel = points - self.point
        radial = rel - np.outer(rel @ self.axis, self.axis)
        return radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)


@dataclass
class CurvedPatch:
    id: int
    faces: np.ndarray
    cylinder: Cylinder
    convex: bool
    """Material inside the cylinder (a boss, a rounded outside edge) rather than outside."""
    coverage: float
    """Fraction of the way round the axis that the scan covers."""
    axial_range: tuple[float, float]
    """Extent along the axis, measured from cylinder.point."""
    rms_mm: float
    neighbours: set[int] = field(default_factory=set)
    """Ids of the planar regions this patch borders."""


def fit_cylinder(points: np.ndarray, start: Cylinder) -> Cylinder:
    """Least-squares cylinder through points, refined from a starting estimate."""
    axis, centre = start.axis, start.point
    u, v = _basis(axis)
    if len(points) > MAX_FIT_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), MAX_FIT_POINTS, False)]

    # Parametrise the axis by a tilt (a, b) of the start direction and an offset (x, y) in the
    # start cross-section plane, so the fit has no gauge freedom.
    def residuals(params: np.ndarray) -> np.ndarray:
        a, b, x, y, radius = params
        d = axis + a * u + b * v
        d /= np.linalg.norm(d)
        rr = points - (centre + x * u + y * v)
        radial = rr - np.outer(rr @ d, d)
        return np.linalg.norm(radial, axis=1) - radius

    if len(points) < MIN_FIT_POINTS:
        return start
    sol = least_squares(residuals, [0.0, 0.0, 0.0, 0.0, start.radius], method="lm")
    a, b, x, y, radius = sol.x
    d = axis + a * u + b * v
    return Cylinder(centre + x * u + y * v, d / np.linalg.norm(d), abs(float(radius)), start.convex)


def fit_circle_on_axis(points: np.ndarray, axis: np.ndarray, start: Cylinder) -> Cylinder:
    """Least-squares cylinder with a fixed axis direction: only its position and radius move."""
    u, v = _basis(axis)
    if len(points) > MAX_FIT_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), MAX_FIT_POINTS, False)]
    if len(points) < MIN_FIT_POINTS:
        return start
    xy = np.stack([points @ u, points @ v], axis=1)
    c0 = np.array([start.point @ u, start.point @ v])

    def residuals(params: np.ndarray) -> np.ndarray:
        return np.linalg.norm(xy - params[:2], axis=1) - params[2]

    x, y, radius = least_squares(residuals, [*c0, start.radius], method="lm").x
    along = start.point @ axis
    return Cylinder(x * u + y * v + along * axis, axis, abs(float(radius)), start.convex)


def _seed_cylinder(a: Region, b: Region, ca: np.ndarray, cb: np.ndarray) -> Cylinder | None:
    """The cylinder two adjacent strips are chords of, from their planes and centroids.

    Each strip's centroid sits roughly one radius from the axis along its normal (inward for a
    hole, outward for a boss), which pins down a centre and radius without a fragile circle fit
    through two thin bands of noisy points.
    """
    na, nb = a.plane.normal, b.plane.normal
    axis = np.cross(na, nb)
    if np.linalg.norm(axis) < 0.05 or na @ nb < -0.2:  # parallel, or a sharp fold
        return None
    axis /= np.linalg.norm(axis)
    # Convex if b falls away behind a's plane.
    convex = a.plane.distance(cb[None])[0] < 0
    sign = 1.0 if convex else -1.0
    # centre = ca - sign*r*na = cb - sign*r*nb (in the cross-section), least squares for r.
    diff = cb - ca
    diff -= (diff @ axis) * axis
    dn = sign * (nb - na)
    radius = float(diff @ dn / (dn @ dn))
    if radius <= 0:
        return None
    return Cylinder(ca - sign * radius * na, axis, radius, convex)


def find_cylinders(
    mesh: Geo, regions: list[Region], level: ToleranceLevel
) -> tuple[np.ndarray, list[CurvedPatch]]:
    """Find cylindrical patches, seeded from pairs of planar strips and grown over triangles.
    Returns, per face, the patch id or -1 if the face stays planar, and the patches."""
    tol = level.fit_tolerance_mm
    by_id = {r.id: r for r in regions}
    centroids = {r.id: mesh.vertices[np.unique(mesh.faces[r.faces])].mean(axis=0) for r in regions}
    normals = smoothed_normals(mesh)
    centres = mesh.vertices[mesh.faces].mean(axis=1)
    adjacency = _adjacency_matrix(mesh)
    cos_normal = np.cos(np.radians(MAX_NORMAL_DEG))
    min_area = level.min_feature_mm**2

    patch_of = np.full(len(mesh.faces), -1, dtype=int)
    patches: list[CurvedPatch] = []

    def points(faces: np.ndarray) -> np.ndarray:
        return mesh.vertices[np.unique(mesh.faces[faces])]

    def within(cyl: Cylinder, pts: np.ndarray) -> bool:
        return bool(np.percentile(np.abs(cyl.distance(pts)), FIT_PERCENTILE) <= tol)

    def grow(cyl: Cylinder, seed_faces: np.ndarray) -> tuple[Cylinder, np.ndarray]:
        """Triangles connected to the seed that lie on the cylinder and face the right way,
        refitting a few times as the patch grows."""
        sign = 1.0 if cyl.convex else -1.0
        faces = seed_faces
        for _ in range(GROW_ROUNDS):
            on_surface = (np.abs(cyl.distance(mesh.vertices)) <= tol)[mesh.faces].all(axis=1)
            facing = sign * (cyl.radial(centres) * normals).sum(axis=1) >= cos_normal
            ok = np.flatnonzero(on_surface & facing & (patch_of == -1))
            _, comp = connected_components(adjacency[ok][:, ok], directed=False)
            local = np.searchsorted(ok, seed_faces)
            hit = local < len(ok)
            hit[hit] = ok[local[hit]] == seed_faces[hit]
            if not hit.any():
                return cyl, np.array([], dtype=int)
            faces = ok[np.isin(comp, comp[local[hit]])]
            cyl = fit_cylinder(points(faces), cyl)
        return cyl, faces

    # Seed from the narrowest pairs: strips of a tight curve are small.
    pairs = sorted(
        {(min(a, b), max(a, b)) for a in by_id for b in by_id[a].neighbours},
        key=lambda p: max(by_id[p[0]].area, by_id[p[1]].area),
    )
    for a, b in pairs:
        seed_faces = np.concatenate([by_id[a].faces, by_id[b].faces])
        seed_faces = np.sort(seed_faces[patch_of[seed_faces] == -1])
        if mesh.area_faces[seed_faces].sum() < 0.5 * (by_id[a].area + by_id[b].area):
            continue  # mostly taken by an earlier patch
        seed = _seed_cylinder(by_id[a], by_id[b], centroids[a], centroids[b])
        if seed is None:
            continue
        cyl = fit_cylinder(points(seed_faces), seed)
        if not within(cyl, points(seed_faces)):
            continue
        cyl, faces = grow(cyl, seed_faces)
        if mesh.area_faces[faces].sum() < min_area or not within(cyl, points(faces)):
            continue
        # A curve gentle enough to fit a plane is a plane.
        pts = points(faces)
        if np.percentile(np.abs(fit_plane(pts).distance(pts)), FIT_PERCENTILE) <= tol:
            continue
        if len(pts) < MIN_FIT_POINTS:
            continue
        patch = _describe(mesh, len(patches), faces, cyl, normals)
        if patch is None:
            continue
        patch_of[faces] = patch.id
        patches.append(patch)
    return patch_of, patches


def _adjacency_matrix(mesh: Geo) -> csr_matrix:
    n = len(mesh.faces)
    a, b = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    data = np.ones(len(a), dtype=bool)
    return csr_matrix((data, (a, b)), shape=(n, n))


def _describe(
    mesh: Geo, pid: int, faces: np.ndarray, cyl: Cylinder, normals: np.ndarray | None = None
) -> CurvedPatch | None:
    verts = np.unique(mesh.faces[faces])
    pts = mesh.vertices[verts]
    centres = mesh.vertices[mesh.faces[faces]].mean(axis=1)
    outward = (mesh.face_normals[faces] * cyl.radial(centres)).sum(axis=1)
    convex = bool(np.average(outward, weights=mesh.area_faces[faces]) > 0)

    u, v = _basis(cyl.axis)
    rel = pts - cyl.point
    angles = np.arctan2(rel @ v, rel @ u)
    sectors = np.unique(((angles + np.pi) / (2 * np.pi) * COVERAGE_SECTORS).astype(int))
    coverage = len(sectors) / COVERAGE_SECTORS
    # A pair of strips spanning almost no angle is just two planes meeting at a shallow crease.
    if coverage < 2 / COVERAGE_SECTORS:
        return None
    # A sliver of a cylinder much bigger than the part is a flat wall that bowed (a print that
    # warped, a panel that sagged); it stays a plane.
    size = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    if cyl.radius > WARP_RADIUS_SHARE * size and coverage < WARP_MAX_COVERAGE:
        return None
    # On a real cylinder the surface turns as far as the arc does. A wide, gentle "cylinder"
    # fitted to a few big flat triangles spans an arc while its normals barely turn.
    n = normals[faces] if normals is not None else mesh.face_normals[faces]
    turn = np.arctan2(n @ v, n @ u)
    turned = np.unique(((turn + np.pi) / (2 * np.pi) * COVERAGE_SECTORS).astype(int))
    # The other way round, normals turning far more than the arc, is a corner with flat sides
    # (two faces and the fillet between them) passing for one wide, gentle cylinder.
    # Only with smoothed normals: raw ones on a noisy scan point every which way.
    most = NORMAL_TURN_MAX * len(sectors) + 2 if normals is not None else COVERAGE_SECTORS
    if not NORMAL_TURN_SHARE * len(sectors) <= len(turned) <= most:
        return None
    # A curve turns steadily; a crease (a wall meeting a chamfer) jumps from one direction to
    # the other, with hardly any surface facing in between. This and the lag check need
    # smoothed normals too, and a full circle needs neither.
    partial = normals is not None and coverage < FULL_COVERAGE
    if partial and _middle_share(turn, mesh.area_faces[faces]) < MIN_MIDDLE:
        return None
    if partial and _normal_lag(cyl, centres, n, convex, mesh.area_faces[faces]) > (
        MAX_NORMAL_LAG * _arc(angles)
    ):
        return None

    t = rel @ cyl.axis
    rms = float(np.sqrt(np.mean(cyl.distance(pts) ** 2)))
    return CurvedPatch(pid, faces, cyl, convex, coverage, (float(t.min()), float(t.max())), rms)


def _normal_lag(
    cyl: Cylinder, centres: np.ndarray, normals: np.ndarray, convex: bool, weights: np.ndarray
) -> float:
    """Largest steady lag, in turns, of the face normals behind the radial direction.

    The patch is cut into slices along its arc and the signed lag averaged in each: noise
    averages out, while a flat stretch (whose normal stays put as the arc moves on) does not."""
    u, v = _basis(cyl.axis)
    rel = centres - cyl.point
    pos = np.arctan2(rel @ v, rel @ u)
    lag = np.arctan2(normals @ v, normals @ u) + (0.0 if convex else np.pi) - pos
    lag = (lag + np.pi) % (2 * np.pi) - np.pi
    mid = np.arctan2(
        np.average(np.sin(pos), weights=weights), np.average(np.cos(pos), weights=weights)
    )
    pos = (pos - mid + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.percentile(pos, [2, 98])
    slices = np.clip(((pos - lo) / max(hi - lo, 1e-9) * LAG_SLICES).astype(int), 0, LAG_SLICES - 1)
    worst = 0.0
    for k in range(LAG_SLICES):
        here = slices == k
        if here.sum() > 3:
            worst = max(worst, abs(float(np.average(lag[here], weights=weights[here]))))
    return worst / (2 * np.pi)


def _arc(angles: np.ndarray) -> float:
    """Arc the angles span, in turns."""
    mid = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    rel = (angles - mid + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.percentile(rel, [2, 98])
    return float(hi - lo) / (2 * np.pi)


def _middle_share(angles: np.ndarray, weights: np.ndarray) -> float:
    """Share of the weight facing the middle third of the angles' range."""
    mean = np.arctan2(np.average(np.sin(angles), weights=weights),
                      np.average(np.cos(angles), weights=weights))  # fmt: skip
    rel = (angles - mean + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.percentile(rel, [2, 98])
    middle = np.abs(rel - (lo + hi) / 2) <= (hi - lo) / 6
    return float(weights[middle].sum() / weights.sum())


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit: a good start for the geometric fit."""
    a = np.column_stack([xy, np.ones(len(xy))])
    b = (xy**2).sum(axis=1)
    (p, q, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = p / 2, q / 2
    return float(cx), float(cy), float(np.sqrt(max(c + cx**2 + cy**2, 1e-12)))


def planes_around(
    mesh: Geo,
    regions: list[Region],
    patch_of: np.ndarray,
    patches: list[CurvedPatch],
    level: ToleranceLevel,
) -> tuple[list[Region], np.ndarray]:
    """Planar regions rebuilt from the triangles the cylinders left, as (regions, plane_of).

    Taking a cylinder's triangles out can leave pieces of the strips it was found from; those
    join the cylinder or a neighbouring plane. plane_of gives each face's region id, or -1 for
    faces on a cylinder.
    """
    labels = face_labels(regions, len(mesh.faces))
    labels[patch_of >= 0] = -1
    # Each leftover piece becomes its own candidate region.
    keep = np.flatnonzero(labels >= 0)
    same = labels[mesh.face_adjacency[:, 0]] == labels[mesh.face_adjacency[:, 1]]
    pairs = mesh.face_adjacency[same & (labels[mesh.face_adjacency[:, 0]] >= 0)]
    n = len(mesh.faces)
    graph = csr_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    piece = np.full(n, -1)
    _, piece[keep] = np.unique(comp[keep], return_inverse=True)

    # Leftovers merge into a neighbour; patches are labelled after the pieces. A piece goes
    # into a cylinder it lies on, or into the only plane it touches (an island), or, if it is
    # too small to be a feature, into the neighbour it shares the most boundary with. A flat
    # piece ringed by one cylinder is kept: it is the top of a boss or the floor of a hole.
    offset = piece.max() + 1
    combined = np.where(patch_of >= 0, offset + patch_of, piece)
    min_area = level.min_feature_mm**2
    tol = level.fit_tolerance_mm
    while True:
        areas = np.bincount(combined, weights=mesh.area_faces)
        touching: dict[int, list[tuple[float, int]]] = {}
        for (a, b), length in shared_lengths(mesh, combined).items():
            touching.setdefault(a, []).append((length, b))
            touching.setdefault(b, []).append((length, a))
        moves: dict[int, int] = {}
        for i in sorted(touching, key=lambda i: areas[i]):
            if i >= offset or i in moves.values():
                continue
            nbrs = touching[i]
            target = None
            if len(nbrs) == 1 and nbrs[0][1] < offset:
                target = nbrs[0][1]
            else:
                # Triangle centres too: a boss top tessellated as a fan has all its corners
                # on the rim, so on the cylinder.
                faces = combined == i
                pts = np.vstack(
                    [
                        mesh.vertices[np.unique(mesh.faces[faces])],
                        mesh.vertices[mesh.faces[faces]].mean(axis=1),
                    ]
                )
                for _, j in sorted(nbrs, reverse=True):
                    if j >= offset:
                        cyl = patches[j - offset].cylinder
                        if np.percentile(np.abs(cyl.distance(pts)), FIT_PERCENTILE) <= tol:
                            target = j
                            break
                if target is None and areas[i] < min_area:
                    target = max(nbrs)[1]
            if target is not None and target not in moves:
                moves[i] = target
        if not moves:
            break
        for i, j in moves.items():
            combined[combined == i] = j

    patch_of[:] = np.where(combined >= offset, combined - offset, -1)
    plane_ids = np.unique(combined[combined < offset])
    plane_of = np.full(n, -1)
    out = []
    for new_id, old in enumerate(plane_ids):
        faces = np.flatnonzero(combined == old)
        plane_of[faces] = new_id
        out.append(
            Region(
                new_id,
                faces,
                oriented_plane(mesh, faces, robust=True),
                float(mesh.area_faces[faces].sum()),
            )
        )
    update_deviation(mesh, out)
    return out, plane_of


def link(mesh: Geo, regions: list[Region], patches: list[CurvedPatch]) -> np.ndarray:
    """Number planes 0..P-1 and patches after them, set every neighbour set, and return each
    face's label in that shared numbering."""
    labels = np.full(len(mesh.faces), -1)
    for i, r in enumerate(regions):
        r.id = i
        labels[r.faces] = i
    for j, p in enumerate(patches):
        p.id = len(regions) + j
        labels[p.faces] = p.id
    for item in [*regions, *patches]:
        item.neighbours = set()
    by_id = {x.id: x for x in [*regions, *patches]}
    for a, b in shared_lengths(mesh, labels):
        if a >= 0 and b >= 0:
            by_id[a].neighbours.add(b)
            by_id[b].neighbours.add(a)
    return labels


def shared_lengths(mesh: Geo, labels: np.ndarray) -> dict[tuple[int, int], float]:
    a = labels[mesh.face_adjacency[:, 0]]
    b = labels[mesh.face_adjacency[:, 1]]
    edges = mesh.vertices[mesh.face_adjacency_edges]
    lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    differ = a != b
    shared: dict[tuple[int, int], float] = {}
    for la, lb, length in zip(a[differ], b[differ], lengths[differ], strict=True):
        key = (int(min(la, lb)), int(max(la, lb)))
        shared[key] = shared.get(key, 0.0) + float(length)
    return shared


def consolidate(
    mesh: Geo, planes: list[Region], patches: list[CurvedPatch], level: ToleranceLevel
) -> tuple[list[Region], list[CurvedPatch], np.ndarray]:
    """Tidy the patches up, then link everything; returns (planes, patches, labels).

    Two cleanups, repeated until nothing changes: neighbouring patches that one cylinder fits
    become one patch (noise can split a fillet in two), and a plane strip that one cylinder
    fits together with a neighbouring patch becomes part of it (on a noisy scan a wide strip
    of a fillet can look flat enough to stay a plane).
    """
    tol = level.fit_tolerance_mm
    sin_side = np.sin(np.radians(MAX_AXIS_TILT_DEG))
    labels = link(mesh, planes, patches)

    def points(faces: np.ndarray) -> np.ndarray:
        return mesh.vertices[np.unique(mesh.faces[faces])]

    changed = True
    while changed:
        changed = False
        by_id = {x.id: x for x in [*planes, *patches]}

        for p in patches:
            for q in [by_id[n] for n in p.neighbours]:
                if not isinstance(q, CurvedPatch) or q.convex != p.convex:
                    continue
                if abs(p.cylinder.axis @ q.cylinder.axis) < np.cos(np.radians(MAX_AXIS_TILT_DEG)):
                    continue
                faces = np.concatenate([p.faces, q.faces])
                cyl = fit_cylinder(points(faces), p.cylinder)
                if np.percentile(np.abs(cyl.distance(points(faces))), FIT_PERCENTILE) > tol:
                    continue
                merged = _describe(mesh, p.id, faces, cyl)
                if merged is None:
                    continue
                patches = [x for x in patches if x is not p and x is not q] + [merged]
                changed = True
                break
            if changed:
                break
        if changed:
            labels = link(mesh, planes, patches)
            continue

        for p in patches:
            sides = [
                by_id[n]
                for n in p.neighbours
                if isinstance(by_id[n], Region)
                and abs(by_id[n].plane.normal @ p.cylinder.axis) <= sin_side
            ]
            for side in sorted(sides, key=lambda r: r.area):
                faces = np.concatenate([p.faces, side.faces])
                cyl = fit_cylinder(points(faces), p.cylinder)
                if np.percentile(np.abs(cyl.distance(points(faces))), FIT_PERCENTILE) > tol:
                    continue
                patch = _describe(mesh, p.id, faces, cyl)
                if patch is None:
                    continue
                patches = [x for x in patches if x is not p] + [patch]
                planes = [r for r in planes if r is not side]
                changed = True
                break
            if changed:
                break
        if changed:
            labels = link(mesh, planes, patches)

    # Where fillets meet at a corner the scan has a small blended patch that is neither plane
    # nor cylinder. Filleting the edges in CAD recreates it, so it just joins a fillet.
    by_id = {x.id: x for x in [*planes, *patches]}
    shared = shared_lengths(mesh, labels)
    corners = []
    for r in planes:
        round_nbrs = [by_id[n] for n in r.neighbours if isinstance(by_id[n], CurvedPatch)]
        if len(round_nbrs) < 2 or len(round_nbrs) < len(r.neighbours) - 1:
            continue
        # A flat ring between holes or bosses (a counterbore's floor) is a face of its own.
        if any(p.coverage >= FULL_COVERAGE for p in round_nbrs):
            continue
        if r.area > sum((2 * p.cylinder.radius) ** 2 for p in round_nbrs):
            continue
        best = max(round_nbrs, key=lambda p: shared.get((min(r.id, p.id), max(r.id, p.id)), 0))
        best.faces = np.concatenate([best.faces, r.faces])
        corners.append(r)
    if corners:
        planes = [r for r in planes if not any(r is c for c in corners)]
        labels = link(mesh, planes, patches)
    return planes, patches, labels
