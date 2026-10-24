"""Split a mesh into planar regions and fit a plane to each.

Region growing works on triangles: a region starts at the flattest unassigned triangle and
absorbs neighbours whose normal agrees with the region plane and whose vertices lie within the
fit tolerance of it. Regions smaller than the level's minimum feature size are noise and get
merged into the neighbour they share the longest boundary with.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .tolerance import ToleranceLevel

# A triangle joins a region only if its normal is within this angle of the region normal.
# Generous on purpose: the distance-to-plane test is what actually bounds the fit.
MAX_GROW_ANGLE_DEG = 25.0
SMOOTHING_ROUNDS = 8
# Neighbours whose normals differ by more than this are across an edge: not averaged.
SMOOTHING_KEEP_DEG = 60.0
# Plane fits are redone without points this many median deviations off, this many times.
OUTLIER_MADS = 4.0
PLANE_REFITS = 2


@dataclass
class Plane:
    normal: np.ndarray
    """Unit normal, pointing out of the material."""
    offset: float
    """Signed distance: points p on the plane satisfy normal . p == offset."""

    def distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal - self.offset


@dataclass
class Region:
    id: int
    faces: np.ndarray
    plane: Plane
    area: float
    rms_mm: float = 0.0
    max_dev_mm: float = 0.0
    bow_mm: float = 0.0
    """RMS distance from the plane leaving out stray points (a seam, a sliver of the next
    face): how far the face itself bows."""
    neighbours: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class Geo:
    """Plain-array snapshot of a mesh. trimesh re-hashes the mesh on every cached-property
    access, which dominates run time in loops that touch thousands of regions."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    area_faces: np.ndarray
    face_adjacency: np.ndarray
    face_adjacency_edges: np.ndarray

    @classmethod
    def of(cls, mesh: trimesh.Trimesh | Geo) -> Geo:
        if isinstance(mesh, Geo):
            return mesh
        return cls(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            np.asarray(mesh.face_normals),
            np.asarray(mesh.area_faces),
            np.asarray(mesh.face_adjacency),
            np.asarray(mesh.face_adjacency_edges),
        )


def fit_plane(points: np.ndarray, weights: np.ndarray | None = None) -> Plane:
    """Least-squares plane through points (optionally weighted)."""
    w = np.ones(len(points)) if weights is None else weights
    centroid = (points * w[:, None]).sum(axis=0) / w.sum()
    centred = (points - centroid) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    return Plane(normal / np.linalg.norm(normal), float(normal @ centroid))


def oriented_plane(mesh: Geo | trimesh.Trimesh, faces: np.ndarray, robust: bool = False) -> Plane:
    """Fit a plane to the given faces, with its normal agreeing with the face normals.

    robust is for a region's final plane: each vertex counts for the area around it, so
    finely tessellated detail (embossed text, a dense patch of scan) does not outweigh the
    broad faces of the plane, and the fit is redone without outliers (text on the face, the
    edge of a fillet it was grown into). Region growing keeps the plain fit it was tuned on."""
    if robust:
        pts, weights = _weighted_vertices(mesh, faces)
    else:
        pts = mesh.vertices[np.unique(mesh.faces[faces])]
        weights = np.ones(len(pts))
    plane = fit_plane(pts, weights)
    for _ in range(PLANE_REFITS if robust else 0):
        keep = _inliers(plane.distance(pts), weights)
        if keep is None:
            break
        plane = fit_plane(pts[keep], weights[keep])
    mean_normal = (mesh.face_normals[faces] * mesh.area_faces[faces, None]).sum(axis=0)
    if plane.normal @ mean_normal < 0:
        plane = Plane(-plane.normal, -plane.offset)
    return plane


def plane_offset(mesh: Geo | trimesh.Trimesh, faces: np.ndarray, normal: np.ndarray) -> float:
    """Offset of the plane with the given normal through these faces, weighted and with
    outliers left out as in oriented_plane."""
    pts, weights = _weighted_vertices(mesh, faces)
    d = pts @ normal
    keep = _inliers(d - np.average(d, weights=weights), weights)
    if keep is not None:
        d, weights = d[keep], weights[keep]
    return float(np.average(d, weights=weights))


def _weighted_vertices(mesh: Geo | trimesh.Trimesh, faces: np.ndarray):
    corners = mesh.faces[faces]
    verts, idx = np.unique(corners, return_inverse=True)
    weights = np.bincount(idx.ravel(), np.repeat(mesh.area_faces[faces] / 3, 3), len(verts))
    return mesh.vertices[verts], weights + 1e-12


def _inliers(d: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Points within a few median deviations of the median, or None if that leaves out
    nothing, or too much to trust."""
    mid = _weighted_median(d, weights)
    spread = _weighted_median(np.abs(d - mid), weights)
    keep = np.abs(d - mid) <= OUTLIER_MADS * spread + 1e-3
    if keep.all() or keep.sum() < 3 or weights[keep].sum() < 0.5 * weights.sum():
        return None
    return keep


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def segment_planes(mesh: trimesh.Trimesh, level: ToleranceLevel) -> list[Region]:
    mesh = Geo.of(mesh)
    labels = _grow_regions(mesh, level)
    labels = _merge_small_regions(mesh, labels, level)
    labels = _merge_similar_regions(mesh, labels, level)
    labels = _merge_islands(mesh, labels, level)
    return _build_regions(mesh, labels)


def face_labels(regions: list[Region], face_count: int) -> np.ndarray:
    labels = np.full(face_count, -1, dtype=int)
    for r in regions:
        labels[r.faces] = r.id
    return labels


def face_neighbours(mesh: Geo) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nbrs[a].append(int(b))
        nbrs[b].append(int(a))
    return nbrs


def smoothed_normals(mesh: Geo, rounds: int = SMOOTHING_ROUNDS) -> np.ndarray:
    """Edge-preserving (bilateral) smoothing of face normals, so scanner noise does not break
    up flat areas while sharp edges stay sharp. Only used for the angle test; distances always
    use the raw scan."""
    normals = mesh.face_normals.copy()
    area = mesh.area_faces[:, None]
    pairs = mesh.face_adjacency
    cos_keep = np.cos(np.radians(SMOOTHING_KEEP_DEG))
    for _ in range(rounds):
        keep = ((normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(axis=1) > cos_keep)[:, None]
        acc = normals * area
        np.add.at(acc, pairs[:, 0], normals[pairs[:, 1]] * area[pairs[:, 1]] * keep)
        np.add.at(acc, pairs[:, 1], normals[pairs[:, 0]] * area[pairs[:, 0]] * keep)
        normals = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    return normals


def _grow_regions(mesh: Geo, level: ToleranceLevel) -> np.ndarray:
    tri_verts = mesh.vertices[mesh.faces]  # (F, 3, 3)
    nbrs = face_neighbours(mesh)
    normals = smoothed_normals(mesh)
    cos_max = np.cos(np.radians(MAX_GROW_ANGLE_DEG))
    tol = level.fit_tolerance_mm

    # Seed from the flattest triangles: least normal variation against their neighbours.
    flatness = np.array(
        [
            np.mean([1.0 - normals[f] @ normals[n] for n in nb]) if nb else 1.0
            for f, nb in enumerate(nbrs)
        ]
    )
    order = np.argsort(flatness, kind="stable")

    labels = np.full(len(mesh.faces), -1, dtype=int)
    next_label = 0
    for seed in order:
        if labels[seed] != -1:
            continue
        label = next_label
        next_label += 1
        labels[seed] = label
        members = [int(seed)]
        plane = Plane(normals[seed], float(normals[seed] @ tri_verts[seed].mean(axis=0)))
        refit_at = 8
        queue = deque(nbrs[seed])
        while queue:
            f = queue.popleft()
            if labels[f] != -1:
                continue
            if normals[f] @ plane.normal < cos_max:
                continue
            if np.abs(plane.distance(tri_verts[f])).max() > tol:
                continue
            labels[f] = label
            members.append(f)
            queue.extend(n for n in nbrs[f] if labels[n] == -1)
            if len(members) >= refit_at:
                plane = oriented_plane(mesh, np.array(members))
                refit_at *= 2
    return labels


def _region_adjacency(mesh: Geo, labels: np.ndarray) -> dict[tuple[int, int], float]:
    """Shared boundary length between each pair of adjacent regions."""
    shared: dict[tuple[int, int], float] = {}
    a = labels[mesh.face_adjacency[:, 0]]
    b = labels[mesh.face_adjacency[:, 1]]
    edges = mesh.vertices[mesh.face_adjacency_edges]
    lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    for la, lb, length in zip(a, b, lengths, strict=True):
        if la != lb:
            key = (min(la, lb), max(la, lb))
            shared[key] = shared.get(key, 0.0) + float(length)
    return shared


def _merge_small_regions(mesh: Geo, labels: np.ndarray, level: ToleranceLevel) -> np.ndarray:
    """Absorb regions below the minimum feature size into a neighbour: the one facing the same
    way if there is one (a split-off piece of the same face), else the longest-shared one."""
    labels = labels.copy()
    min_area = level.min_feature_mm**2
    cos_same = np.cos(np.radians(MAX_GROW_ANGLE_DEG))
    weighted = mesh.face_normals * mesh.area_faces[:, None]
    while True:
        areas = np.bincount(labels, weights=mesh.area_faces)
        small = {int(r) for r in np.flatnonzero((areas > 0) & (areas < min_area))}
        if not small:
            return labels
        normals = np.stack(
            [np.bincount(labels, weights=weighted[:, c], minlength=len(areas)) for c in range(3)],
            axis=1,
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        best: dict[int, tuple[bool, float, int]] = {}
        for (a, b), length in _region_adjacency(mesh, labels).items():
            for r, other in ((a, b), (b, a)):
                if r in small:
                    score = (bool(normals[r] @ normals[other] >= cos_same), length, other)
                    if score > best.get(r, (False, 0.0, -1)):
                        best[r] = score
        if not best:
            return labels
        # Smallest first, each into its neighbour's current label, so chains resolve in one pass.
        target = np.arange(labels.max() + 1)
        for r in sorted(best, key=lambda r: areas[r]):
            dest = best[r][2]
            while target[dest] != dest:
                dest = target[dest]
            if dest != r:
                target[r] = dest
        for r in range(len(target)):
            root = r
            while target[root] != root:
                root = target[root]
            target[r] = root
        labels = target[labels]


def _merge_similar_regions(mesh: Geo, labels: np.ndarray, level: ToleranceLevel) -> np.ndarray:
    """Merge neighbours that are really one face split by noise: a single plane through both
    fits within tolerance. This is the noise-versus-feature test; a real edge fails it."""
    labels = labels.copy()
    cos_crease = np.cos(np.radians(MAX_GROW_ANGLE_DEG))
    tol = level.fit_tolerance_mm

    faces_of = {int(r): np.flatnonzero(labels == r) for r in np.unique(labels)}
    planes = {r: oriented_plane(mesh, f) for r, f in faces_of.items()}
    rejected: set[tuple[int, int]] = set()

    while True:
        merged = False
        touched: set[int] = set()
        pairs = sorted(_region_adjacency(mesh, labels).items(), key=lambda kv: -kv[1])
        for (a, b), _ in pairs:
            if a in touched or b in touched or (a, b) in rejected:
                continue
            if planes[a].normal @ planes[b].normal < cos_crease:
                rejected.add((a, b))
                continue
            faces = np.concatenate([faces_of[a], faces_of[b]])
            pts = mesh.vertices[np.unique(mesh.faces[faces])]
            if np.percentile(np.abs(fit_plane(pts).distance(pts)), 99) > tol:
                rejected.add((a, b))
                continue
            labels[faces_of[b]] = a
            faces_of[a] = faces
            del faces_of[b], planes[b]
            planes[a] = oriented_plane(mesh, faces)
            # a's plane moved: its earlier rejections no longer hold.
            rejected = {pair for pair in rejected if a not in pair}
            touched.update((a, b))
            merged = True
        if not merged:
            return labels


def _merge_islands(mesh: Geo, labels: np.ndarray, level: ToleranceLevel) -> np.ndarray:
    """A real face of a closed solid meets at least 3 others; fewer means a noise patch, which
    joins the neighbour it most nearly continues. Only if one plane still fits both, though:
    a strip of a hole wall also has few neighbours, and is left for the cylinder finder."""
    labels = labels.copy()
    tol = level.fit_tolerance_mm
    kept: set[int] = set()
    while True:
        shared = _region_adjacency(mesh, labels)
        counts: dict[int, list[tuple[float, int]]] = {}
        for (a, b), length in shared.items():
            counts.setdefault(a, []).append((length, b))
            counts.setdefault(b, []).append((length, a))
        areas = np.bincount(labels, weights=mesh.area_faces)
        islands = [r for r, nb in counts.items() if len(nb) < 3 and r not in kept]
        if not islands or len(counts) <= 4:
            return labels
        r = min(islands, key=lambda r: areas[r])
        mine_faces = np.flatnonzero(labels == r)
        mine = oriented_plane(mesh, mine_faces).normal
        options = sorted(
            counts[r],
            key=lambda nb: -(mine @ oriented_plane(mesh, np.flatnonzero(labels == nb[1])).normal),
        )
        target = None
        for _, nb in options:
            faces = np.concatenate([mine_faces, np.flatnonzero(labels == nb)])
            pts = mesh.vertices[np.unique(mesh.faces[faces])]
            if np.percentile(np.abs(fit_plane(pts).distance(pts)), 99) <= tol:
                target = nb
                break
        if target is None and areas[r] < level.min_feature_mm**2:
            target = options[0][1]
        if target is None:
            kept.add(r)
            continue
        labels[mine_faces] = target


def _build_regions(mesh: Geo, labels: np.ndarray) -> list[Region]:
    # Renumber densely so region ids are 0..n-1.
    _, dense = np.unique(labels, return_inverse=True)
    regions: list[Region] = []
    for rid in range(dense.max() + 1):
        faces = np.flatnonzero(dense == rid)
        plane = oriented_plane(mesh, faces, robust=True)
        regions.append(Region(rid, faces, plane, float(mesh.area_faces[faces].sum())))
    for a, b in _region_adjacency(mesh, dense):
        regions[a].neighbours.add(b)
        regions[b].neighbours.add(a)
    update_deviation(mesh, regions)
    return regions


def update_deviation(mesh: trimesh.Trimesh, regions: list[Region]) -> None:
    for r in regions:
        d = r.plane.distance(mesh.vertices[np.unique(mesh.faces[r.faces])])
        r.rms_mm = float(np.sqrt(np.mean(d**2)))
        r.max_dev_mm = float(np.abs(d).max())
        keep = _inliers(d, np.ones(len(d)))
        r.bow_mm = r.rms_mm if keep is None else float(np.sqrt(np.mean(d[keep] ** 2)))
