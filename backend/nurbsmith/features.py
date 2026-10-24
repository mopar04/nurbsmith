"""Turn cylindrical patches into CAD features: holes, slots, bosses and fillets, plus the
narrow flat strips that are chamfers.

The base solid is rebuilt from the planes alone, as if every hole were filled, every boss cut
off and every fillet sharp. The features are then applied with ordinary CAD operations (fuse a
cylinder, fillet an edge, cut a cylinder), so the STEP file holds the same kind of geometry a
designer would have made, with clean sizes snapped to standard values.
"""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
import numpy as np
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from scipy.optimize import minimize_scalar

from .cylinders import FULL_COVERAGE, CurvedPatch, Cylinder, fit_circle_on_axis
from .segment import Geo, Plane, Region
from .tolerance import ToleranceLevel

# A plane is an end of a cylinder if its normal is within this angle of the axis, and a side
# (tangent to a fillet) if within this angle of perpendicular.
END_ANGLE_DEG = 15.0
SIDE_ANGLE_DEG = 15.0
# The two faces a fillet joins must meet at an angle (|cos| below this), not be parallel.
MAX_SIDE_DOT = np.cos(np.radians(20.0))
# Snap a size to a standard value if that moves the surface by at most this share of the
# fit tolerance.
SNAP_SHARE = 0.5
# Holes are stricter: a diameter moves by at most this share of the fit tolerance. Stamped,
# printed and cast holes are rarely drill sizes, and a size the user measures with calipers
# has to come out as measured.
HOLE_SNAP_SHARE = 0.35
# Cutting and fusing tools overshoot the faces they end on by this much, so no sliver is left.
OVERSHOOT_MM = 1.0
# Features that round or bevel the edge between two planes.
EDGE_KINDS = ("fillet", "sharp", "chamfer")
# A slot end is a half-round, concave patch: its coverage lies in this range.
SLOT_COVERAGE = (0.4, 0.65)
# Where rounded edges meet at a corner, the blend between them leaves small facets that no
# plane intersection can rebuild; up to this many min_feature squares they are left to the
# fillets.
BLEND_FACET_AREA = 4.0
# A chamfer is a flat strip between two faces, this many min_feature wide (narrower is a
# rounded or noisy edge) and at least this many times longer than wide.
CHAMFER_MIN_WIDTH = 0.5
CHAMFER_MAX_WIDTH = 2.0
CHAMFER_MIN_LENGTH = 3.0

# Common drill sizes (mm) and what they are for.
_METRIC = {
    "M2": (2.4, 2.2, 1.6),
    "M2.5": (2.9, 2.7, 2.05),
    "M3": (3.4, 3.2, 2.5),
    "M4": (4.5, 4.3, 3.3),
    "M5": (5.5, 5.3, 4.2),
    "M6": (6.6, 6.4, 5.0),
    "M8": (9.0, 8.4, 6.8),
    "M10": (11.0, 10.5, 8.5),
    "M12": (13.5, 13.0, 10.2),
    "M16": (17.5, 17.0, 14.0),
    "M20": (22.0, 21.0, 17.5),
}
_INCH = {
    "1/8": 1 / 8,
    "5/32": 5 / 32,
    "3/16": 3 / 16,
    "1/4": 1 / 4,
    "5/16": 5 / 16,
    "3/8": 3 / 8,
    "7/16": 7 / 16,
    "1/2": 1 / 2,
    "5/8": 5 / 8,
    "3/4": 3 / 4,
    "1": 1.0,
}
STANDARD_HOLES: list[tuple[float, str]] = sorted(
    [(d, f"{m} clearance") for m, (d, _, _) in _METRIC.items()]
    + [(d, f"{m} close fit") for m, (_, d, _) in _METRIC.items()]
    + [(d, f"{m} tap drill") for m, (_, _, d) in _METRIC.items()]
    + [(round(v * 25.4, 3), f'{k}"') for k, v in _INCH.items()]
)


@dataclass
class Feature:
    kind: str
    """"hole", "slot", "boss", "fillet", "chamfer", "body" for a round part's outside, or
    "sharp" for a fillet the accuracy level leaves out."""
    patch: int
    radius: float
    axis: np.ndarray
    start: np.ndarray
    """Axis point where the feature starts: the base face for a boss, the far end of a hole."""
    end: np.ndarray
    """Axis point where it ends: the boss top, or the face a hole is drilled from."""
    planes: tuple[int, ...] = ()
    """For a fillet or chamfer, the two planes it joins."""
    through: bool = False
    label: str | None = None
    """Standard size it was snapped to, if any."""
    span: np.ndarray | None = None
    """For a slot, from the centre of one round end to the other."""
    also: tuple[int, ...] = ()
    """Further patches the feature covers (a slot's second end)."""

    def as_dict(self) -> dict:
        out: dict = {"type": self.kind}
        if self.kind == "fillet":
            out["radius_mm"] = round(self.radius, 3)
        elif self.kind == "chamfer":
            out["size_mm"] = round(self.radius, 3)
        elif self.kind == "slot":
            out["width_mm"] = round(2 * self.radius, 3)
            out["length_mm"] = round(float(np.linalg.norm(self.span)) + 2 * self.radius, 3)
            length = float(np.linalg.norm(self.end - self.start))
            out.update({"through": True} if self.through else {"depth_mm": round(length, 3)})
        elif self.kind == "body":
            out["diameter_mm"] = round(2 * self.radius, 3)
            out["height_mm"] = round(float(np.linalg.norm(self.end - self.start)), 3)
        else:
            out["diameter_mm"] = round(2 * self.radius, 3)
            length = float(np.linalg.norm(self.end - self.start))
            if self.kind == "boss":
                out["height_mm"] = round(length, 3)
            elif self.through:
                out["through"] = True
            else:
                out["depth_mm"] = round(length, 3)
        if self.label:
            out["standard"] = self.label
        return out


def plan_features(
    mesh: Geo,
    planes: list[Region],
    patches: list[CurvedPatch],
    shared: dict[tuple[int, int], float],
    level: ToleranceLevel,
) -> tuple[list[Feature], set[int], str | None]:
    """Decide what each patch is. Returns the features, the planes that belong to a feature
    rather than to the base solid (the floor of a blind hole, the top of a boss), and a note
    on any curved areas left to the flat faces around them because they fit no feature."""
    tol = level.fit_tolerance_mm
    by_id = {r.id: r for r in planes}
    cos_end = np.cos(np.radians(END_ANGLE_DEG))
    sin_side = np.sin(np.radians(SIDE_ANGLE_DEG))
    directions = _directions(planes)
    features: list[Feature] = []
    caps: set[int] = set()
    skipped: list[str] = []
    holes = {p.id: p for p in patches if p.coverage >= FULL_COVERAGE and not p.convex}
    slotted: set[int] = set()
    for slot in _find_slots(mesh, by_id, patches, directions, level):
        features.append(slot)
        slotted.update((slot.patch, *slot.also))
        caps.update(slot.planes)

    for chamfer in _find_chamfers(by_id, shared, caps, level):
        features.append(chamfer)
        caps.add(chamfer.patch)

    for patch in patches:
        if patch.id in slotted:
            continue
        cyl = patch.cylinder
        nbrs = [by_id[n] for n in patch.neighbours if n in by_id]
        ends = [r for r in nbrs if abs(r.plane.normal @ cyl.axis) >= cos_end]
        sides = [r for r in nbrs if abs(r.plane.normal @ cyl.axis) <= sin_side]
        pts = mesh.vertices[np.unique(mesh.faces[patch.faces])]

        if patch.coverage >= FULL_COVERAGE:
            cyl = _snap_axis(cyl, directions, level, pts)
            cap_ends = [r for r in ends if r.neighbours <= {patch.id}]
            # A counterbore: the ring between it and the narrower hole below is its floor.
            cap_ends += [
                r
                for r in ends
                if r not in cap_ends
                and not patch.convex
                and r.neighbours <= set(holes)
                and any(holes[n].cylinder.radius < cyl.radius - tol for n in r.neighbours)
            ]
            open_ends = [r for r in ends if r not in cap_ends]
            kind = "boss" if patch.convex else "hole"
            # A round part (a puck, a tube, a turned spacer): the cylinder is the outside of
            # the whole part and every flat face is an end of it.
            if (
                patch.convex
                and len(ends) >= 2
                and all(abs(r.plane.normal @ cyl.axis) >= cos_end for r in planes)
            ):
                hits = [_axis_hit(cyl, r.plane) for r in ends]
                t = [(h - cyl.point) @ cyl.axis for h in hits]
                radius, label = _snap_radius(cyl.radius, tol, "boss")
                start, end = hits[int(np.argmin(t))], hits[int(np.argmax(t))]
                features.append(Feature("body", patch.id, radius, cyl.axis, start, end))
                continue
            if kind == "boss" and (len(cap_ends) != 1 or len(open_ends) != 1):
                skipped.append(f"a round boss \u00d8{2 * cyl.radius:.1f} mm without a flat top")
                continue
            if kind == "hole" and len(cap_ends) > 1:
                skipped.append("a closed round cavity")
                continue
            radius, label = _snap_radius(cyl.radius, tol, kind)
            if kind == "boss":
                start = _axis_hit(cyl, open_ends[0].plane)
                end = _axis_hit(cyl, cap_ends[0].plane)
                caps.add(cap_ends[0].id)
                features.append(
                    Feature("boss", patch.id, radius, cyl.axis, start, end, label=label)
                )
                continue
            # Hole: from its floor (or the far face) out to the face it was drilled from.
            t = (pts - cyl.point) @ cyl.axis
            lo, hi = cyl.point + t.min() * cyl.axis, cyl.point + t.max() * cyl.axis
            hits = [(_axis_hit(cyl, r.plane), r) for r in ends]
            through = not cap_ends
            if cap_ends:
                caps.add(cap_ends[0].id)
                start = _axis_hit(cyl, cap_ends[0].plane)
                others = [p for p, r in hits if r is not cap_ends[0]]
                far = lo if np.linalg.norm(lo - start) > np.linalg.norm(hi - start) else hi
                end = others[0] if others else far
            else:
                start = hits[0][0] if hits else lo
                end = hits[1][0] if len(hits) > 1 else (hi if hits else hi)
                if len(hits) == 1:
                    far = lo if np.linalg.norm(lo - start) > np.linalg.norm(hi - start) else hi
                    end, start = start, far
            features.append(
                Feature(
                    "hole", patch.id, radius, cyl.axis, start, end, through=through, label=label
                )
            )
            continue

        # Part of the way round: a fillet between the two planes it is tangent to.
        if len(sides) < 2:
            skipped.append(f"a curved surface (radius about {cyl.radius:.1f} mm)")
            continue
        sides.sort(key=lambda r: -shared.get((min(r.id, patch.id), max(r.id, patch.id)), 0.0))
        a = sides[0]
        crossing = [r for r in sides[1:] if abs(r.plane.normal @ a.plane.normal) < MAX_SIDE_DOT]
        if not crossing:
            skipped.append(f"a curved surface (radius about {cyl.radius:.1f} mm)")
            continue
        b = crossing[0]
        radius = _fillet_radius(a.plane, b.plane, patch.convex, pts, cyl.radius)
        radius, label = _snap_radius(radius, tol, "fillet")
        # Below the level's smallest kept fillet, the edge is simply made sharp.
        kind = "fillet"
        if not level.keep_fillets or radius < level.min_fillet_radius_mm:
            kind = "sharp"
        axis = np.cross(a.plane.normal, b.plane.normal)
        axis /= np.linalg.norm(axis)
        features.append(
            Feature(kind, patch.id, radius, axis, pts.mean(0), pts.mean(0), (a.id, b.id))
        )
    caps |= _blend_facets(planes, features, level)
    return features, caps, _skipped_note(skipped)


def _blend_facets(planes: list[Region], features: list[Feature], level: ToleranceLevel) -> set[int]:
    """Small planes beside a fillet they are not part of: facets of a corner blend."""
    rounded = {f.patch: set(f.planes) for f in features if f.kind in EDGE_KINDS}
    small = BLEND_FACET_AREA * level.min_feature_mm**2
    return {
        r.id
        for r in planes
        if r.area < small and any(r.id not in rounded[n] for n in r.neighbours if n in rounded)
    }


def apply_features(
    base: cq.Shape, features: list[Feature], planes: list[Region]
) -> tuple[cq.Shape, str | None]:
    """Base solid, then bosses, then fillets, then holes: the order a designer would use.

    A feature that OpenCASCADE cannot apply (a fillet too big for its edge, say) is left out
    and noted, rather than failing the whole part."""
    by_id = {r.id: r for r in planes}
    solid: cq.Shape = base
    bodies = len(base.Solids())
    failed: list[str] = []

    def attempt(f: Feature, op) -> None:
        nonlocal solid
        try:
            result = op(solid)
            parts = result.Solids()
            # A feature must not split a body, merge bodies or break one.
            if len(parts) == bodies and all(p.isValid() for p in parts):
                solid = parts[0] if bodies == 1 else cq.Compound.makeCompound(parts)
                return
        except Exception:  # OpenCASCADE raises a variety of errors here
            pass
        failed.append(f.kind)

    for f in features:
        if f.kind == "boss":
            rod = cylinder_between(f.start - OVERSHOOT_MM * _unit(f.end - f.start), f.end, f.radius)
            attempt(f, lambda s, rod=rod: s.fuse(rod).clean())
    fillets = [f for f in features if f.kind == "fillet"]
    # Biggest first, keeping each fillet that still works together with those kept so far.
    kept: list[Feature] = []
    for f in sorted(fillets, key=lambda f: -f.radius):
        if _fillet_all(solid, [*kept, f], by_id) is not None:
            kept.append(f)
        else:
            failed.append(f.kind)
    together = _fillet_all(solid, kept, by_id) if kept else None
    if together is not None and len(together.Solids()) == bodies:
        solid = together
    else:
        failed += [f.kind for f in kept]
    # Chamfers after fillets, as a designer would: a chamfer then runs on round the corners.
    chamfers = [f for f in features if f.kind == "chamfer"]
    if chamfers:
        cut = _chamfer_all(solid, chamfers, by_id)
        if cut is not None and len(cut.Solids()) == bodies:
            solid = cut
        else:
            failed += [f.kind for f in chamfers]
    for f in features:
        if f.kind in ("hole", "slot"):
            d = _unit(f.end - f.start)
            start = f.start - OVERSHOOT_MM * d if f.through else f.start
            end = f.end + OVERSHOOT_MM * d
            tool = cylinder_between(start, end, f.radius) if f.kind == "hole" else None
            if f.kind == "slot":
                tool = slot_between(start, end, f.span, f.radius)
            attempt(f, lambda s, tool=tool: s.cut(tool).clean())
    note = None
    if failed:
        note = f"Could not apply {len(failed)} feature(s) ({', '.join(failed)}); left out."
    return solid, note


# ---------------------------------------------------------------------------------------------


def _skipped_note(skipped: list[str]) -> str | None:
    if not skipped:
        return None
    listed = "; ".join(sorted(set(skipped)))
    return (
        f"{len(skipped)} curved area{'s' if len(skipped) > 1 else ''} could not be made into a "
        f"hole, boss or fillet yet and {'were' if len(skipped) > 1 else 'was'} left to the flat "
        f"faces around {'them' if len(skipped) > 1 else 'it'} ({listed})."
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> cq.Solid:
    length = float(np.linalg.norm(end - start))
    return cq.Solid.makeCylinder(
        radius, length, cq.Vector(*start), cq.Vector(*((end - start) / length))
    )


def slot_between(start: np.ndarray, end: np.ndarray, span: np.ndarray, radius: float) -> cq.Solid:
    """Slot tool: a stadium of the given radius whose ends are start and start + span, swept
    from start to end along the axis."""
    depth = end - start
    length = float(np.linalg.norm(depth))
    plane = cq.Plane(
        origin=cq.Vector(*(start + span / 2)),
        xDir=cq.Vector(*_unit(span)),
        normal=cq.Vector(*(depth / length)),
    )
    wp = cq.Workplane(plane).slot2D(float(np.linalg.norm(span)) + 2 * radius, 2 * radius)
    return wp.extrude(length).val()


def _find_slots(
    mesh: Geo,
    by_id: dict[int, Region],
    patches: list[CurvedPatch],
    directions: np.ndarray,
    level: ToleranceLevel,
) -> list[Feature]:
    """Closed slots: two half-round ends joined by a pair of parallel walls facing each other.

    The base solid is rebuilt without the walls, as if the slot were filled, and the slot is
    then cut like a hole."""
    tol = level.fit_tolerance_mm
    sin_side = np.sin(np.radians(SIDE_ANGLE_DEG))
    cos_end = np.cos(np.radians(END_ANGLE_DEG))
    ends: dict[tuple[int, int], list[tuple[CurvedPatch, Cylinder]]] = {}
    for patch in patches:
        if patch.convex or not SLOT_COVERAGE[0] <= patch.coverage <= SLOT_COVERAGE[1]:
            continue
        pts = mesh.vertices[np.unique(mesh.faces[patch.faces])]
        cyl = _snap_axis(patch.cylinder, directions, level, pts)
        walls = [
            by_id[n]
            for n in patch.neighbours
            if n in by_id and abs(by_id[n].plane.normal @ cyl.axis) <= sin_side
        ]
        for i, a in enumerate(walls):
            for b in walls[i + 1 :]:
                if a.plane.normal @ b.plane.normal > -MAX_SIDE_DOT:
                    continue
                # Walls facing each other, one diameter apart, with the axis halfway.
                gap = _gap(a.plane, b.plane)
                da, db = a.plane.distance(cyl.point), b.plane.distance(cyl.point)
                if abs(gap - 2 * cyl.radius) > 2 * tol or abs(da - db) > 2 * tol:
                    continue
                ends.setdefault((min(a.id, b.id), max(a.id, b.id)), []).append((patch, cyl))
    slots = []
    for (wa, wb), pair in ends.items():
        if len(pair) != 2:
            continue
        (pa, ca), (pb, cb) = pair
        a = by_id[wa].plane
        axis = ca.axis
        along = _unit(np.cross(axis, a.normal))
        # Width from the walls: they are the biggest, flattest part of the slot.
        width = _gap(a, by_id[wb].plane)
        radius, _ = _snap_radius(width / 2, tol, "boss")
        centre_a = ca.point - (a.distance(ca.point) - width / 2) * a.normal
        gap = float((cb.point - centre_a) @ along)
        span = _round_size(abs(gap), tol) * np.sign(gap) * along
        # Depth from the faces the ends run into: a floor makes it blind.
        nbrs = {n for p in (pa, pb) for n in p.neighbours if n in by_id and n not in (wa, wb)}
        tops = [by_id[n] for n in nbrs if abs(by_id[n].plane.normal @ axis) >= cos_end]
        floors = [r for r in tops if r.neighbours <= {pa.id, pb.id, wa, wb}]
        hits = sorted(
            ((_axis_hit(ca, r.plane) - centre_a) @ axis, r.id) for r in tops if r not in floors
        )
        if len(floors) > 1 or len(hits) + len(floors) < 2 or not hits:
            continue
        if floors:
            start = _axis_hit(ca, by_id[floors[0].id].plane)
            far = max(hits, key=lambda h: abs((centre_a + h[0] * axis - start) @ axis))
            end = centre_a + far[0] * axis
        else:
            start, end = centre_a + hits[0][0] * axis, centre_a + hits[-1][0] * axis
        start = start - ((start - centre_a) @ along) * along
        end = end - ((end - centre_a) @ along) * along
        planes = (wa, wb, *(r.id for r in floors))
        slots.append(
            Feature(
                "slot",
                pa.id,
                radius,
                axis,
                start,
                end,
                planes,
                through=not floors,
                span=span,
                also=(pb.id,),
            )  # fmt: skip
        )
    return slots


def _find_chamfers(
    by_id: dict[int, Region],
    shared: dict[tuple[int, int], float],
    taken: set[int],
    level: ToleranceLevel,
) -> list[Feature]:
    """Narrow flat strips that bevel the edge between two bigger faces."""
    tol = level.fit_tolerance_mm
    sin_side = np.sin(np.radians(SIDE_ANGLE_DEG))
    found: list[Feature] = []
    used: set[int] = set()
    for c in sorted(by_id.values(), key=lambda r: r.area):
        if c.id in taken or c.id in used:
            continue

        def border(r: Region, c: Region = c) -> float:
            return shared.get((min(r.id, c.id), max(r.id, c.id)), 0.0)

        sides = sorted(
            (by_id[n] for n in c.neighbours if n in by_id and n not in taken | used),
            key=lambda r: -border(r),
        )
        if len(sides) < 2:
            continue
        a, b = sides[0], sides[1]
        if a.area <= c.area or b.area <= c.area:
            continue
        na, nb, nc = a.plane.normal, b.plane.normal, c.plane.normal
        if abs(na @ nb) >= MAX_SIDE_DOT:
            continue
        edge = _unit(np.cross(na, nb))
        # The strip leans between the two faces and runs along their edge.
        if not 0.2 < nc @ na < MAX_SIDE_DOT or not 0.2 < nc @ nb < MAX_SIDE_DOT:
            continue
        if abs(nc @ edge) > sin_side:
            continue
        length = min(border(a), border(b))
        width = c.area / max(length, 1e-9)
        if not CHAMFER_MIN_WIDTH <= width / level.min_feature_mm <= CHAMFER_MAX_WIDTH:
            continue
        if length < CHAMFER_MIN_LENGTH * width:
            continue
        # Set-backs along each face, from where the faces would meet to where the strip starts.
        centre = np.linalg.solve(np.vstack([na, nb, edge]), [a.plane.offset, b.plane.offset, 0.0])
        on_a = np.linalg.solve(np.vstack([na, nc, edge]), [a.plane.offset, c.plane.offset, 0.0])
        on_b = np.linalg.solve(np.vstack([nb, nc, edge]), [b.plane.offset, c.plane.offset, 0.0])
        # Only an outside edge: the faces would meet in front of the strip, outside the part.
        if c.plane.distance(centre) <= 0:
            continue
        da, db = float(np.linalg.norm(on_a - centre)), float(np.linalg.norm(on_b - centre))
        size = _round_size((da + db) / 2, SNAP_SHARE * tol)
        if abs(da - db) > tol:
            continue  # an uneven bevel is more likely a sloped face of the part
        found.append(Feature("chamfer", c.id, size, edge, centre, centre, (a.id, b.id)))
        used.add(c.id)
    return found


def _gap(a: Plane, b: Plane) -> float:
    """Distance from plane a to plane b, measured along a's normal."""
    return float(a.distance(b.offset * b.normal))


def _axis_hit(cyl: Cylinder, plane: Plane) -> np.ndarray:
    t = (plane.offset - plane.normal @ cyl.point) / (plane.normal @ cyl.axis)
    return cyl.point + t * cyl.axis


def _directions(planes: list[Region]) -> np.ndarray:
    normals = np.array([r.plane.normal for r in planes])
    return normals if len(normals) else np.zeros((0, 3))


def _snap_axis(
    cyl: Cylinder, directions: np.ndarray, level: ToleranceLevel, pts: np.ndarray
) -> Cylinder:
    """Square a hole or boss up with the part when it is nearly square already."""
    if not len(directions):
        return cyl
    dots = np.abs(directions @ cyl.axis)
    best = int(np.argmax(dots))
    if dots[best] < np.cos(np.radians(level.angle_snap_deg)):
        return cyl
    axis = directions[best] * np.sign(directions[best] @ cyl.axis)
    return fit_circle_on_axis(pts, axis, cyl)


def _snap_radius(radius: float, tol: float, kind: str) -> tuple[float, str | None]:
    budget = SNAP_SHARE * tol
    if kind == "hole":
        d = 2 * radius
        budget = HOLE_SNAP_SHARE * tol
        named = min(STANDARD_HOLES, key=lambda s: abs(s[0] - d))
        plain = _round_size(d, budget)
        if abs(named[0] - d) <= budget and abs(named[0] - d) <= abs(plain - d) + 1e-9:
            return named[0] / 2, named[1]
        return plain / 2, None
    if kind == "boss":
        return _round_size(2 * radius, budget * 2) / 2, None
    return _round_size(radius, budget), None


def _round_size(value: float, budget: float) -> float:
    for step in (1.0, 0.5, 0.1):
        snapped = round(value / step) * step
        if abs(snapped - value) <= budget:
            return snapped
    return round(value, 2)


def _fillet_radius(a: Plane, b: Plane, convex: bool, pts: np.ndarray, guess: float) -> float:
    """Radius of the fillet that rounds the edge between planes a and b, fitted to the scan.

    The model is the real cross-section: plane a up to the tangent line, the arc, then plane b.
    Points the cylinder took from either plane near the tangent lines then fit the model as
    well, instead of flattening the arc and inflating the radius.
    """
    axis = _unit(np.cross(a.normal, b.normal))
    sign = -1.0 if convex else 1.0
    # A point on the edge line: on both planes, closest to the patch centroid.
    m = np.vstack([a.normal, b.normal, axis])
    edge = np.linalg.solve(m, [a.offset, b.offset, axis @ pts.mean(0)])

    def centre(r: float) -> np.ndarray:
        return np.linalg.solve(m, [a.offset + sign * r, b.offset + sign * r, axis @ edge])

    def cost(r: float) -> float:
        c = centre(r)
        rel = pts - c
        rel -= np.outer(rel @ axis, axis)
        ta, tb = -sign * a.normal, -sign * b.normal  # centre to the tangent points
        turn = np.cross(ta, tb) @ axis
        inside = (np.cross(ta, rel) @ axis * turn >= 0) & (np.cross(rel, tb) @ axis * turn >= 0)
        d_arc = np.abs(np.linalg.norm(rel, axis=1) - r)
        d_planes = np.minimum(np.abs(a.distance(pts)), np.abs(b.distance(pts)))
        d = np.where(inside, d_arc, d_planes)
        return float(np.mean(d**2))

    res = minimize_scalar(cost, bounds=(0.05 * guess, 3.0 * guess), method="bounded")
    return float(res.x)


def _fillet_all(
    solid: cq.Shape, fillets: list[Feature], by_id: dict[int, Region]
) -> cq.Shape | None:
    """Every fillet in one operation, each edge with its own radius; None if it fails."""
    maker = BRepFilletAPI_MakeFillet(solid.wrapped)
    for f in fillets:
        edges = _edges_on(solid, by_id[f.planes[0]].plane, by_id[f.planes[1]].plane)
        if not edges:
            return None
        for e in edges:
            maker.Add(f.radius, e.wrapped)
    try:
        maker.Build()
        if not maker.IsDone():
            return None
        result = cq.Shape.cast(maker.Shape())
    except Exception:  # OpenCASCADE raises a variety of errors here
        return None
    if not all(p.isValid() for p in result.Solids()):
        return None
    return result


def _chamfer_all(
    solid: cq.Shape, chamfers: list[Feature], by_id: dict[int, Region]
) -> cq.Shape | None:
    """Every chamfer in one operation; None if it fails."""
    maker = BRepFilletAPI_MakeChamfer(solid.wrapped)
    for f in chamfers:
        if not _edges_on(solid, by_id[f.planes[0]].plane, by_id[f.planes[1]].plane):
            return None
    try:
        for f in chamfers:
            for e in _edges_on(solid, by_id[f.planes[0]].plane, by_id[f.planes[1]].plane):
                # An edge tangent to one already added is in that chamfer's chain.
                if not maker.Contour(e.wrapped):
                    maker.Add(f.radius, e.wrapped)
        maker.Build()
        if not maker.IsDone():
            return None
        result = cq.Shape.cast(maker.Shape())
    except Exception:  # OpenCASCADE raises a variety of errors here
        return None
    if not all(p.isValid() for p in result.Solids()):
        return None
    return result


def _edges_on(solid: cq.Shape, a: Plane, b: Plane) -> list[cq.Edge]:
    size = solid.BoundingBox().DiagonalLength
    eps = 1e-6 * size + 1e-7
    line = np.cross(a.normal, b.normal)
    line /= np.linalg.norm(line)
    out = []
    for e in solid.Edges():
        pts = np.array([e.startPoint().toTuple(), e.endPoint().toTuple()])
        if np.abs(a.distance(pts)).max() > eps or np.abs(b.distance(pts)).max() > eps:
            continue
        # Only edges running along the line where the planes meet: a sliver or a point where
        # they happen to touch makes the fillet fail.
        run = pts[1] - pts[0]
        if np.linalg.norm(run) > 1e-3 * size and abs(run @ line) > 0.99 * np.linalg.norm(run):
            out.append(e)
    return out
