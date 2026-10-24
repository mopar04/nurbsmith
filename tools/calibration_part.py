"""NURBSmith calibration part: every feature the converter handles (or should), on one print.

Prints flat with no supports, about 90 x 60 x 28 mm. Scan both sides, then convert and
compare against the STEP this writes, feature by feature (see FEATURES).

    backend/.venv/bin/python tools/calibration_part.py out_dir
"""

import json
import sys
from pathlib import Path

import cadquery as cq

L, W, T = 90.0, 60.0, 8.0  # base plate

# name, kind, nominal values (mm), where (x, y from plate centre)
FEATURES = []


def note(name, kind, where, **dims):
    FEATURES.append({"name": name, "kind": kind, "at_mm": where, **dims})


def build() -> cq.Workplane:
    part = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    note("base plate", "planes", (0, 0), size=[L, W, T])

    # Outer vertical corners: fillets of two sizes.
    part = part.edges("|Z and >X").fillet(8.0)
    note("outer corner fillets, +X side", "fillet", (L / 2, 0), radius=8.0)
    part = part.edges("|Z and <X").fillet(3.0)
    note("outer corner fillets, -X side", "fillet", (-L / 2, 0), radius=3.0)

    # Top edges. The edge chain is tangent all the way round (the corner fillets join the
    # sides), so the chamfer picked on -Y runs round the whole top, and the small fillet picked
    # on +Y then rounds the chamfer's upper edge all round as well. Kept as printed.
    part = part.faces(">Z").edges("<Y").chamfer(1.5)
    note("top edge chamfer, all round", "chamfer", (0, -W / 2), size=1.5)
    part = part.faces(">Z").edges(">Y").fillet(1.5)
    note("fillet on the chamfer's upper edge, all round", "fillet", (0, W / 2), radius=1.5)

    top = lambda: part.faces(">Z").workplane(centerOption="CenterOfBoundBox")  # noqa: E731

    # Through holes: three standard clearances and one deliberately non-standard size.
    for (x, y), d, label in (
        ((-36, -20), 3.4, "M3 clearance"),
        ((-36, -8), 5.5, "M5 clearance"),
        ((-36, 4), 6.6, "M6 clearance"),
        ((-36, 16), 7.3, "non-standard"),
    ):
        part = top().center(x, y).hole(d)
        note(f"through hole {label}", "hole", (x, y), diameter=d, through=True)

    # Countersunk M4 and counterbored M5.
    part = top().center(-22, -18).cskHole(4.5, 9.0, 90)
    note("countersunk hole M4", "countersink", (-22, -18), diameter=4.5, csk_diameter=9.0,
         csk_angle_deg=90, through=True)  # fmt: skip
    part = top().center(-22, 0).cboreHole(5.5, 10.0, 4.0)
    note("counterbored hole M5", "counterbore", (-22, 0), diameter=5.5, cbore_diameter=10.0,
         cbore_depth=4.0, through=True)  # fmt: skip

    # Blind hole.
    part = top().center(-22, 18).hole(6.0, depth=5.0)
    note("blind hole", "hole", (-22, 18), diameter=6.0, depth=5.0, through=False)

    # Slot through the plate (two half-cylinders and two flats).
    part = top().center(-4, -18).slot2D(20.0, 6.0).cutThruAll()
    note("through slot", "slot", (-4, -18), length=20.0, width=6.0)

    # Rectangular pocket with rounded inside corners.
    part = (
        top().center(-4, 4).sketch().rect(16.0, 12.0).vertices().fillet(3.0).finalize()
        .cutBlind(-4.0)
    )  # fmt: skip
    note("rectangular pocket", "pocket", (-4, 4), size=[16.0, 12.0], depth=4.0,
         corner_radius=3.0)  # fmt: skip

    # Boss with a cross hole: a round body standing on the plate.
    boss = cq.Workplane("XY").workplane(offset=T).center(18, 14).circle(8.0).extrude(20.0)
    part = part.union(boss)
    note("boss", "boss", (18, 14), diameter=16.0, height=20.0)
    cross = (
        cq.Workplane("YZ").workplane(offset=0).center(14, T + 12.0).circle(2.5)
        .extrude(40.0, both=True)
    )  # fmt: skip
    part = part.cut(cross.translate((18, 0, 0)))
    note("cross hole through the boss (along X)", "hole", (18, 14), diameter=5.0,
         height_above_plate=12.0, through=True)  # fmt: skip

    # Thin upright wall (2 mm wall, 15 mm tall).
    wall = cq.Workplane("XY").workplane(offset=T).center(18, -14).rect(30.0, 2.0).extrude(15.0)
    part = part.union(wall)
    note("thin wall", "thin wall", (18, -14), thickness=2.0, length=30.0, height=15.0)

    # Sloped face: a 30 degree ramp on the +X end of the plate top.
    ramp = (
        cq.Workplane("XZ")
        .polyline([(36, T), (44, T), (44, T + 4.6188)]).close()
        .extrude(20.0, both=True)
    )  # fmt: skip
    # Its high end (x = 44) stands just past where the top's edge chamfer starts, so it
    # overhangs the chamfer by about a millimetre. Kept as printed.
    part = part.union(ramp)
    note("30 degree ramp", "plane (angled)", (40, 0), angle_deg=30.0, run=8.0, rise=4.62)

    # Shallow embossed lettering: noise, which the converter should NOT reproduce.
    try:
        text = (
            cq.Workplane("XY").workplane(offset=T).center(-4, 22)
            .text("NURBSmith", 5.0, 0.4, halign="center", valign="center")
        )  # fmt: skip
        part = part.union(text)
        note("embossed text (should be ignored)", "noise", (-4, 22), height=0.4)
    except Exception as exc:  # no font available
        print("skipped text:", exc)

    return part


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    part = build()
    solid = part.val()
    assert solid.isValid()
    cq.exporters.export(part, str(out / "nurbsmith_calibration.step"))
    cq.exporters.export(part, str(out / "nurbsmith_calibration.stl"), tolerance=0.01,
                        angularTolerance=0.05)  # fmt: skip
    (out / "nurbsmith_calibration.json").write_text(json.dumps(FEATURES, indent=2))
    bb = solid.BoundingBox()
    print(f"size {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, volume {solid.Volume():.0f}")
    print(len(FEATURES), "features")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
