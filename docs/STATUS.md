# Status

_Last updated: 2026-10-24 (calibration part)_

## Done
- **M0 scaffold:** API, mesh loading, viewer, Docker, CI.
- **M1 flat-faced parts:** segmentation, plane snapping, exact solid rebuild, confidence
  metrics, deviation view, STEP download. AGPL-3.0, NURBSmith naming, UI reference notes.
- **M2 mechanical features:**
  - Cylinder finding (`cylinders.py`): pairs of planar strips seed a cylinder fit, which is
    grown over triangles and refitted. Checks reject gentle curves that fit a plane, wide
    "cylinders" fitted to a few big flat triangles, and corners passing as one gentle cylinder.
  - Features (`features.py`): through and blind holes, counterbores, slots, bosses, fillets,
    chamfers and round bodies. Hole sizes snap to standard metric clearance / close fit / tap
    drill and inch sizes only within 0.35x the tolerance, otherwise they are kept as measured.
    Fillets below the level's smallest radius are made sharp.
  - Base solid: the exact plane-intersection rebuild, with a fallback (`cells.py`) that cuts a
    block with the fitted planes near their regions and keeps the inside blocks. The fallback
    survives gaps, warps and ragged rims, and keeps each separate body of an assembly scan.
  - Real scans: loose floating pieces are dropped and counted. Dense scans are decimated
    gently (several low-aggression passes; the default one cut corners). Noise is estimated
    on a densified copy, so clean coarse meshes don't read as noisy. Faces warped a little for
    their size (bent sheet metal) stay flat. Deviation scoring is chunked and approximates
    far-off points, and the block fuse runs in rounds (10 GB down to 3 GB on a 2.2M-face scan).
  - Guards: refuse a result that matches under 90% of the scan at the working tolerance, and
    refuse parts where over 15% of the surface is freeform (curved but not a feature).
  - API returns `features` and `notes`; the viewer lists features ("4× Hole Ø6.6 mm (M6
    clearance), through") and notes.
  - Tests: five CAD-built parts (holed plate, filleted block, boss plate, blind hole block,
    bracket) at 0.05 mm noise recover exact face types, feature sizes and volume.

## Real scans tried (Zac's, in `Developer/Nurbsmith/Test Scans`)
- Scale checks out: tape roll 92.5/76.8/48.5 vs 93/76/48.5; cap 102.2–102.8 vs 102.
- Box with floating bits: bits dropped; Rough converts (98% within 1 mm); Functional is held
  back by faces warped up to 2 mm.
- RC stand (two bent plates on a threaded post), full scan: both plates with holes and
  fillets, 82% within 0.3 mm, so still refused. The post and the round feet are not modelled.
- Bin (thin walls), tape roll (open end, rough edges), cap (front and back as separate scans),
  smoke detector and bottle (freeform): not converting yet.

## In progress
- **Real-scan readiness, part 1 (scanner holder):** Zac's own design (`tests/fixtures/
  scanner_holder.stl`) is now a ground-truth test. It went from 77% to 95% within 0.3 mm,
  with volume within 0.5%. Changes:
  - Bowed walls (a huge-radius sliver of a cylinder) stay planes.
  - The block solid starts from the oriented bounding box.
  - Planes cut through to the sharp corner under their fillets, and blocks there vote by
    that sharp corner.
  - Hole walls count as solid until drilled, and boss walls as empty until added.
  - Fillets are applied together, biggest first, keeping those that work together.
  - Stray zero-length edges no longer break fillets.
  - The scan of the holder converts at 87%. Its deep pockets were not fully scanned, which
    Zac says is a coverage problem, not ours.

- **Calibration part** (`tools/calibration_part.py`, 19 known features; Zac is printing and
  scanning it). The design mesh went from 90% to 98% within 0.3 mm at Functional. Now right:
  seven of the eight holes including the non-standard 7.3 mm one, the counterbore (10 mm, 4 deep, over
  5.5 mm), the blind hole depth (5.0), the slot (6 × 20), the boss (16 × 20) with its cross
  hole, the thin wall, the r8 and r3 corner fillets and the 1.5 mm chamfers. Still wrong:
  - the countersunk hole: no cone feature yet, and the cone muddles the 4.5 mm hole under it,
    which is left out,
  - the corner blends where the r8 fillets meet the top chamfer (read as r11 fillets), and some
    fillets and chamfers that don't apply together,
  - the rounded pocket corners (rebuilt as small flats),
  - the ramp's overhanging end, which leaves a top face that OpenCASCADE's mesher can't
    tessellate in memory (it does after STEP export).

## Next (M2.5: real-scan readiness, agreed with Zac)
- Thin walls: detect wall pairs, measure thickness, patch missing wall areas from it (bin).
- Open and partial scans: close an open end with a plane; trim ragged rims.
- Merge front and back scans of one part (cap) before converting.
- Fewer spurious planes on warped real faces (RC stand gives about 1,000 faces).
- Background jobs: a 2M-face scan takes 5 to 10 minutes.

## Known limits
- Noise matrix (5 test parts × noise 0 / 0.02 / 0.1 / 0.3 mm × 3 levels): correct up to 0.1 mm
  noise, with a few misses (the bracket at 0.1 mm Functional keeps 1 of 3 cylinders; the
  holed plate at 0.02 mm Rough finds 3 of 4 holes). At 0.3 mm noise results are poor; the
  guard catches some but not all. Zac's laser scans are 0.01 to 0.03 mm.
- No cones, spheres, tori or freeform yet; chamfers come out as plain planes.
- Conversion runs inside the request. Results are kept in a temp folder, last 50 only.

## Open questions for Zac
- Default units: mm or inches.
- Local-only at first, or public hosting from day one.
- Donate link URL.
