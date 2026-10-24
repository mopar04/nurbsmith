# Decisions

Newest first. One entry per choice a later session should not have to relitigate.

## 2026-10-24 (calibration part)
- **Hole sizes snap only within 0.35x the tolerance** on the diameter (0.1 mm at Functional),
  stricter than other sizes. Stamped, printed and cast holes are rarely drill sizes, and a
  size Zac measures with calipers must come out as measured (RC stand 13.15 / 16.2 / 9.15).
- **Plane fits weight vertices by area and drop outliers.** Fine detail (embossed text) and
  strips of a neighbouring fillet were pulling the top face of the calibration part 0.15 mm
  high and tilting it. Depths and heights are now exact on design meshes.
- **Slots and chamfers are features.** A closed slot (two half-round ends and two facing
  walls) is filled in the base and cut afterwards; a narrow even bevel between two faces is
  made sharp in the base and chamfered after the fillets, so it runs round filleted corners
  as it would in CAD.
- **A curved patch must turn like a cylinder.** Normals that lag steadily behind the arc
  (measured in slices so noise averages out) mean flat stretches and tight corners, such as a
  pocket end, a cone or a corner blend, posing as one cylinder.
- **Calibration part** (`tools/calibration_part.py`): one print with every feature. Its top
  edge chamfer and fillet run all the way round (CadQuery follows tangent edges), and the ramp
  overhangs that chamfer by about 1 mm. Kept as printed.

## 2026-10-24 (real scans)
- **Scan coverage is the user's job** (Zac). Fill gaps only from context: extended planes,
  measured wall thickness, closing an open end. When big regions were never scanned, say so
  rather than invent them.
- **Design files are the best tests.** A mesh exported from the user's CAD gives exact
  ground truth; Zac's scanner holder is the first (`tests/fixtures/`).
- **The base solid is the part before features.** Blocks near a fillet vote by the sharp
  corner, blocks in a hole count as solid, blocks in a boss as empty. Then features apply.

## 2026-10-24 (M2)
- **Features are CAD operations on a flat base solid.** The base is built from planes only;
  bosses are fused, fillets applied with OpenCASCADE's fillet, holes cut. A feature that
  fails is left out with a note, never a failed part. This is also the M4 replay order.
- **Standard sizes snap only within half the tolerance.** Otherwise the measured size is kept,
  rounded. A hole is never moved to a standard size the scan doesn't support.
- **Cylinders come from strip pairs, not curvature.** Two neighbouring planar strips give a
  seed cylinder; it is grown over triangles and refitted. A patch must not fit a plane, and
  its normals must turn about as far as its arc (0.5× to 2×).
- **Freeform share is checked after features are planned.** Curved patches that become no
  feature count as freeform with curved flat faces; over 15% of the surface is refused.
- **Warp is relative to face size.** A face counts as curved only above both 0.5× tolerance
  and 0.5% of its size (RMS). Real flat parts are warped; refusing them helped nobody.
- **Block-based fallback solid** (`cells.py`) when the exact rebuild cannot close, and every
  sizeable body is kept (an assembly scan can fall apart where a joining piece isn't
  modelled). A result under 90% within the working tolerance is refused, not returned.
- **Decimate gently.** fast-simplification at aggression 4 in repeated passes; the default
  made triangles that cut across corners by millimetres.
- **Next milestone is real-scan readiness** (thin walls, open ends, front/back merge) before
  M3, because Zac's real scans hit those first.

## 2026-10-23 (M1)
- **Open source under AGPL-3.0, with a donate link, no commercial plans** (Zac). This matches
  CNC Kitchen's tools and lets us reuse ScanRuler code with credit. Set the donate link with
  `VITE_DONATE_URL`. The repo stays private until Zac decides to make it public.
- **Display name is "NURBSmith"** (NURB in capitals). Code identifiers stay `nurbsmith`.
- **UI follows BumpMesh and ScanRuler** (Zac): see `docs/UI_REFERENCES.md`.
- **Never model scan noise.** Scan noise sets a floor on the fit tolerance (3.5 sigma). If the
  chosen level is tighter, we fit looser and warn, rather than chase artifacts. The fit score
  is still measured against the level the user picked.
- **Corners come from plane intersections, not mesh vertices.** Each region's boundary gives
  the order of its neighbours, and each change of neighbour is a corner (3-plane
  intersection). The output is exact even when the scan's edges are ragged.
- **Curved parts are refused in M1, not faceted.** A 12-sided prism can be "within tolerance"
  of a cylinder but is the wrong CAD. Curved surfaces are detected (shallow creases, or a
  region residual well above the noise) and get a clear message until M2.
- **Edge-preserving normal smoothing** (bilateral, 8 rounds, 60 deg cutoff) before region
  growing. Plain averaging blurred edges; raw normals fragmented noisy faces.
- **Scope added:** Thicken skin (M3) and Combine scans (M5), per Zac.

## 2026-10-23
- **Target CAD tools:** Fusion 360 and Onshape first. Shapr3D gets the same STEP (it imports STEP
  fine but has no parametric history), no special handling.
- **Geometry in Python with OpenCASCADE (CadQuery/OCP).** Same kernel as FreeCAD; writes
  AP214/AP242 that all target tools import. Browser does viewing and editing only.
- **Python 3.12.** CadQuery has no 3.13 wheels yet.
- **Editable history via replay scripts, not STEP.** STEP cannot carry a feature tree, so M4
  generates Onshape FeatureScript and Fusion API scripts from a feature recipe.
- **Tolerance levels:** Rough 1.0 mm / Functional 0.3 mm (default) / Precise 0.1 mm, each also
  setting the noise filter size. Starting values, to be tuned on real scans.
- **Threads ("screws")** are modelled as plain cylinders plus a thread annotation; CAD tools add
  cosmetic threads.
- **No job queue in M0.** Uploads are analysed inline; Redis + RQ comes in when conversion runs.
- **Repo is the source of truth** for plans, decisions and status so work continues from Claude
  or Claude Code.
