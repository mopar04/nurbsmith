"""HTTP API: mesh upload/summary, conversion to STEP, and result downloads."""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import __version__
from .brep import UnsupportedGeometry
from .convert import convert
from .merge import MergeError, merge_scans
from .mesh_io import SUPPORTED_FORMATS, MeshLoadError, load_mesh, summarize
from .quality import deviation_colors
from .step_io import write_step
from .tolerance import DEFAULT_LEVEL, LEVELS, get_level

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_STORED_RESULTS = 50
UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

app = FastAPI(title="NURBSmith", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_results_dir = Path(tempfile.mkdtemp(prefix="nurbsmith-"))
_results: OrderedDict[str, Path] = OrderedDict()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/tolerances")
def tolerances() -> dict:
    return {
        "default": DEFAULT_LEVEL.key,
        "levels": [lvl.as_dict() for lvl in LEVELS.values()],
    }


async def _read_mesh(file: UploadFile, units: str = "mm"):
    if units not in UNIT_SCALE:
        raise HTTPException(422, f"Unknown units {units!r}; expected one of {list(UNIT_SCALE)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Mesh is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        mesh = load_mesh(data, file.filename or "")
    except MeshLoadError as exc:
        raise HTTPException(422, str(exc)) from exc
    if UNIT_SCALE[units] != 1.0:
        mesh.apply_scale(UNIT_SCALE[units])
    return mesh


@app.post("/api/meshes")
async def upload_mesh(file: UploadFile = File(...), units: str = Form("mm")) -> dict:
    mesh = await _read_mesh(file, units)
    return {
        "filename": file.filename,
        "summary": summarize(mesh).as_dict(),
        "supported_formats": list(SUPPORTED_FORMATS),
    }


@app.post("/api/convert")
async def convert_mesh(
    file: UploadFile = File(...),
    level: str = Form(DEFAULT_LEVEL.key),
    units: str = Form("mm"),
    second_file: UploadFile | None = File(None),
) -> dict:
    """Convert a scan. second_file is an optional scan of the other side of the same part
    (the bottom, or the back); it is lined up with the first and merged before converting."""
    try:
        tol = get_level(level)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    mesh = await _read_mesh(file, units)
    merged = None
    if second_file is not None:
        other = await _read_mesh(second_file, units)
        try:
            merged = await run_in_threadpool(merge_scans, mesh, other)
        except MergeError as exc:
            raise HTTPException(422, str(exc)) from exc
        mesh = merged.mesh
    try:
        result = await run_in_threadpool(convert, mesh, tol)
    except UnsupportedGeometry as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # OpenCASCADE failures surface as assorted exception types
        log.exception("Conversion failed for %s at %s", file.filename, level)
        raise HTTPException(
            422, "Could not rebuild a clean solid from this scan. Try a looser accuracy level."
        ) from exc

    result_id = uuid.uuid4().hex
    folder = _results_dir / result_id
    folder.mkdir()
    write_step(result.solid, folder / "model.step")
    result.solid.exportStl(str(folder / "model.stl"), tolerance=tol.fit_tolerance_mm / 4)
    scan = result.scan_mesh.copy()
    scan.visual.vertex_colors = deviation_colors(result.deviation_mm, tol.fit_tolerance_mm)
    scan.export(folder / "deviation.ply")
    _remember(result_id, folder)

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename or "model").stem) or "model"
    report = result.report()
    if merged is not None:
        report["merge"] = merged.report()
    return {
        "id": result_id,
        "level": tol.as_dict(),
        "report": report,
        "downloads": {
            "step": f"/api/results/{result_id}/{stem}.step",
            "model_stl": f"/api/results/{result_id}/model.stl",
            "deviation_ply": f"/api/results/{result_id}/deviation.ply",
        },
    }


@app.get("/api/results/{result_id}/{name}")
def get_result(result_id: str, name: str) -> FileResponse:
    folder = _results.get(result_id)
    if folder is None:
        raise HTTPException(404, "Result expired or not found; convert again")
    if name.endswith(".step"):
        return FileResponse(folder / "model.step", media_type="application/step", filename=name)
    if name in ("model.stl", "deviation.ply"):
        return FileResponse(folder / name, media_type="application/octet-stream")
    raise HTTPException(404, "Unknown file")


def _remember(result_id: str, folder: Path) -> None:
    _results[result_id] = folder
    while len(_results) > MAX_STORED_RESULTS:
        _, old = _results.popitem(last=False)
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
