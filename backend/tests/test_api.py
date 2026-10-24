from fastapi.testclient import TestClient

from nurbsmith import fixtures
from nurbsmith.api import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_tolerances():
    body = client.get("/api/tolerances").json()
    assert body["default"] == "functional"
    assert [lvl["key"] for lvl in body["levels"]] == ["rough", "functional", "precise"]


def test_upload_stl():
    data = fixtures.box().export(file_type="stl")
    res = client.post("/api/meshes", files={"file": ("box.stl", data)})
    assert res.status_code == 200
    assert res.json()["summary"]["watertight"] is True


def test_upload_bad_format():
    res = client.post("/api/meshes", files={"file": ("box.dwg", b"xx")})
    assert res.status_code == 422


def test_convert_and_download():
    data = fixtures.l_bracket().export(file_type="stl")
    res = client.post(
        "/api/convert", files={"file": ("bracket.stl", data)}, data={"level": "functional"}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["face_types"] == {"PLANE": 8}
    assert body["report"]["fit"]["within_tolerance_pct"] == 100.0

    step = client.get(body["downloads"]["step"])
    assert step.status_code == 200
    assert step.content.startswith(b"ISO-10303-21")
    assert "bracket.step" in step.headers["content-disposition"]
    for key in ("model_stl", "deviation_ply"):
        assert client.get(body["downloads"][key]).status_code == 200


def test_convert_scales_inches():
    data = fixtures.box(size=(1.0, 2.0, 3.0), subdivisions=0).export(file_type="stl")
    res = client.post("/api/convert", files={"file": ("b.stl", data)}, data={"units": "in"})
    assert res.json()["report"]["size_mm"] == [25.4, 50.8, 76.2]


def test_convert_curved_part_is_a_clear_422():
    data = fixtures.ball().export(file_type="stl")
    res = client.post("/api/convert", files={"file": ("c.stl", data)})
    assert res.status_code == 422
    assert "curved" in res.json()["detail"]


def test_convert_merges_a_second_scan():
    top, bottom = (fixtures.subdivided(s, 1.0) for s in fixtures.sides(fixtures.l_bracket()))
    bottom = fixtures.rotated(bottom, seed=3)
    bottom.apply_translation([30.0, -10.0, 5.0])
    res = client.post(
        "/api/convert",
        files={
            "file": ("top.stl", top.export(file_type="stl")),
            "second_file": ("bottom.stl", bottom.export(file_type="stl")),
        },
    )
    assert res.status_code == 200, res.text
    report = res.json()["report"]
    assert report["face_types"] == {"PLANE": 8}
    assert report["fit"]["within_tolerance_pct"] > 99
    assert report["merge"]["overlap_pct"] > 5


def test_scans_that_cannot_be_merged_are_a_clear_422():
    small = fixtures.ball(radius=10.0, squash=1.0).export(file_type="stl")
    big = fixtures.ball(radius=40.0, squash=1.0).export(file_type="stl")
    res = client.post(
        "/api/convert", files={"file": ("a.stl", small), "second_file": ("b.stl", big)}
    )
    assert res.status_code == 422
    assert "overlap" in res.json()["detail"]


def test_unknown_result_404():
    assert client.get("/api/results/nope/model.step").status_code == 404
