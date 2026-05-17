from gelu_1l.artifacts import load_json, save_json


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    save_json(path, {"feature_id": 8, "status": "ok"})
    assert load_json(path) == {"feature_id": 8, "status": "ok"}
