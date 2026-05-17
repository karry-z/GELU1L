import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
