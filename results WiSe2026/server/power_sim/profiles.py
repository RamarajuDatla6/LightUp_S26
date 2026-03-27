import json
import os
from typing import Any, Dict, List


_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "profiles.json")


def _load_all() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(_PROFILES_PATH):
        raise FileNotFoundError(f"profiles.json not found at: {_PROFILES_PATH}")

    # Detect empty file early
    if os.path.getsize(_PROFILES_PATH) == 0:
        raise ValueError(f"profiles.json is empty at: {_PROFILES_PATH}")

    with open(_PROFILES_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"profiles.json contains only whitespace at: {_PROFILES_PATH}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"profiles.json is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("profiles.json must contain a JSON object at the top-level")

    return data


def list_models() -> List[Dict[str, Any]]:
    data = _load_all()
    out: List[Dict[str, Any]] = []
    for mid, profile in data.items():
        out.append(
            {
                "id": mid,
                "name": profile.get("name", mid),
                "notes": profile.get("notes", ""),
                "battery_wh_default": profile.get("battery_wh_default", 19.0),
            }
        )
    return out


def load_profile(model_id: str) -> Dict[str, Any]:
    data = _load_all()
    if model_id not in data:
        raise KeyError(f"Unknown model_id '{model_id}'. Available: {list(data.keys())}")
    return data[model_id]
