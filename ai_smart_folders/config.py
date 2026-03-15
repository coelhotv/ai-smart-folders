from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .models import AppConfig


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_path() -> Path:
    env_path = os.environ.get("AI_SMART_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    data_dir = os.environ.get("AI_SMART_DATA_DIR")
    if data_dir:
        return Path(data_dir).expanduser() / "config.yaml"

    local = Path("config.yaml").resolve()
    if local.exists():
        return local

    return (Path.home() / ".ai-smart-folders-data" / "config.yaml").expanduser()


def load_config(config_path: Path | None = None) -> AppConfig:
    candidate = (config_path or default_config_path()).expanduser()
    data: Dict[str, Any] = {}
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    legacy_model = data.pop("ollama_model", None)
    if legacy_model:
        data.setdefault("models", {})
        data["models"].setdefault("classification_model", legacy_model)
        data["models"].setdefault("understanding_model", legacy_model)

    env_overrides: Dict[str, Any] = {}
    if os.environ.get("AI_SMART_DATA_DIR"):
        env_overrides["data_dir"] = os.environ["AI_SMART_DATA_DIR"]
    elif "data_dir" not in data and candidate.name == "config.yaml":
        env_overrides["data_dir"] = str(candidate.parent / ".ai-smart-folders-data")

    merged = _merge_dict(data, env_overrides)
    config = AppConfig(**merged)
    config.data_dir = config.data_dir.expanduser()
    config.inbox_dir = config.inbox_dir.expanduser()
    config.organized_dir = config.organized_dir.expanduser()
    return config
