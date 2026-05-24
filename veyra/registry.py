from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .prompts import normalize_mode


APP_NAME = "veyra"


def _xdg_path(env_name: str, fallback: str) -> Path:
    root = os.environ.get(env_name)
    if root:
        return Path(root) / APP_NAME
    return Path.home() / fallback / APP_NAME


CONFIG_DIR = _xdg_path("XDG_CONFIG_HOME", ".config")
DATA_DIR = _xdg_path("XDG_DATA_HOME", ".local/share")
STATE_DIR = _xdg_path("XDG_STATE_HOME", ".local/state")

CONFIG_PATH = CONFIG_DIR / "config.json"
MODELS_DIR = DATA_DIR / "models"
CHATS_DIR = DATA_DIR / "chats"
HISTORY_PATH = STATE_DIR / "history.txt"


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "autoload": True,
    "theme": "veyra",
    "device": "cpu",
    "stats": False,
    "current_model": None,
    "current_mode": "chatml",
    "defaults": {
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    "models": {},
}


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def default_config() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_config() -> dict[str, Any]:
    ensure_dirs()
    if not CONFIG_PATH.exists():
        config = default_config()
        save_config(config)
        return config
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Config file is invalid JSON: {CONFIG_PATH}") from exc

    config = default_config()
    deep_update(config, loaded)
    theme = config.get("theme")
    if theme == "neon":
        config["theme"] = "blue"
    elif theme not in {"veyra", "warm", "green", "blue", "mono"}:
        config["theme"] = "veyra"
    if config.get("device") not in {"cpu", "cuda", "directml", "coreml", "openvino", "rocm", "tensorrt"}:
        config["device"] = "cpu"
    config["current_mode"] = normalize_mode(config.get("current_mode"))
    if not isinstance(config.get("stats"), bool):
        config["stats"] = False
    if (
        loaded.get("theme") != config.get("theme")
        or loaded.get("device") != config.get("device")
        or loaded.get("stats") != config.get("stats")
    ):
        save_config(config)
    return config


def deep_update(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def save_config(config: dict[str, Any]) -> None:
    ensure_dirs()
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    tmp.replace(CONFIG_PATH)


def models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return config.setdefault("models", {})


def current_model_entry(config: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    name = config.get("current_model")
    if not name:
        return None, None
    return name, models(config).get(name)


def register_model(config: dict[str, Any], name: str, entry: dict[str, Any], select: bool = True) -> None:
    models(config)[name] = entry
    if select:
        config["current_model"] = name
        if entry.get("mode"):
            config["current_mode"] = normalize_mode(entry["mode"])
    save_config(config)


def remove_model(config: dict[str, Any], name: str, delete_files: bool = False) -> None:
    entry = models(config).pop(name, None)
    if config.get("current_model") == name:
        config["current_model"] = None
    save_config(config)
    if delete_files and entry and entry.get("source") == "huggingface" and entry.get("path"):
        shutil.rmtree(entry["path"], ignore_errors=True)


def safe_model_name(name: str) -> str:
    keep = []
    for ch in name.strip().replace("\\", "/").split("/")[-1]:
        keep.append(ch if ch.isalnum() or ch in "-_." else "-")
    return "".join(keep).strip(".-") or "model"
