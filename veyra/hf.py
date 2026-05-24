from __future__ import annotations

from pathlib import Path
from typing import Any

from .inspect import inspect_model
from .prompts import infer_prompt_mode
from .registry import MODELS_DIR, safe_model_name


HF_ORG = "veyra-ai"
ALLOW_PATTERNS = [
    "*.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
]


def list_veyra_models() -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    api = HfApi()
    repos = list(api.list_models(author=HF_ORG, full=True))
    choices: list[dict[str, Any]] = []
    for repo in repos:
        repo_id = repo.modelId
        try:
            files = list_repo_files(repo_id)
        except Exception:
            continue
        if any(f.endswith(".onnx") for f in files) and "tokenizer.json" in files:
            choices.append({"repo_id": repo_id, "files": files, "downloads": getattr(repo, "downloads", None)})
    choices.sort(key=lambda r: ("onnx" not in r["repo_id"].lower(), r["repo_id"]))
    return choices


def download_model(repo_id: str, revision: str = "main") -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    name = safe_model_name(repo_id)
    target = MODELS_DIR / name
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )
    commit = None
    try:
        info = HfApi().model_info(repo_id, revision=revision)
        commit = getattr(info, "sha", None)
    except Exception:
        pass
    return Path(path), commit


def registry_entry(repo_id: str, path: str | Path, revision: str = "main", commit: str | None = None) -> dict[str, Any]:
    info = inspect_model(path)
    model_type = info.model_type or (info.architecture or "unknown").lower()
    config = _read_json(Path(path) / "config.json")
    tokenizer_config = _read_json(Path(path) / "tokenizer_config.json")
    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_commit": commit,
        "path": str(Path(path).expanduser().resolve()),
        "runtime": "onnx",
        "architecture": model_type,
        "mode": infer_prompt_mode(config, tokenizer_config),
        "quantized": "int8" in repo_id.lower() or "quant" in repo_id.lower(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        import json
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
