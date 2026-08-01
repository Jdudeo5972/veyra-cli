from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from .inspect import inspect_model
from .prompts import infer_prompt_mode
from .registry import MODELS_DIR, safe_model_name


HF_ORG = "veyra-ai"
MODEL_METADATA_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
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
        onnx_files = onnx_variants(files)
        if not onnx_files or "tokenizer.json" not in files:
            continue
        choices.append(
            {
                "repo_id": repo_id,
                "files": files,
                "onnx_files": onnx_files,
                "downloads": getattr(repo, "downloads", None),
            }
        )
    choices.sort(key=lambda r: ("onnx" not in r["repo_id"].lower(), r["repo_id"]))
    return choices


def get_hf_model(repo_id: str, revision: str = "main") -> dict[str, Any]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    repo_id = normalize_repo_id(repo_id)
    try:
        files = list_repo_files(repo_id, revision=revision)
    except Exception as exc:
        raise RuntimeError(f"Could not access Hugging Face model {repo_id}: {exc}") from exc
    if "tokenizer.json" not in files:
        raise RuntimeError(
            f"{repo_id} does not contain tokenizer.json. Veyra requires a fast tokenizer.json beside the ONNX export."
        )
    variants = onnx_variants(files)
    if not variants:
        raise RuntimeError(f"No ONNX files found in {repo_id}.")
    return {"repo_id": repo_id, "files": files, "onnx_files": variants, "downloads": None}


def normalize_repo_id(value: str) -> str:
    value = value.strip().rstrip("/")
    prefix = "https://huggingface.co/"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if value.count("/") != 1:
        raise RuntimeError("Use a Hugging Face model ID such as owner/model or its huggingface.co URL.")
    return value


def download_model(
    repo_id: str,
    revision: str = "main",
    onnx_file: str | None = None,
) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, list_repo_files, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    files = list_repo_files(repo_id, revision=revision)
    available_onnx = onnx_variants(files)
    if not available_onnx:
        raise RuntimeError(f"No ONNX files found in {repo_id}.")
    if onnx_file is None:
        onnx_file = recommended_onnx_file(available_onnx)
    if onnx_file not in available_onnx:
        raise RuntimeError(f"ONNX file '{onnx_file}' was not found in {repo_id}.")
    if "tokenizer.json" not in files:
        raise RuntimeError(f"{repo_id} does not contain tokenizer.json.")

    name = fetched_model_name(repo_id, onnx_file, len(available_onnx))
    target = MODELS_DIR / name
    download_files = _onnx_download_files(files, onnx_file, available_onnx)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=[*download_files, *MODEL_METADATA_PATTERNS],
    )
    commit = None
    try:
        info = HfApi().model_info(repo_id, revision=revision)
        commit = getattr(info, "sha", None)
    except Exception:
        pass
    return Path(path), commit


def registry_entry(
    repo_id: str,
    path: str | Path,
    revision: str = "main",
    commit: str | None = None,
    onnx_file: str | None = None,
) -> dict[str, Any]:
    info = inspect_model(path)
    model_type = info.model_type or (info.architecture or "unknown").lower()
    config = _read_json(Path(path) / "config.json")
    tokenizer_config = _read_json(Path(path) / "tokenizer_config.json")
    mode = infer_prompt_mode(config, tokenizer_config)
    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_commit": commit,
        "onnx_file": onnx_file or info.onnx_path.relative_to(Path(path)).as_posix(),
        "path": str(Path(path).expanduser().resolve()),
        "runtime": "onnx",
        "architecture": model_type,
        "mode": mode,
        "profile": {"mode": mode, "assistant_name": "Veyra"},
        "quantized": _is_quantized(onnx_file or info.onnx_path.name),
    }


def recommended_onnx_file(files: list[str]) -> str:
    priorities = ("model_int8.onnx", "model_quantized.onnx", "model_q4.onnx", "model.onnx")
    by_name = {PurePosixPath(path).name.casefold(): path for path in files}
    for filename in priorities:
        if filename in by_name:
            return by_name[filename]
    return sorted(files)[0]


def onnx_variants(files: list[str]) -> list[str]:
    onnx_files = sorted(path for path in files if path.lower().endswith(".onnx"))
    model_files = [path for path in onnx_files if PurePosixPath(path).stem.casefold().startswith("model")]
    return model_files or onnx_files


def variant_name(onnx_file: str) -> str:
    stem = PurePosixPath(onnx_file).stem
    if stem.casefold() == "model":
        return "default"
    if stem.casefold().startswith("model_"):
        return stem[6:]
    return stem


def fetched_model_name(repo_id: str, onnx_file: str, variant_count: int = 2) -> str:
    base = safe_model_name(repo_id)
    variant = variant_name(onnx_file)
    if variant_count <= 1 or variant == "default":
        return base
    return safe_model_name(f"{base}-{variant}")


def _is_quantized(path: str) -> bool:
    name = PurePosixPath(path).stem.casefold()
    return any(marker in name for marker in ("int8", "uint8", "quant", "q4", "bnb"))


def _onnx_download_files(files: list[str], selected: str, variants: list[str]) -> list[str]:
    selected_dir = PurePosixPath(selected).parent
    variant_set = set(variants)
    result = [selected]
    for path in files:
        candidate = PurePosixPath(path)
        if path in variant_set or candidate.parent != selected_dir:
            continue
        lower = path.casefold()
        if lower.endswith(".onnx") or lower.startswith(selected.casefold() + ".") or lower.startswith(
            selected.casefold() + "_"
        ):
            result.append(path)
    return sorted(set(result))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        import json
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
