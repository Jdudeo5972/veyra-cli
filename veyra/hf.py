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
    "*.py",
    "**/*.py",
]
TRANSFORMERS_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]


def list_veyra_models() -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    api = HfApi()
    try:
        repos = list(api.list_models(author=HF_ORG, full=True))
    except Exception as exc:
        raise RuntimeError(hub_access_error(HF_ORG, exc)) from exc
    choices: list[dict[str, Any]] = []
    for repo in repos:
        repo_id = repo.modelId
        try:
            files = list_repo_files(repo_id)
        except Exception:
            continue
        onnx_files = onnx_variants(files)
        has_transformers = has_safetensors(files)
        if (not onnx_files and not has_transformers) or "tokenizer.json" not in files:
            continue
        choices.append(
            {
                "repo_id": repo_id,
                "files": files,
                "onnx_files": onnx_files,
                "has_transformers": has_transformers,
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
        raise RuntimeError(hub_access_error(repo_id, exc)) from exc
    if "tokenizer.json" not in files:
        raise RuntimeError(
            f"{repo_id} does not contain tokenizer.json. Veyra requires a fast tokenizer.json beside the model weights."
        )
    variants = onnx_variants(files)
    transformers = has_safetensors(files)
    if not variants and not transformers:
        raise RuntimeError(f"No ONNX or Safetensors model files found in {repo_id}.")
    return {
        "repo_id": repo_id,
        "files": files,
        "onnx_files": variants,
        "has_transformers": transformers,
        "downloads": None,
    }


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
    runtime: str = "onnx",
) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, list_repo_files, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for fetching models.") from exc

    try:
        files = list_repo_files(repo_id, revision=revision)
    except Exception as exc:
        raise RuntimeError(hub_access_error(repo_id, exc)) from exc
    if "tokenizer.json" not in files:
        raise RuntimeError(f"{repo_id} does not contain tokenizer.json.")
    available_onnx = onnx_variants(files)
    runtime = runtime.lower()
    if runtime == "transformers":
        if not has_safetensors(files):
            raise RuntimeError(f"No Safetensors model files found in {repo_id}.")
        onnx_file = None
        allow_patterns = [*TRANSFORMERS_WEIGHT_PATTERNS, *MODEL_METADATA_PATTERNS]
    else:
        if not available_onnx:
            raise RuntimeError(f"No ONNX files found in {repo_id}.")
        if onnx_file is None:
            onnx_file = recommended_onnx_file(available_onnx)
        if onnx_file not in available_onnx:
            raise RuntimeError(f"ONNX file '{onnx_file}' was not found in {repo_id}.")
        allow_patterns = [*_onnx_download_files(files, onnx_file, available_onnx), *MODEL_METADATA_PATTERNS]

    name = fetched_model_name(repo_id, onnx_file, len(available_onnx), runtime=runtime)
    target = MODELS_DIR / name
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise RuntimeError(hub_access_error(repo_id, exc)) from exc
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
    runtime: str = "onnx",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    root = Path(path)
    config = _read_json(root / "config.json")
    tokenizer_config = _read_json(root / "tokenizer_config.json")
    has_template = (root / "chat_template.jinja").exists() or bool(tokenizer_config.get("chat_template"))
    if has_template:
        mode = "template"
    elif config.get("is_encoder_decoder"):
        mode = "base"
    else:
        mode = infer_prompt_mode(config, tokenizer_config)
    architectures = config.get("architectures") or []
    model_type = config.get("model_type") or (architectures[0] if architectures else "unknown")
    selected_onnx = None
    if runtime == "onnx":
        info = inspect_model(path)
        model_type = info.model_type or (info.architecture or model_type)
        selected_onnx = onnx_file or info.onnx_path.relative_to(root).as_posix()
    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_commit": commit,
        "onnx_file": selected_onnx,
        "path": str(root.expanduser().resolve()),
        "runtime": runtime,
        "trust_remote_code": bool(trust_remote_code),
        "architecture": str(model_type).lower(),
        "mode": mode,
        "profile": {"mode": mode, "assistant_name": "Veyra"},
        "quantized": _is_quantized(selected_onnx or ""),
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


def has_safetensors(files: list[str]) -> bool:
    return any(path.lower().endswith(".safetensors") for path in files)


def variant_name(onnx_file: str) -> str:
    stem = PurePosixPath(onnx_file).stem
    if stem.casefold() == "model":
        return "default"
    if stem.casefold().startswith("model_"):
        return stem[6:]
    return stem


def fetched_model_name(
    repo_id: str,
    onnx_file: str | None,
    variant_count: int = 2,
    runtime: str = "onnx",
) -> str:
    base = safe_model_name(repo_id)
    if runtime == "transformers":
        return base
    if not onnx_file:
        return safe_model_name(f"{base}-onnx")
    variant = variant_name(onnx_file)
    suffix = "onnx" if variant == "default" else f"onnx-{variant}"
    return safe_model_name(f"{base}-{suffix}")


def _is_quantized(path: str) -> bool:
    name = PurePosixPath(path).stem.casefold()
    return any(marker in name for marker in ("int8", "uint8", "quant", "q4", "bnb"))


def hub_access_error(repo_id: str, exc: Exception) -> str:
    message = str(exc)
    lowered = message.casefold()
    if "401" in lowered or "unauthorized" in lowered or "invalid user token" in lowered:
        return (
            f"Could not authenticate while accessing {repo_id}. Use `veyra auth login` or `/hf login` "
            "with a read-only Hugging Face token."
        )
    if "403" in lowered or "forbidden" in lowered or "gated" in lowered or "access denied" in lowered:
        return (
            f"Access to {repo_id} was denied. Accept the model's access terms on Hugging Face, then grant "
            "your read-only token access to that model."
        )
    return f"Could not access Hugging Face model {repo_id}: {message}"


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
