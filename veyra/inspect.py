from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_INPUTS = {"input_ids", "inputs_embeds", "attention_mask", "position_ids", "num_logits_to_keep"}
PAST_RE = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")
PRESENT_RE = re.compile(r"^(?:present|present_key_values|past_key_values)\.(\d+)\.(key|value)$")
STATE_RE = re.compile(r"^past_(conv|recurrent)\.(\d+)$")
CACHE_CONFIG_FIELDS = ("num_hidden_layers", "num_key_value_heads", "head_dim")


@dataclass
class TensorInfo:
    name: str
    shape: list[Any]
    type: str
    required: bool = True


@dataclass
class ModelInspection:
    model_dir: Path
    onnx_path: Path
    tokenizer_path: Path
    config_path: Path | None
    architecture: str | None
    model_type: str | None
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    unsupported_inputs: list[str]
    cache_inputs: list[str]
    cache_outputs: list[str]
    missing_cache_config: list[str]

    @property
    def supported(self) -> bool:
        return not self.unsupported_inputs and not self.missing_cache_config


def find_onnx_file(model_dir: str | Path) -> Path:
    root = Path(model_dir).expanduser().resolve()
    if root.is_file() and root.suffix == ".onnx":
        return root
    files = sorted(root.glob("*.onnx"))
    if not files:
        files = sorted(root.rglob("*.onnx"))
    if not files:
        raise FileNotFoundError(f"No .onnx file found in {root}")
    return files[0]


def inspect_model(model_dir: str | Path) -> ModelInspection:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to inspect ONNX models.") from exc

    root = Path(model_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    onnx_path = find_onnx_file(root)
    tokenizer_path = root / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {root}")

    config_path = root / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    arch = _architecture(config)
    model_type = config.get("model_type")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inputs = [_tensor_info(i) for i in session.get_inputs()]
    outputs = [_tensor_info(o) for o in session.get_outputs()]
    cache_inputs = [i.name for i in inputs if is_past_input(i.name)]
    cache_outputs = [o.name for o in outputs if is_present_output(o.name)]
    unsupported = [
        i.name
        for i in inputs
        if i.required and i.name not in SUPPORTED_INPUTS and not is_past_input(i.name) and not is_state_input(i.name)
    ]
    missing_cache_config = missing_cache_fields(config) if cache_inputs else []
    return ModelInspection(
        root,
        onnx_path,
        tokenizer_path,
        config_path if config_path.exists() else None,
        arch,
        model_type,
        inputs,
        outputs,
        unsupported,
        cache_inputs,
        cache_outputs,
        missing_cache_config,
    )


def format_inspection(info: ModelInspection) -> str:
    lines = [
        f"ONNX: {info.onnx_path}",
        f"Tokenizer: {info.tokenizer_path}",
        f"Architecture: {info.architecture or 'unknown'}",
        f"Model type: {info.model_type or 'unknown'}",
        "",
        "Inputs:",
    ]
    lines.extend(f"  {t.name}: {t.type} {t.shape}" for t in info.inputs)
    lines.append("")
    lines.append("Outputs:")
    lines.extend(f"  {t.name}: {t.type} {t.shape}" for t in info.outputs)
    lines.append("")
    if info.cache_inputs:
        lines.append(f"KV cache inputs: {len(info.cache_inputs)}")
        lines.append(f"KV cache outputs: {len(info.cache_outputs)}")
        if info.missing_cache_config:
            lines.append("Missing cache config: " + ", ".join(info.missing_cache_config))
        else:
            lines.append("KV cache metadata: available")
        lines.append("")
    if info.supported:
        lines.append("Status: supported")
    else:
        lines.append("Status: unsupported")
        if info.unsupported_inputs:
            lines.append("Unsupported required inputs: " + ", ".join(info.unsupported_inputs))
        if info.missing_cache_config:
            lines.append("Missing required config fields: " + ", ".join(info.missing_cache_config))
    return "\n".join(lines)


def _tensor_info(value: Any) -> TensorInfo:
    shape = list(getattr(value, "shape", []) or [])
    return TensorInfo(
        name=value.name,
        shape=shape,
        type=getattr(value, "type", "unknown"),
        required=not _has_dynamic_optional(shape),
    )


def _has_dynamic_optional(shape: list[Any]) -> bool:
    return False


def is_past_input(name: str) -> bool:
    return PAST_RE.match(name) is not None


def is_present_output(name: str) -> bool:
    return PRESENT_RE.match(name) is not None


def parse_cache_name(name: str) -> tuple[int, str] | None:
    match = PAST_RE.match(name) or PRESENT_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def missing_cache_fields(config: dict[str, Any]) -> list[str]:
    return [field for field in CACHE_CONFIG_FIELDS if not isinstance(_config_value(config, field), int)]


def is_state_input(name: str) -> bool:
    return STATE_RE.match(name) is not None


def _config_value(config: dict[str, Any], field: str) -> Any:
    if isinstance(config.get(field), int):
        return config[field]
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get(field), int):
        return text_config[field]
    if field == "head_dim":
        hidden = _config_value(config, "hidden_size")
        heads = _config_value(config, "num_attention_heads")
        if isinstance(hidden, int) and isinstance(heads, int) and heads:
            return hidden // heads
    return config.get(field)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _architecture(config: dict[str, Any]) -> str | None:
    archs = config.get("architectures")
    if isinstance(archs, list) and archs:
        return str(archs[0])
    if config.get("model_type"):
        return str(config["model_type"])
    return None
