from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tokenizers import Tokenizer

from .inspect import SUPPORTED_INPUTS, find_onnx_file, is_past_input, missing_cache_fields, parse_cache_name


class UnsupportedModelError(RuntimeError):
    def __init__(self, unsupported_inputs: list[str]) -> None:
        self.unsupported_inputs = unsupported_inputs
        super().__init__(
            "Unsupported required ONNX inputs: "
            + ", ".join(unsupported_inputs)
            + ". Run `veyra inspect PATH` for details."
        )


class CacheConfigError(RuntimeError):
    pass


class CacheOutputError(RuntimeError):
    pass


class OnnxCausalLMRunner:
    def __init__(self, model_dir: str | Path, threads: int = 2) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to run ONNX models.") from exc

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.onnx_path = find_onnx_file(self.model_dir)
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.model_dir}")
        self.config = self._read_json("config.json")
        self.generation_config = self._read_json("generation_config.json")
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.eos_ids = self._eos_ids()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads or 2))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.input_by_name = {i.name: i for i in self.inputs}
        self.output_names = [o.name for o in self.outputs]
        self.past_inputs = self._past_inputs()
        self.uses_cache = bool(self.past_inputs)
        unsupported = [i.name for i in self.inputs if i.name not in SUPPORTED_INPUTS and not is_past_input(i.name)]
        if unsupported:
            raise UnsupportedModelError(unsupported)
        if self.uses_cache:
            self.cache_meta = self._cache_meta()
            self.present_output_indices = self._present_output_indices()
        else:
            self.cache_meta = {}
            self.present_output_indices = {}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if not ids:
            ids = [self._bos_id()]
        generated: list[int] = []
        previous_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        cache = self._empty_cache() if self.uses_cache else {}

        for _ in range(max(0, int(max_new_tokens))):
            if self.uses_cache and generated:
                input_ids = np.asarray([[generated[-1]]], dtype=np.int64)
                total_len = len(ids) + len(generated)
                position_start = total_len - 1
            else:
                input_ids = np.asarray([ids], dtype=np.int64) if self.uses_cache else np.asarray([ids + generated], dtype=np.int64)
                total_len = input_ids.shape[1]
                position_start = 0
            feed = self._feed(input_ids, total_len=total_len, position_start=position_start, cache=cache)
            result = self.session.run(None, feed)
            if self.uses_cache:
                cache = self._update_cache(result)
            logits = np.asarray(result[0])[0, -1, :].astype(np.float64)
            next_id = sample_next_token(
                logits,
                generated,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )
            if next_id in self.eos_ids:
                break
            generated.append(next_id)
            text = self.tokenizer.decode(ids + generated, skip_special_tokens=False)
            if text.startswith(previous_text):
                delta = text[len(previous_text) :]
            else:
                delta = self.tokenizer.decode([next_id], skip_special_tokens=False)
            previous_text = text
            if delta:
                yield delta

    def _feed(
        self,
        input_ids: np.ndarray,
        total_len: int,
        position_start: int,
        cache: dict[tuple[int, str], np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        step_len = input_ids.shape[1]
        feed: dict[str, np.ndarray] = {}
        for input_info in self.inputs:
            name = input_info.name
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = np.ones((1, total_len), dtype=np.int64)
            elif name == "position_ids":
                feed[name] = np.arange(position_start, position_start + step_len, dtype=np.int64)[None, :]
            elif cache is not None and is_past_input(name):
                parsed = parse_cache_name(name)
                if parsed is not None:
                    feed[name] = cache[parsed]
        return feed

    def _past_inputs(self) -> dict[tuple[int, str], Any]:
        parsed: dict[tuple[int, str], Any] = {}
        for input_info in self.inputs:
            item = parse_cache_name(input_info.name)
            if item is not None:
                parsed[item] = input_info
        return parsed

    def _cache_meta(self) -> dict[str, Any]:
        missing = missing_cache_fields(self.config)
        if missing:
            raise CacheConfigError(
                "This ONNX model requires KV-cache inputs, but config.json is missing: "
                + ", ".join(missing)
                + ". Required fields are num_hidden_layers, num_key_value_heads, and head_dim."
            )
        return {
            "num_hidden_layers": int(self.config["num_hidden_layers"]),
            "num_key_value_heads": int(self.config["num_key_value_heads"]),
            "head_dim": int(self.config["head_dim"]),
            "dtype": _numpy_dtype(self.config.get("dtype")),
        }

    def _empty_cache(self) -> dict[tuple[int, str], np.ndarray]:
        layers = int(self.cache_meta["num_hidden_layers"])
        cache: dict[tuple[int, str], np.ndarray] = {}
        for layer in range(layers):
            for kind in ("key", "value"):
                input_info = self.past_inputs.get((layer, kind))
                if input_info is None:
                    raise CacheConfigError(f"Missing required cache input past_key_values.{layer}.{kind}")
                shape = self._empty_cache_shape(input_info)
                cache[(layer, kind)] = np.zeros(shape, dtype=self.cache_meta["dtype"])
        return cache

    def _empty_cache_shape(self, input_info: Any) -> tuple[int, int, int, int]:
        heads = int(self.cache_meta["num_key_value_heads"])
        head_dim = int(self.cache_meta["head_dim"])
        raw_shape = list(getattr(input_info, "shape", []) or [])
        if len(raw_shape) == 4:
            dims: list[int | None] = []
            for dim in raw_shape:
                dims.append(int(dim) if isinstance(dim, int) and dim > 0 else None)
            if dims[1] == heads and dims[3] == head_dim:
                return (1, heads, 0, head_dim)
            if dims[2] == heads and dims[3] == head_dim:
                return (1, 0, heads, head_dim)
            if dims[1] == head_dim and dims[3] == heads:
                return (1, head_dim, 0, heads)
        return (1, heads, 0, head_dim)

    def _present_output_indices(self) -> dict[tuple[int, str], int]:
        indices: dict[tuple[int, str], int] = {}
        for idx, name in enumerate(self.output_names):
            parsed = parse_cache_name(name)
            if parsed is not None:
                indices[parsed] = idx
        if not indices:
            shapes = "\n".join(f"  {o.name}: {getattr(o, 'type', 'unknown')} {getattr(o, 'shape', [])}" for o in self.outputs)
            raise CacheOutputError(
                "This ONNX model requires KV-cache inputs, but Veyra could not find named present key/value outputs. "
                "Supported output names include present.{i}.key, present.{i}.value, "
                "present_key_values.{i}.key, and past_key_values.{i}.value.\nOutputs:\n"
                + shapes
            )
        missing: list[str] = []
        layers = int(self.cache_meta["num_hidden_layers"])
        for layer in range(layers):
            for kind in ("key", "value"):
                if (layer, kind) not in indices:
                    missing.append(f"{layer}.{kind}")
        if missing:
            raise CacheOutputError("Missing present KV-cache outputs for: " + ", ".join(missing))
        return indices

    def _update_cache(self, result: list[np.ndarray]) -> dict[tuple[int, str], np.ndarray]:
        return {key: np.asarray(result[idx]) for key, idx in self.present_output_indices.items()}

    def _read_json(self, name: str) -> dict:
        path = self.model_dir / name
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for source in (self.config, self.generation_config):
            value = source.get("eos_token_id")
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, list):
                ids.update(int(v) for v in value if isinstance(v, int))
        vocab = self.tokenizer.get_vocab()
        for token in ("<|im_end|>", "</s>", "<eos>"):
            if token in vocab:
                ids.add(vocab[token])
        return ids

    def _bos_id(self) -> int:
        value = self.config.get("bos_token_id")
        return int(value) if isinstance(value, int) else 0


def sample_next_token(
    logits: np.ndarray,
    generated: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    if repetition_penalty and repetition_penalty != 1.0:
        for token_id in set(generated):
            if 0 <= token_id < logits.shape[0]:
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-6)

    if top_k > 0 and top_k < logits.shape[0]:
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)

    probs = _softmax(logits)
    if 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep = cumulative <= top_p
        keep[0] = True
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()

    return int(np.random.choice(np.arange(probs.shape[0]), p=probs))


def _softmax(logits: np.ndarray) -> np.ndarray:
    finite = np.isfinite(logits)
    if not finite.any():
        logits = np.zeros_like(logits)
    max_logit = np.max(logits[finite]) if finite.any() else 0.0
    exp = np.exp(np.where(finite, logits - max_logit, -np.inf))
    total = exp.sum()
    if total <= 0:
        return np.ones_like(exp) / exp.shape[0]
    return exp / total


def _numpy_dtype(name: Any) -> np.dtype:
    if str(name).lower() in {"float16", "fp16"}:
        return np.dtype(np.float16)
    if str(name).lower() in {"bfloat16", "bf16"}:
        return np.dtype(np.float32)
    return np.dtype(np.float32)
