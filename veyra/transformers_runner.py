from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .prompts import format_prompt
from .runner import COMMON_STOP_TOKENS, model_context_length, sample_next_token


class TransformersRunner:
    def __init__(self, model_dir: str | Path, device: str = "cpu", trust_remote_code: bool = False) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers models require the optional runtime. Install it with "
                "`uv tool install 'veyra[transformers]'` or `pip install 'veyra[transformers]'`."
            ) from exc

        self.model_dir = Path(model_dir).expanduser().resolve()
        if not (self.model_dir / "tokenizer.json").exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.model_dir}")
        if not list(self.model_dir.glob("*.safetensors")):
            raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")
        if device != "cpu":
            raise RuntimeError(
                f"Transformers runtime currently supports Veyra's cpu device mode, not '{device}'. "
                "Use `/device cpu` or select an ONNX model for that provider."
            )

        self.torch = torch
        self.device = "cpu"
        self.config = self._read_json("config.json")
        self.tokenizer_config = self._read_json("tokenizer_config.json")
        auto_config = AutoConfig.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        self.is_encoder_decoder = bool(getattr(auto_config, "is_encoder_decoder", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        template_path = self.model_dir / "chat_template.jinja"
        if template_path.exists():
            self.tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        model_class = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()
        self.max_context_length = model_context_length(self.config, self.tokenizer_config)
        self.uses_cache = bool(getattr(self.model.config, "use_cache", True))

    def format_conversation(
        self,
        user_text: str,
        mode: str,
        history: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        if mode != "template":
            return format_prompt(user_text, mode, history=history, system_prompt=system_prompt)
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("Prompt mode is template, but this tokenizer does not provide a chat template.")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            {"role": item["role"], "content": item.get("content", "")}
            for item in history
            if item.get("role") in {"user", "assistant"}
        )
        messages.append({"role": "user", "content": user_text})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=self.is_encoder_decoder))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        context_length: int | None = None,
    ) -> Iterator[str]:
        torch = self.torch
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.is_encoder_decoder,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        prompt_len = int(input_ids.shape[1])
        limit = int(context_length) if context_length is not None else self.max_context_length
        if self.max_context_length and limit and limit > self.max_context_length:
            raise ValueError(f"Context length {limit} exceeds this model's limit of {self.max_context_length}.")
        required = prompt_len if self.is_encoder_decoder else prompt_len + max(0, int(max_new_tokens))
        if limit and required > limit:
            detail = f"Prompt ({prompt_len} tokens)"
            if not self.is_encoder_decoder:
                detail += f" plus output budget ({max_new_tokens})"
            raise ValueError(f"{detail} exceeds context length {limit}.")

        if self.is_encoder_decoder:
            yield from self._generate_seq2seq(
                input_ids,
                attention_mask,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
            )
            return
        yield from self._generate_causal(
            input_ids,
            attention_mask,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
        )

    def _generate_causal(
        self,
        input_ids,
        attention_mask,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int | None,
    ) -> Iterator[str]:
        torch = self.torch
        generated: list[int] = []
        previous_text = ""
        rng = np.random.default_rng(seed)
        past = None
        eos_ids = self._eos_ids()
        with torch.inference_mode():
            for _ in range(max(0, int(max_new_tokens))):
                step_ids = input_ids if past is None else input_ids[:, -1:]
                outputs = self.model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[0, -1].float().cpu().numpy().astype(np.float64)
                next_id = sample_next_token(
                    logits,
                    generated,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    rng=rng,
                )
                if next_id in eos_ids:
                    break
                generated.append(next_id)
                next_tensor = torch.tensor([[next_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, next_tensor), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_tensor)), dim=1)
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                delta = text[len(previous_text) :] if text.startswith(previous_text) else self.tokenizer.decode([next_id])
                previous_text = text
                if delta:
                    yield delta

    def _generate_seq2seq(
        self,
        input_ids,
        attention_mask,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int | None,
    ) -> Iterator[str]:
        torch = self.torch
        start_id = self._decoder_start_id()
        decoder_ids = torch.tensor([[start_id]], dtype=input_ids.dtype, device=self.device)
        generated: list[int] = []
        previous_text = ""
        rng = np.random.default_rng(seed)
        past = None
        eos_ids = self._eos_ids()
        with torch.inference_mode():
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            for _ in range(max(0, int(max_new_tokens))):
                step_ids = decoder_ids if past is None else decoder_ids[:, -1:]
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=step_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[0, -1].float().cpu().numpy().astype(np.float64)
                next_id = sample_next_token(
                    logits,
                    generated,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    rng=rng,
                )
                if next_id in eos_ids:
                    break
                generated.append(next_id)
                next_tensor = torch.tensor([[next_id]], dtype=decoder_ids.dtype, device=self.device)
                decoder_ids = torch.cat((decoder_ids, next_tensor), dim=1)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                delta = (
                    text[len(previous_text) :]
                    if text.startswith(previous_text)
                    else self.tokenizer.decode([next_id], skip_special_tokens=True)
                )
                previous_text = text
                if delta:
                    yield delta

    def _decoder_start_id(self) -> int:
        for source in (getattr(self.model, "generation_config", None), self.model.config, self.tokenizer):
            for field in ("decoder_start_token_id", "bos_token_id", "pad_token_id"):
                value = getattr(source, field, None)
                if isinstance(value, int):
                    return value
        raise RuntimeError("Seq2Seq model does not define decoder_start_token_id, bos_token_id, or pad_token_id.")

    def _eos_ids(self) -> set[int]:
        values: list[int | None] = []
        for value in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.model.config, "eos_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
        ):
            values.extend(value if isinstance(value, list) else [value])
        vocab = self.tokenizer.get_vocab()
        values.extend(vocab[token] for token in COMMON_STOP_TOKENS if token in vocab)
        return {int(item) for item in values if isinstance(item, int)}

    def _read_json(self, name: str) -> dict:
        try:
            with (self.model_dir / name).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}


TransformersCausalLMRunner = TransformersRunner
