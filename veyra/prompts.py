from __future__ import annotations


CHATML_END = "<|im_end|>"
PROMPT_MODES = ("base", "chatml", "qwen", "gemma", "mistral", "llama3")


def format_prompt(
    user_text: str,
    mode: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
) -> str:
    mode = normalize_mode(mode)
    if mode == "base":
        return user_text
    if mode == "gemma":
        return format_gemma(user_text, history or [], system_prompt)
    if mode == "mistral":
        return format_mistral(user_text, history or [], system_prompt)
    if mode == "llama3":
        return format_llama3(user_text, history or [], system_prompt)
    if mode == "qwen":
        return format_qwen(user_text, history or [], system_prompt)
    return format_chatml(user_text, history or [], system_prompt)


def format_chatml(user_text: str, history: list[dict[str, str]], system_prompt: str | None = None) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}{CHATML_END}\n")
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        parts.append(f"<|im_start|>{role}\n{message.get('content', '')}{CHATML_END}\n")
    parts.append(f"<|im_start|>user\n{user_text}{CHATML_END}\n<|im_start|>assistant\n")
    return "".join(parts)


def format_qwen(user_text: str, history: list[dict[str, str]], system_prompt: str | None = None) -> str:
    return format_chatml(user_text, history, system_prompt)


def format_gemma(user_text: str, history: list[dict[str, str]], system_prompt: str | None = None) -> str:
    parts: list[str] = []
    pending_system = system_prompt.strip() + "\n\n" if system_prompt else ""
    for message in history:
        role = message.get("role")
        if role == "user":
            content = pending_system + message.get("content", "")
            pending_system = ""
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{message.get('content', '')}<end_of_turn>\n")
    parts.append(f"<start_of_turn>user\n{pending_system}{user_text}<end_of_turn>\n<start_of_turn>model\n")
    return "".join(parts)


def format_mistral(user_text: str, history: list[dict[str, str]], system_prompt: str | None = None) -> str:
    parts: list[str] = []
    pending_system = system_prompt.strip() + "\n\n" if system_prompt else ""
    for message in history:
        role = message.get("role")
        if role == "user":
            content = pending_system + message.get("content", "")
            pending_system = ""
            parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f" {message.get('content', '')}</s>")
    parts.append(f"[INST] {pending_system}{user_text} [/INST]")
    return "".join(parts)


def format_llama3(user_text: str, history: list[dict[str, str]], system_prompt: str | None = None) -> str:
    parts = ["<|begin_of_text|>"]
    if system_prompt:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{message.get('content', '')}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "chatml").lower()
    aliases = {"raw": "base", "qwen2": "qwen", "qwen2.5": "qwen", "qwen3": "qwen", "qwen3next": "qwen"}
    value = aliases.get(value, value)
    return value if value in PROMPT_MODES else "chatml"


def infer_prompt_mode(config: dict | None = None, tokenizer_config: dict | None = None) -> str:
    config = config or {}
    tokenizer_config = tokenizer_config or {}
    haystack = " ".join(
        str(x).lower()
        for x in [
            config.get("model_type", ""),
            " ".join(config.get("architectures", []) or []),
            tokenizer_config.get("chat_template", ""),
        ]
    )
    if "gemma" in haystack:
        return "gemma"
    if "mistral" in haystack or "[inst]" in haystack:
        return "mistral"
    if "llama-3" in haystack or "start_header_id" in haystack:
        return "llama3"
    if "qwen" in haystack:
        return "qwen"
    if "<|im_start|>" in haystack:
        return "chatml"
    return "chatml"
