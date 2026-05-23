from __future__ import annotations


CHATML_END = "<|im_end|>"


def format_prompt(
    user_text: str,
    mode: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
) -> str:
    if mode == "base":
        return user_text
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
