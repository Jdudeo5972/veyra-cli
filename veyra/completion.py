from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


TOP = {
    "/model": "Show or change current model",
    "/mode": "Show or change prompt mode",
    "/chat": "Manage saved chats",
    "/theme": "Change CLI color theme",
    "/device": "Select ONNX Runtime device",
    "/stats": "Toggle response stats",
    "/autoload": "Toggle model autoload on startup",
    "/temp": "Set or show sampling temperature",
    "/tokens": "Set or show max new tokens",
    "/topk": "Set or show top-k sampling",
    "/topp": "Set or show top-p sampling",
    "/repetition": "Set or show repetition penalty",
    "/system": "Set or clear the system prompt",
    "/update": "Show CLI update instructions",
    "/status": "Show runtime status",
    "/help": "Show commands",
    "/exit": "Exit Veyra",
    "/quit": "Exit Veyra",
    "/clear": "Clear the screen",
}

MODEL = {
    "list": "List installed models",
    "use": "Switch current model",
    "fetch": "Fetch ONNX models from veyra-ai",
    "refresh": "Refresh remote model list",
    "update": "Update installed model files",
    "add": "Add local ONNX model",
    "inspect": "Inspect current model",
    "remove": "Remove model from registry",
}

MODE = {
    "base": "Raw completion mode",
    "chatml": "ChatML conversation mode",
    "qwen": "Qwen ChatML-style mode",
    "gemma": "Gemma start_of_turn mode",
    "mistral": "Mistral [INST] mode",
    "llama3": "Llama 3 header mode",
}
AUTOLOAD = {"on": "Autoload current model", "off": "Start shell without loading model"}
THEME = {
    "list": "List themes",
    "veyra": "Pink and gray default theme",
    "warm": "Warm amber theme",
    "green": "Green terminal theme",
    "blue": "Blue and cyan theme",
    "mono": "No decorative colors",
}
DEVICE = {
    "list": "List available devices",
    "cpu": "CPU execution provider",
    "cuda": "NVIDIA CUDA provider",
    "directml": "Windows DirectML provider",
    "coreml": "Apple Core ML provider",
    "openvino": "Intel OpenVINO provider",
    "rocm": "AMD ROCm provider",
    "tensorrt": "NVIDIA TensorRT provider",
}
STATS = {"on": "Show response stats", "off": "Hide response stats"}
CHAT = {
    "new": "Start a new chat",
    "list": "List saved chats",
    "load": "Load a saved chat",
    "rename": "Rename current chat",
    "export": "Export current chat",
    "path": "Show chat file path",
}
MODEL_UPDATE = {"all": "Update all Hugging Face models"}
CHAT_EXPORT = {"markdown": "Export current chat as Markdown"}


class VeyraCompleter(Completer):
    def __init__(self, model_names_cb, chat_names_cb=None) -> None:
        self.model_names_cb = model_names_cb
        self.chat_names_cb = chat_names_cb or (lambda: [])

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parts = text.split()
        if text.endswith(" "):
            parts.append("")

        if len(parts) <= 1 and not text.endswith(" "):
            yield from _dict_completions(TOP, text, -len(text))
            return

        command = parts[0]
        current = parts[-1]
        start = -len(current)

        if command == "/model":
            if len(parts) == 2:
                yield from _dict_completions(MODEL, current, start)
            elif len(parts) == 3 and parts[1] in {"use", "remove"}:
                for name in self.model_names_cb():
                    if name.startswith(current):
                        yield Completion(name, start_position=start)
            elif len(parts) == 3 and parts[1] == "update":
                yield from _dict_completions(MODEL_UPDATE, current, start)
        elif command == "/mode" and len(parts) == 2:
            yield from _dict_completions(MODE, current, start)
        elif command == "/autoload" and len(parts) == 2:
            yield from _dict_completions(AUTOLOAD, current, start)
        elif command == "/theme" and len(parts) == 2:
            yield from _dict_completions(THEME, current, start)
        elif command == "/device" and len(parts) == 2:
            yield from _dict_completions(DEVICE, current, start)
        elif command == "/stats" and len(parts) == 2:
            yield from _dict_completions(STATS, current, start)
        elif command == "/chat" and len(parts) == 2:
            yield from _dict_completions(CHAT, current, start)
        elif command == "/chat" and len(parts) == 3 and parts[1] in {"load", "rename"}:
            for name in self.chat_names_cb():
                if name.startswith(current):
                    yield Completion(name, start_position=start)
        elif command == "/chat" and len(parts) == 3 and parts[1] == "export":
            yield from _dict_completions(CHAT_EXPORT, current, start)


def _dict_completions(values: dict[str, str], current: str, start: int):
    for key, description in values.items():
        if key.startswith(current):
            yield Completion(key, start_position=start, display_meta=description)
