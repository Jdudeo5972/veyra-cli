from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from . import __version__
from .chat_store import ChatStore
from .completion import VeyraCompleter
from .hf import download_model, list_veyra_models, registry_entry
from .inspect import format_inspection, inspect_model
from .prompts import format_prompt
from .registry import HISTORY_PATH, current_model_entry, load_config, models, register_model, remove_model, safe_model_name, save_config
from .runner import OnnxCausalLMRunner, UnsupportedModelError, available_devices, device_rows, normalize_device, provider_for_device
from .theme import THEMES, get_theme, normalize_theme


class VeyraShell:
    def __init__(self, args: argparse.Namespace) -> None:
        _prefer_utf8_stdio()
        self.args = args
        self.config = load_config()
        self.theme = get_theme(self.config.get("theme"))
        self.runner: OnnxCausalLMRunner | None = None
        self.load_error: str | None = None
        self.system_prompt: str | None = None
        self.chat = ChatStore.latest() if args.continue_chat else None
        if self.chat is None:
            self.chat = ChatStore.new(self.config.get("current_model"), self.config.get("current_mode", "chatml"))
        self.kb = KeyBindings()
        self.kb.add("enter")(self._accept_completion_or_line)
        self.kb.add("tab")(self._accept_completion_or_menu)
        self.session = self._make_session() if sys.stdin.isatty() and sys.stdout.isatty() else None

    def _make_session(self) -> PromptSession:
        return PromptSession(
            history=FileHistory(str(HISTORY_PATH)),
            completer=VeyraCompleter(lambda: sorted(models(self.config)), self.chat_names),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
            key_bindings=self.kb,
        )

    def _accept_completion_or_line(self, event) -> None:
        buffer = event.current_buffer
        completion = self._active_completion(buffer)
        if completion:
            buffer.apply_completion(completion)
        else:
            buffer.validate_and_handle()

    def _accept_completion_or_menu(self, event) -> None:
        buffer = event.current_buffer
        completion = self._active_completion(buffer)
        if completion:
            buffer.apply_completion(completion)
        elif buffer.suggestion:
            buffer.insert_text(buffer.suggestion.text)
        else:
            buffer.start_completion(select_first=True)

    def _active_completion(self, buffer):
        state = buffer.complete_state
        if not state:
            return None
        if state.current_completion:
            return state.current_completion
        if state.completions:
            return state.completions[0]
        return None

    def run(self) -> int:
        state = "unloaded"
        if not models(self.config):
            state = "no model"
        elif not self.args.no_load and self.config.get("autoload", True):
            loaded = self.load_current_model(quiet=True)
            state = "ready" if loaded else "failed"
        self.banner(state)
        if self.load_error:
            self.error(self.load_error)
        self.ready_message()
        while True:
            try:
                text = self.read_input()
            except (EOFError, KeyboardInterrupt):
                print("")
                return 0
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if self.handle_command(text):
                    return 0
            else:
                self.handle_prompt(text)

    def read_input(self) -> str:
        if self.session is not None:
            return self.session.prompt(self.theme.prompt("user_prompt", "You \u203a "))
        return input("You \u203a ")

    def banner(self, state: str) -> None:
        width = 58
        title = f" Veyra v{__version__} "
        border_role = "border"
        top = (
            self.theme.text(border_role, "\u256d\u2500")
            + self.theme.text("title", title)
            + self.theme.text(border_role, "\u2500" * (width - len(title) - 2) + "\u256e")
        )
        print(top)
        for segments in self.banner_segments(state):
            self.print_box_line(segments, width)
        print(self.theme.text(border_role, "\u2570" + "\u2500" * (width - 2) + "\u256f"))

    def print_box_line(self, segments: list[tuple[str, str]], width: int) -> None:
        plain = "".join(text for _, text in segments)
        body = "".join(self.theme.text(role, text) for role, text in segments)
        body += " " * max(0, width - 4 - len(plain))
        border = self.theme.text("border", "\u2502")
        print(f"{border} {body} {border}")

    def ready_message(self) -> None:
        if not models(self.config):
            self.warn("No model installed yet.")
            self.command_hint("Use", ["/model fetch"], "to find ONNX models from veyra-ai.")
            self.command_hint("Use", ["/model add PATH"], "to add a local ONNX model.")
            return
        self.command_hint("type", ["/help", "/model", "/mode", "/device", "/chat", "/exit"], "")

    def banner_segments(self, state: str) -> list[list[tuple[str, str]]]:
        defaults = self.config.get("defaults", {})
        autoload = "on" if self.config.get("autoload", True) else "off"
        status_role = {
            "ready": "status_ready",
            "loading": "status_loading",
            "failed": "status_error",
            "no model": "status_empty",
            "unloaded": "status_empty",
        }.get(state, "status_empty")
        return [
            [(status_role, "\u25cf " + state), ("muted", f" autoload:{autoload}")],
            [("label", "model  "), ("value", self.config.get("current_model") or "none")],
            [("label", "mode   "), ("value", self.config.get("current_mode", "chatml"))],
            [
                ("label", "gen    "),
                (
                    "value",
                    f"tokens:{defaults.get('max_new_tokens', 128)}  temp:{defaults.get('temperature', 0.8)}  "
                    f"top-k:{defaults.get('top_k', 40)}  top-p:{defaults.get('top_p', 1.0)}",
                ),
            ],
            [("label", "       "), ("value", f"repeat:{defaults.get('repetition_penalty', 1.0)}")],
        ]

    def chat_names(self) -> list[str]:
        return [path.stem for path in ChatStore.list()]

    def load_current_model(self, quiet: bool = False) -> bool:
        name, entry = current_model_entry(self.config)
        if not entry:
            return False
        try:
            self.runner = OnnxCausalLMRunner(entry["path"], device=self.config.get("device", "cpu"))
            self.load_error = None
            return True
        except Exception as exc:
            self.runner = None
            self.load_error = f"Could not load {name}: {exc}"
            if not quiet:
                self.error(self.load_error)
            return False

    def handle_prompt(self, text: str) -> None:
        if self.runner is None and not self.load_current_model():
            self.error("Missing model. Use /model fetch or /model add PATH.")
            return
        assert self.runner is not None
        mode = self.config.get("current_mode", "chatml")
        history = self.chat.history() if mode == "chatml" and self.chat else []
        prompt = format_prompt(text, mode, history=history, system_prompt=self.system_prompt)
        defaults = self.config.get("defaults", {})
        if self.chat:
            self.chat.message("user", text)
        print(self.theme.text("assistant_prompt", "Veyra \u203a "), end="", flush=True)
        chunks: list[str] = []
        try:
            for delta in self.runner.generate(prompt, **defaults):
                chunks.append(delta)
                print(delta, end="", flush=True)
        except KeyboardInterrupt:
            self.warn("\n[generation stopped]")
        except UnsupportedModelError as exc:
            self.error(f"\n{exc}")
        except Exception as exc:
            self.error(f"\nGeneration failed: {exc}")
        finally:
            print("")
            if chunks and self.chat:
                self.chat.message("assistant", "".join(chunks))

    def handle_command(self, text: str) -> bool:
        parts = text.split()
        cmd = parts[0]
        args = parts[1:]
        if cmd in {"/exit", "/quit"}:
            return True
        if cmd == "/help":
            self.help()
        elif cmd == "/status":
            self.status()
        elif cmd == "/clear":
            self.clear_visible_screen(force=True)
        elif cmd == "/model":
            self.model_command(args)
        elif cmd == "/mode":
            self.mode_command(args)
        elif cmd == "/theme":
            self.theme_command(args)
        elif cmd == "/device":
            self.device_command(args)
        elif cmd == "/autoload":
            self.autoload_command(args)
        elif cmd in {"/temp", "/tokens", "/topk", "/topp", "/repetition"}:
            self.default_command(cmd, args)
        elif cmd == "/system":
            self.system_prompt = text.removeprefix("/system").strip() or None
            self.success("System prompt updated." if self.system_prompt else "System prompt cleared.")
        elif cmd == "/chat":
            self.chat_command(args)
        elif cmd == "/update":
            update_message(self.theme)
        else:
            self.error(f"Unknown command: {cmd}. Type /help.")
        return False

    def help(self) -> None:
        rows = [
            ("/model", "[list|use|fetch|refresh|update|add|inspect|remove]"),
            ("/mode", "[base|chatml]"),
            ("/theme", "[list|veyra|warm|green|blue|mono]"),
            ("/device", "[list|cpu|cuda|directml|coreml|openvino|rocm|tensorrt]"),
            ("/autoload", "[on|off]"),
            ("/temp", "VALUE  /tokens N  /topk N  /topp VALUE  /repetition VALUE"),
            ("/system", "TEXT  /update"),
            ("/chat", "[new|list|load|rename|export|path]"),
            ("/status", " /clear  /help  /exit  /quit"),
        ]
        for command, rest in rows:
            print(self.theme.text("command", command) + (" " + rest if rest else ""))

    def status(self, show_chat: bool = True) -> None:
        if not models(self.config):
            state = "no model"
        else:
            state = "ready" if self.runner else "unloaded"
        self.banner(state)
        if show_chat:
            print(self.theme.text("label", "chat   ") + self.theme.text("value", str(self.chat.path if self.chat else "none")))
            print(self.theme.text("label", "device ") + self.theme.text("value", normalize_device(self.config.get("device"))))

    def theme_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "theme  ") + self.theme.text("value", normalize_theme(self.config.get("theme"))))
            print(self.theme.text("muted", "available: ") + " ".join(self.theme.text("command", name) for name in THEMES))
            return
        if args[0] == "list":
            print(" ".join(self.theme.text("command", name) for name in THEMES))
            return
        if args[0] not in THEMES:
            self.error(f"Unknown theme: {args[0]}")
            self.warn("Valid themes: " + ", ".join(THEMES))
            return
        selected = args[0]
        self.config["theme"] = selected
        save_config(self.config)
        self.theme = get_theme(selected)
        self.clear_visible_screen()
        self.status(show_chat=False)
        self.success(f"theme: {selected}")

    def device_command(self, args: list[str]) -> None:
        current = normalize_device(self.config.get("device"))
        if not args:
            print(self.theme.text("label", "device ") + self.theme.text("value", current))
            available = available_devices()
            print(self.theme.text("muted", "available: ") + " ".join(self.theme.text("command", name) for name in available))
            return
        if args[0] == "list":
            for name, provider, is_available in device_rows():
                mark = "*" if name == current else " "
                status = "available" if is_available else "unavailable"
                role = "success" if is_available else "muted"
                print(
                    f"{mark} "
                    + self.theme.text("command", name.ljust(8))
                    + " "
                    + self.theme.text(role, status.ljust(11))
                    + " "
                    + self.theme.text("muted", provider)
                )
            return
        selected = normalize_device(args[0])
        if selected != args[0].lower() and args[0].lower() not in {"gpu", "dml"}:
            self.error(f"Unknown device: {args[0]}")
            self.warn("Valid devices: " + ", ".join(name for name, _, _ in device_rows()))
            return
        available = available_devices()
        if selected not in available:
            self.error(f"Device '{selected}' is not available in this ONNX Runtime install.")
            self.warn("Available devices: " + ", ".join(available or ["none"]))
            return
        self.config["device"] = selected
        save_config(self.config)
        self.runner = None
        if models(self.config) and self.config.get("current_model"):
            self.load_current_model()
        self.success(f"device: {selected} ({provider_for_device(selected)})")

    def model_command(self, args: list[str]) -> None:
        if not args:
            self.status()
            self.list_models()
            return
        action = args[0]
        if action == "list":
            self.list_models()
        elif action == "use" and len(args) >= 2:
            self.use_model(args[1])
        elif action == "fetch":
            self.fetch_model()
        elif action == "refresh":
            self.remote_list()
        elif action == "update":
            self.update_models(all_models=len(args) >= 2 and args[1] == "all")
        elif action == "add" and len(args) >= 2:
            self.add_model(args[1], None)
        elif action == "inspect":
            self.inspect_current()
        elif action == "remove" and len(args) >= 2:
            remove_model(self.config, args[1])
            self.success(f"Removed {args[1]}.")
        else:
            self.warn("Usage: /model [list|use NAME|fetch|refresh|update [all]|add PATH|inspect|remove NAME]")

    def list_models(self) -> None:
        if not models(self.config):
            self.warn("No models installed.")
            return
        current = self.config.get("current_model")
        for name, entry in models(self.config).items():
            mark = "*" if name == current else " "
            print(f"{mark} {self.theme.text('value', name)} {self.theme.text('muted', '(' + entry.get('source', 'unknown') + ')')}")

    def use_model(self, name: str) -> None:
        if name not in models(self.config):
            self.error(f"Unknown model: {name}")
            return
        self.config["current_model"] = name
        entry = models(self.config)[name]
        if entry.get("mode"):
            self.config["current_mode"] = entry["mode"]
        save_config(self.config)
        self.runner = None
        if self.load_current_model():
            self.success(f"Using {name}.")

    def fetch_model(self) -> None:
        choices = self.remote_list()
        if not choices:
            return
        raw = input("Select model number: ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
            self.warn("Cancelled.")
            return
        repo_id = choices[int(raw) - 1]["repo_id"]
        try:
            path, commit = download_model(repo_id)
            entry = registry_entry(repo_id, path, commit=commit)
            register_model(self.config, safe_model_name(repo_id), entry)
            self.runner = None
            self.load_current_model()
            self.success(f"Fetched and selected {repo_id}.")
        except Exception as exc:
            self.error(f"Fetch failed: {exc}")

    def remote_list(self):
        try:
            choices = list_veyra_models()
        except Exception as exc:
            self.error(f"Could not query Hugging Face: {exc}")
            return []
        if not choices:
            self.warn("No compatible ONNX models found in veyra-ai.")
            return []
        for idx, item in enumerate(choices, 1):
            print(f"{self.theme.text('label', str(idx) + '.')} {self.theme.text('value', item['repo_id'])}")
        return choices

    def update_models(self, all_models: bool = False) -> None:
        names = list(models(self.config)) if all_models else [self.config.get("current_model")]
        for name in filter(None, names):
            entry = models(self.config).get(name)
            if not entry or entry.get("source") != "huggingface":
                self.warn(f"Skipping {name}: not a Hugging Face model.")
                continue
            try:
                path, commit = download_model(entry["repo_id"], entry.get("revision", "main"))
                entry.update({"path": str(path), "downloaded_commit": commit})
                self.success(f"Updated {name}.")
            except Exception as exc:
                self.error(f"Could not update {name}: {exc}")
        save_config(self.config)

    def add_model(self, path: str, name: str | None) -> None:
        try:
            info = inspect_model(path)
            if not info.supported:
                print(format_inspection(info))
                return
            model_name = name or safe_model_name(Path(path).expanduser().resolve().name)
            entry = {
                "source": "local",
                "repo_id": None,
                "revision": None,
                "downloaded_commit": None,
                "path": str(info.model_dir),
                "runtime": "onnx",
                "architecture": info.model_type or info.architecture or "unknown",
                "mode": self.config.get("current_mode", "chatml"),
                "quantized": "int8" in info.onnx_path.name.lower(),
            }
            register_model(self.config, model_name, entry)
            self.runner = None
            self.load_current_model()
            self.success(f"Added and selected {model_name}.")
        except Exception as exc:
            self.error(f"Could not add model: {exc}")

    def inspect_current(self) -> None:
        _, entry = current_model_entry(self.config)
        if not entry:
            self.warn("No current model.")
            return
        print(format_inspection(inspect_model(entry["path"])))

    def mode_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "mode   ") + self.theme.text("value", self.config.get("current_mode", "chatml")))
            return
        if args[0] not in {"base", "chatml"}:
            self.warn("Usage: /mode [base|chatml]")
            return
        self.config["current_mode"] = args[0]
        save_config(self.config)
        if self.chat:
            self.chat.append({"type": "event", "name": "mode_changed", "value": args[0]})
        self.success(f"mode: {args[0]}")

    def autoload_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "autoload ") + self.theme.text("value", str(self.config.get("autoload", True))))
            return
        self.config["autoload"] = args[0] == "on"
        save_config(self.config)
        self.success(f"autoload: {'on' if self.config['autoload'] else 'off'}")

    def default_command(self, cmd: str, args: list[str]) -> None:
        names = {"/temp": "temperature", "/tokens": "max_new_tokens", "/topk": "top_k", "/topp": "top_p", "/repetition": "repetition_penalty"}
        key = names[cmd]
        if not args:
            print(self.theme.text("label", key + ": ") + self.theme.text("value", str(self.config["defaults"][key])))
            return
        value = int(args[0]) if key in {"max_new_tokens", "top_k"} else float(args[0])
        self.config["defaults"][key] = value
        save_config(self.config)
        self.success(f"{key}: {value}")

    def chat_command(self, args: list[str]) -> None:
        action = args[0] if args else ""
        if action in {"", "path"}:
            print(self.theme.text("label", "chat   ") + self.theme.text("value", str(self.chat.path if self.chat else "No chat.")))
        elif action == "new":
            self.chat = ChatStore.new(self.config.get("current_model"), self.config.get("current_mode", "chatml"))
            self.success(f"New chat: {self.chat.path.stem}")
        elif action == "list":
            for path in ChatStore.list():
                print(self.theme.text("value", path.stem))
        elif action == "load" and len(args) >= 2:
            chat = ChatStore.named(args[1])
            if chat:
                self.chat = chat
                self.success(f"Loaded {chat.path.stem}.")
            else:
                self.error("Chat not found.")
        elif action == "rename" and len(args) >= 2 and self.chat:
            self.success(f"Renamed to {self.chat.rename(args[1]).stem}.")
        elif action == "export" and len(args) >= 2 and args[1] == "markdown" and self.chat:
            self.success(f"Exported {self.chat.export_markdown()}")
        else:
            self.warn("Usage: /chat [new|list|load NAME|rename NAME|export markdown|path]")

    def command_hint(self, prefix: str, commands: list[str], suffix: str) -> None:
        body = self.theme.text("muted", prefix + " ")
        body += self.theme.text("muted", ", ").join(self.theme.text("command", command) for command in commands)
        if suffix:
            body += self.theme.text("muted", " " + suffix)
        print(body)

    def success(self, message: str) -> None:
        print(self.theme.text("success", message))

    def warn(self, message: str) -> None:
        print(self.theme.text("warning", message))

    def error(self, message: str) -> None:
        print(self.theme.text("error", message))

    def clear_visible_screen(self, force: bool = False) -> None:
        if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
            print("\033[2J\033[H", end="")


def update_message(theme=None) -> None:
    def style(role: str, value: str) -> str:
        return theme.text(role, value) if theme else value

    print(style("muted", "To update Veyra, use one of:"))
    print("  " + style("command", "uv tool upgrade veyra"))
    print("  " + style("command", "pipx upgrade veyra"))
    print(style("muted", "For a source install, pull the latest source and reinstall with `uv tool install .`."))


def _prefer_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
