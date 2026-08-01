from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from . import __version__
from .chat_store import ChatStore
from .completion import VeyraCompleter
from .hf import (
    download_model,
    fetched_model_name,
    get_hf_model,
    list_veyra_models,
    recommended_onnx_file,
    registry_entry,
    variant_name,
)
from .inspect import format_inspection, inspect_model
from .prompts import PROMPT_MODES, format_prompt, infer_prompt_mode, normalize_mode
from .registry import CHATS_DIR, CONFIG_PATH, HISTORY_PATH, MODELS_DIR, current_model_entry, load_config, models, register_model, remove_model, safe_model_name, save_config
from .runner import OnnxCausalLMRunner, UnsupportedModelError, available_devices, device_install_hint, device_rows, model_context_length, normalize_device, provider_for_device
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
        self._last_tab_at = 0.0
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
        left_width = 29
        right_width = 42
        title = f" {self.assistant_name()} v{__version__} "
        border_role = "border"
        top = (
            self.theme.text(border_role, "\u256d\u2500")
            + self.theme.text("title", title)
            + self.theme.text(border_role, "\u2500" * max(0, left_width - len(title) - 1))
            + self.theme.text(border_role, "\u252c")
            + self.theme.text(border_role, "\u2500" * right_width)
            + self.theme.text(border_role, "\u256e")
        )
        print(top)
        for left, right in self.banner_rows(state):
            self.print_box_row(left, right, left_width, right_width)
        print(
            self.theme.text(border_role, "\u2570")
            + self.theme.text(border_role, "\u2500" * left_width)
            + self.theme.text(border_role, "\u2534")
            + self.theme.text(border_role, "\u2500" * right_width)
            + self.theme.text(border_role, "\u256f")
        )

    def print_box_row(
        self,
        left_segments: list[tuple[str, str]],
        right_segments: list[tuple[str, str]],
        left_width: int,
        right_width: int,
    ) -> None:
        border = self.theme.text("border", "\u2502")
        left = self.pad_segments(left_segments, left_width)
        right = self.pad_segments(right_segments, right_width)
        print(f"{border}{left}{border}{right}{border}")

    def pad_segments(self, segments: list[tuple[str, str]], width: int) -> str:
        used = 0
        clipped: list[tuple[str, str]] = []
        for role, text in segments:
            remaining = width - used
            if remaining <= 0:
                break
            piece = text[:remaining]
            clipped.append((role, piece))
            used += len(piece)
        plain = "".join(text for _, text in clipped)
        body = "".join(self.theme.text(role, text) for role, text in clipped)
        return body + " " * max(0, width - len(plain))

    def ready_message(self) -> None:
        if not models(self.config):
            self.warn("No model installed yet.")
            self.command_hint("Use", ["/model fetch"], "to find ONNX models from veyra-ai.")
            self.command_hint("Use", ["/model add PATH"], "to add a local ONNX model.")
            return
        self.command_hint("type", ["/help", "/model", "/mode", "/device", "/chat", "/exit"], "")

    def banner_segments(self, state: str) -> list[list[tuple[str, str]]]:
        return [left + right for left, right in self.banner_rows(state)]

    def banner_rows(self, state: str) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
        defaults = self.config.get("defaults", {})
        configured_context = defaults.get("context_length")
        effective_context = configured_context or self.current_model_limit()
        context_text = str(effective_context or "unknown")
        if configured_context is None and effective_context:
            context_text += "(auto)"
        autoload = "on" if self.config.get("autoload", True) else "off"
        status_role = {
            "ready": "status_ready",
            "loading": "status_loading",
            "failed": "status_error",
            "no model": "status_empty",
            "unloaded": "status_empty",
        }.get(state, "status_empty")
        model = self.config.get("current_model") or "none"
        mode = self.config.get("current_mode", "chatml")
        return [
            [[], [("label", "  Tips for getting started")]],
            [[("muted", "    "), (status_role, "\u25cf " + state)], [("label", "  Model: "), ("value", model)]],
            [[("muted", f"    autoload: {autoload}")], [("label", "  Mode:  "), ("value", mode)]],
            [[], []],
            [[("muted", "    Using local ONNX engine")], [("label", "  Generation Settings")]],
            [[("muted", "    Type /help for commands")], [("value", f"  output:{defaults.get('max_new_tokens', 128)} temp:{defaults.get('temperature', 0.8)} top-k:{defaults.get('top_k', 40)}")]],
            [[], [("value", f"  context:{context_text} repeat:{defaults.get('repetition_penalty', 1.0)} top-p:{defaults.get('top_p', 1.0)}")]],
        ]

    def chat_names(self) -> list[str]:
        return [path.stem for path in ChatStore.list()]

    def load_current_model(self, quiet: bool = False) -> bool:
        name, entry = current_model_entry(self.config)
        if not entry:
            return False
        try:
            self.runner = OnnxCausalLMRunner(entry["path"], device=self.config.get("device", "cpu"))
            self.apply_model_profile(entry)
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
        history = self.chat.history() if mode != "base" and self.chat else []
        self.generate_response(text, history, append_user=True)

    def generate_response(self, text: str, history: list[dict[str, str]], append_user: bool) -> None:
        assert self.runner is not None
        defaults = self.config.get("defaults", {})
        try:
            prompt, dropped = self.prompt_with_sliding_window(text, history, defaults)
        except ValueError as exc:
            self.error(str(exc))
            return
        if dropped:
            self.warn(f"Context window: dropped {dropped} oldest message{'s' if dropped != 1 else ''}.")
        if append_user and self.chat:
            self.chat.message("user", text)
        print(self.theme.text("assistant_prompt", f"{self.assistant_name()} \u203a "), end="", flush=True)
        chunks: list[str] = []
        start = time.perf_counter()
        first_token_at: float | None = None
        generated_tokens = 0
        try:
            for delta in self.runner.generate(prompt, **defaults):
                if self.double_tab_stop_requested():
                    self.warn("\n[generation stopped]")
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated_tokens += 1
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
            if self.config.get("stats", False):
                self.print_generation_stats(start, first_token_at, generated_tokens)
            if chunks and self.chat:
                self.chat.message("assistant", "".join(chunks))

    def prompt_with_sliding_window(
        self,
        text: str,
        history: list[dict[str, str]],
        defaults: dict,
    ) -> tuple[str, int]:
        assert self.runner is not None
        mode = self.config.get("current_mode", "chatml")
        remaining = list(history)
        limit = defaults.get("context_length") or self.runner.max_context_length
        output_budget = max(0, int(defaults.get("max_new_tokens", 128)))
        if self.runner.max_context_length and limit and int(limit) > self.runner.max_context_length:
            raise ValueError(f"Context length {limit} exceeds this model's limit of {self.runner.max_context_length}.")
        dropped = 0
        while True:
            prompt = format_prompt(text, mode, history=remaining, system_prompt=self.system_prompt)
            if not limit or self.runner.token_count(prompt) + output_budget <= int(limit):
                return prompt, dropped
            if not remaining:
                raise ValueError(
                    f"The current prompt plus {output_budget} output tokens does not fit in the {limit}-token context window."
                )
            remove_count = 2 if len(remaining) >= 2 and remaining[0].get("role") == "user" and remaining[1].get("role") == "assistant" else 1
            del remaining[:remove_count]
            dropped += remove_count

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
        elif cmd == "/doctor":
            self.doctor()
        elif cmd == "/retry":
            self.retry_command()
        elif cmd == "/clear":
            self.clear_visible_screen(force=True)
        elif cmd == "/model":
            self.model_command(args)
        elif cmd == "/mode":
            self.mode_command(args)
        elif cmd == "/theme":
            self.theme_command(args)
        elif cmd == "/profile":
            self.profile_command(args)
        elif cmd == "/device":
            self.device_command(args)
        elif cmd == "/stats":
            self.stats_command(args)
        elif cmd == "/autoload":
            self.autoload_command(args)
        elif cmd in {"/temp", "/tokens", "/topk", "/topp", "/repetition", "/seed", "/context"}:
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
            ("/model", "[list|use|fetch|refresh|update|test|add|inspect|info|remove]"),
            ("/mode", "[base|chatml|qwen|gemma|mistral|llama3]"),
            ("/theme", "[list|veyra|warm|red|pink|lime|green|blue|cyan|purple|orange|gray|rainbow|mono]"),
            ("/profile", "[show|name NAME|mode MODE]"),
            ("/device", "[list|cpu|directml|openvino]"),
            ("/stats", "[on|off]"),
            ("/autoload", "[on|off]"),
            ("/temp", "VALUE  /tokens N  /topk N  /topp VALUE  /repetition VALUE"),
            ("/seed", "[N|random]  /context [N|auto]  /retry"),
            ("/system", "TEXT  /update"),
            ("/chat", "[new|list|load|rename|export|path]"),
            ("/status", " /doctor  /clear  /help  /exit  /quit"),
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
            print(self.theme.text("label", "stats  ") + self.theme.text("value", "on" if self.config.get("stats", False) else "off"))

    def theme_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "theme  ") + self.theme.text("value", normalize_theme(self.config.get("theme"))))
            print(self.theme.text("muted", "available: ") + " ".join(self.theme.text("command", name) for name in THEMES))
            return
        if args[0] == "help":
            self.warn("Usage: /theme [list|" + "|".join(THEMES) + "]")
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

    def profile_command(self, args: list[str]) -> None:
        if not args or args[0] == "show":
            self.profile_show()
            return
        if args[0] == "name" and len(args) >= 2:
            name = " ".join(args[1:]).strip()
            self.config["assistant_name"] = name
            entry = self.current_entry()
            if entry is not None:
                entry.setdefault("profile", {})["assistant_name"] = name
            save_config(self.config)
            self.success(f"name: {name}")
            return
        if args[0] == "mode" and len(args) >= 2:
            mode = normalize_mode(args[1])
            self.config["current_mode"] = mode
            entry = self.current_entry()
            if entry is not None:
                entry["mode"] = mode
                entry.setdefault("profile", {})["mode"] = mode
            save_config(self.config)
            self.success(f"profile mode: {mode}")
            return
        self.warn("Usage: /profile [show|name NAME|mode MODE]")

    def profile_show(self) -> None:
        print(self.theme.text("label", "name   ") + self.theme.text("value", self.assistant_name()))
        print(self.theme.text("label", "model  ") + self.theme.text("value", self.config.get("current_model") or "none"))
        print(self.theme.text("label", "mode   ") + self.theme.text("value", self.config.get("current_mode", "chatml")))
        defaults = self.config.get("defaults", {})
        for label, key in (
            ("tokens", "max_new_tokens"),
            ("temp", "temperature"),
            ("top-k", "top_k"),
            ("top-p", "top_p"),
            ("repeat", "repetition_penalty"),
            ("seed", "seed"),
            ("context", "context_length"),
        ):
            value = defaults.get(key)
            if key == "seed" and value is None:
                shown = "random"
            elif key == "context_length" and value is None:
                shown = f"auto ({self.current_model_limit() or 'unknown'})"
            else:
                shown = str(value)
            print(self.theme.text("label", label.ljust(9)) + self.theme.text("value", shown))

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
        if args[0] == "help":
            target = normalize_device(args[1] if len(args) > 1 else current)
            print(self.theme.text("label", target + " ") + self.theme.text("value", provider_for_device(target)))
            self.warn(device_install_hint(target))
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
            self.warn(device_install_hint(selected))
            return
        self.config["device"] = selected
        save_config(self.config)
        self.runner = None
        if models(self.config) and self.config.get("current_model"):
            self.load_current_model()
        self.success(f"device: {selected} ({provider_for_device(selected)})")
        if selected == "directml":
            self.warn("DirectML can be slower than CPU for small token-by-token models; use /stats on to compare.")

    def stats_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "stats  ") + self.theme.text("value", "on" if self.config.get("stats", False) else "off"))
            return
        if args[0] not in {"on", "off"}:
            self.warn("Usage: /stats [on|off]")
            return
        self.config["stats"] = args[0] == "on"
        save_config(self.config)
        self.success(f"stats: {args[0]}")

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
            self.fetch_model(args[1] if len(args) >= 2 else None)
        elif action == "refresh":
            self.remote_list()
        elif action == "update":
            self.update_models(all_models=len(args) >= 2 and args[1] == "all")
        elif action == "test":
            self.model_test(args[1] if len(args) >= 2 else None)
        elif action == "add" and len(args) >= 2:
            self.add_model(args[1], None)
        elif action == "inspect":
            self.inspect_current()
        elif action == "info":
            self.model_info(args[1] if len(args) >= 2 else None)
        elif action == "remove" and len(args) >= 2:
            self.remove_model_command(args[1])
        else:
            self.warn("Usage: /model [list|use NAME|fetch [REPO_ID]|refresh|update [all]|test [NAME]|add PATH|inspect|info [NAME]|remove NAME]")

    def list_models(self) -> None:
        if not models(self.config):
            self.warn("No models installed.")
            return
        current = self.config.get("current_model")
        for name, entry in models(self.config).items():
            mark = "*" if name == current else " "
            print(f"{mark} {self.theme.text('value', name)} {self.theme.text('muted', '(' + entry.get('source', 'unknown') + ')')}")

    def remove_model_command(self, name: str) -> None:
        installed = models(self.config)
        if name not in installed:
            self.error(f"Unknown model: {name}")
            if installed:
                self.warn("Installed models: " + ", ".join(sorted(installed)))
            return
        was_current = self.config.get("current_model") == name
        remove_model(self.config, name)
        if was_current:
            self.runner = None
            self.load_error = None
        self.success(f"Removed {name}.")
        if was_current:
            self.warn("No current model selected. Use /model use NAME or /model add PATH.")

    def use_model(self, name: str) -> None:
        if name not in models(self.config):
            self.error(f"Unknown model: {name}")
            return
        self.config["current_model"] = name
        save_config(self.config)
        self.runner = None
        if self.load_current_model():
            self.success(f"Using {name}.")

    def fetch_model(self, repo_id: str | None = None) -> None:
        if repo_id:
            try:
                selected = get_hf_model(repo_id)
            except Exception as exc:
                self.error(f"Fetch failed: {exc}")
                return
        else:
            choices = self.remote_list()
            if not choices:
                return
            raw = input("Select model number: ").strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
                self.warn("Cancelled.")
                return
            selected = choices[int(raw) - 1]
        repo_id = selected["repo_id"]
        onnx_file = self.select_onnx_variant(selected)
        if not onnx_file:
            return
        try:
            path, commit = download_model(repo_id, onnx_file=onnx_file)
            entry = registry_entry(
                repo_id,
                path,
                commit=commit,
                onnx_file=onnx_file,
            )
            name = fetched_model_name(repo_id, onnx_file, len(selected["onnx_files"]))
            register_model(self.config, name, entry)
            self.runner = None
            self.load_current_model()
            self.success(f"Fetched and selected {name} ({variant_name(onnx_file)}).")
        except Exception as exc:
            self.error(f"Fetch failed: {exc}")

    def select_onnx_variant(self, model: dict) -> str | None:
        variants = model.get("onnx_files", [])
        if not variants:
            self.error(f"No ONNX variants found in {model['repo_id']}.")
            return None
        if len(variants) == 1:
            return variants[0]
        recommended = recommended_onnx_file(variants)
        print(self.theme.text("label", "Available ONNX variants:"))
        for idx, path in enumerate(variants, 1):
            suffix = " (recommended)" if path == recommended else ""
            print(
                f"{self.theme.text('label', str(idx) + '.')} "
                f"{self.theme.text('value', path)}"
                f"{self.theme.text('muted', suffix)}"
            )
        default = variants.index(recommended) + 1
        raw = input(f"Select variant number [{default}]: ").strip()
        if not raw:
            return recommended
        if not raw.isdigit() or not (1 <= int(raw) <= len(variants)):
            self.warn("Cancelled.")
            return None
        return variants[int(raw) - 1]

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
                onnx_file = entry.get("onnx_file")
                if not onnx_file and entry.get("path"):
                    info = inspect_model(entry["path"])
                    onnx_file = info.onnx_path.relative_to(info.model_dir).as_posix()
                path, commit = download_model(
                    entry["repo_id"],
                    entry.get("revision", "main"),
                    onnx_file=onnx_file,
                )
                refreshed = registry_entry(
                    entry["repo_id"],
                    path,
                    revision=entry.get("revision", "main"),
                    commit=commit,
                    onnx_file=onnx_file,
                )
                if entry.get("profile"):
                    refreshed["profile"] = entry["profile"]
                entry.update(refreshed)
                self.success(f"Updated {name}.")
            except Exception as exc:
                self.error(f"Could not update {name}: {exc}")
        save_config(self.config)

    def add_model(self, path: str, name: str | None) -> None:
        added = False
        for candidate in find_model_dirs(Path(path).expanduser()):
            try:
                info = inspect_model(candidate)
                if not info.supported:
                    continue
                model_name = name or safe_model_name(info.model_dir.name)
                if name and added:
                    model_name = safe_model_name(info.model_dir.name)
                entry = make_local_entry(info)
                register_model(self.config, model_name, entry)
                added = True
                self.success(f"Added {model_name}.")
            except Exception as exc:
                self.warn(f"Skipping {candidate}: {exc}")
        if added:
            self.runner = None
            self.load_current_model()
            return
        try:
            info = inspect_model(path)
            if not info.supported:
                print(format_inspection(info))
                return
            model_name = name or safe_model_name(Path(path).expanduser().resolve().name)
            entry = make_local_entry(info)
            register_model(self.config, model_name, entry)
            self.runner = None
            self.load_current_model()
            self.success(f"Added and selected {model_name}.")
        except Exception as exc:
            self.error(f"Could not add model: {exc}")

    def model_test(self, name: str | None = None) -> None:
        target = name or self.config.get("current_model")
        entry = models(self.config).get(target)
        if not entry:
            self.error(f"Unknown model: {target}")
            return
        start = time.perf_counter()
        try:
            runner = OnnxCausalLMRunner(entry["path"], device=self.config.get("device", "cpu"))
            load_s = time.perf_counter() - start
            prompt = format_prompt("Say hi", entry.get("mode", self.config.get("current_mode", "chatml")))
            gen_start = time.perf_counter()
            first = next(runner.generate(prompt, max_new_tokens=1, temperature=0), "")
            total = time.perf_counter() - start
            self.success(f"test ok: {target}")
            print(self.theme.text("muted", f"load {load_s:.2f}s | first token {time.perf_counter() - gen_start:.2f}s | total {total:.2f}s | token {first!r}"))
        except Exception as exc:
            self.error(f"test failed: {exc}")

    def inspect_current(self) -> None:
        _, entry = current_model_entry(self.config)
        if not entry:
            self.warn("No current model.")
            return
        print(format_inspection(inspect_model(entry["path"])))

    def model_info(self, name: str | None = None) -> None:
        target = name or self.config.get("current_model")
        entry = models(self.config).get(target)
        if not entry:
            self.error(f"Unknown model: {target}")
            return
        root = Path(entry["path"])
        limit = self.model_limit_for_entry(entry)
        try:
            info = inspect_model(root)
            onnx = info.onnx_path.relative_to(root)
            size = info.onnx_path.stat().st_size / (1024 * 1024)
            cache = f"yes ({len(info.cache_inputs) // 2} layers)" if info.cache_inputs else "no"
            supported = "yes" if info.supported else "no"
        except Exception as exc:
            self.error(f"Could not inspect {target}: {exc}")
            return
        rows = {
            "model": target,
            "source": entry.get("source", "unknown"),
            "repo": entry.get("repo_id") or "local",
            "path": str(root),
            "onnx": f"{onnx} ({size:.1f} MiB)",
            "architecture": entry.get("architecture") or info.model_type or info.architecture or "unknown",
            "context": str(limit or "unknown"),
            "kv cache": cache,
            "supported": supported,
            "mode": entry.get("profile", {}).get("mode", entry.get("mode", "chatml")),
            "generation": self.generation_summary(entry),
        }
        for label, value in rows.items():
            print(self.theme.text("label", label.ljust(13)) + self.theme.text("value", str(value)))

    def mode_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "mode   ") + self.theme.text("value", self.config.get("current_mode", "chatml")))
            return
        mode = normalize_mode(args[0])
        if args[0] != mode and args[0] not in PROMPT_MODES:
            self.warn("Usage: /mode [base|chatml|qwen|gemma|mistral|llama3]")
            return
        self.config["current_mode"] = mode
        entry = self.current_entry()
        if entry is not None:
            entry["mode"] = mode
            entry.setdefault("profile", {})["mode"] = mode
        save_config(self.config)
        if self.chat:
            self.chat.append({"type": "event", "name": "mode_changed", "value": mode})
        self.success(f"mode: {mode}")

    def autoload_command(self, args: list[str]) -> None:
        if not args:
            print(self.theme.text("label", "autoload ") + self.theme.text("value", str(self.config.get("autoload", True))))
            return
        self.config["autoload"] = args[0] == "on"
        save_config(self.config)
        self.success(f"autoload: {'on' if self.config['autoload'] else 'off'}")

    def default_command(self, cmd: str, args: list[str]) -> None:
        names = {
            "/temp": "temperature",
            "/tokens": "max_new_tokens",
            "/topk": "top_k",
            "/topp": "top_p",
            "/repetition": "repetition_penalty",
            "/seed": "seed",
            "/context": "context_length",
        }
        key = names[cmd]
        if not args:
            value = self.config["defaults"].get(key)
            if key == "context_length":
                limit = self.current_model_limit()
                shown = str(value) if value is not None else f"auto ({limit or 'unknown'})"
            elif key == "seed":
                shown = str(value) if value is not None else "random"
            else:
                shown = str(value)
            print(self.theme.text("label", key + ": ") + self.theme.text("value", shown))
            return
        raw = args[0].lower()
        if key in {"seed", "context_length"} and raw in {"auto", "random", "off"}:
            value = None
        else:
            try:
                value = int(args[0]) if key in {"max_new_tokens", "top_k", "seed", "context_length"} else float(args[0])
            except ValueError:
                self.error(f"Invalid value for {cmd}: {args[0]}")
                return
        if key in {"max_new_tokens", "context_length"} and value is not None and value <= 0:
            self.error(f"{key} must be greater than zero.")
            return
        if key == "seed" and value is not None and value < 0:
            self.error("seed must be zero or greater.")
            return
        if key == "context_length" and value is not None:
            model_limit = self.current_model_limit()
            if self.current_entry() is not None and model_limit is None:
                self.error("This model does not declare a context limit, so Veyra cannot validate a custom context length.")
                return
            if model_limit and value > model_limit:
                self.error(f"Context length {value} exceeds this model's limit of {model_limit}.")
                return
            if value <= int(self.config["defaults"].get("max_new_tokens", 128)):
                self.error("Context length must be larger than max_new_tokens.")
                return
        if key == "max_new_tokens" and value is not None:
            context = self.config["defaults"].get("context_length") or self.current_model_limit()
            if context and value >= context:
                self.error(f"max_new_tokens must be smaller than the {context}-token context window.")
                return
        self.config["defaults"][key] = value
        entry = self.current_entry()
        if entry is not None:
            profile = entry.setdefault("profile", {})
            generation = profile.setdefault("generation", dict(self.config["defaults"]))
            generation[key] = value
        save_config(self.config)
        if value is not None:
            shown = value
        elif key == "seed":
            shown = "random"
        else:
            shown = f"auto ({self.current_model_limit() or 'unknown'})"
        self.success(f"{key}: {shown}")

    def retry_command(self) -> None:
        if self.runner is None and not self.load_current_model():
            self.error("Missing model. Use /model fetch or /model add PATH.")
            return
        if not self.chat:
            self.warn("There is no chat to retry.")
            return
        history = self.chat.history()
        user_index = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
        if user_index is None:
            self.warn("There is no user message to retry.")
            return
        text = history[user_index]["content"]
        prior_history = history[:user_index]
        self.chat.retry()
        self.generate_response(text, prior_history, append_user=False)

    def current_model_limit(self) -> int | None:
        if self.runner is not None:
            return self.runner.max_context_length
        entry = self.current_entry()
        return self.model_limit_for_entry(entry)

    def model_limit_for_entry(self, entry: dict | None) -> int | None:
        if not entry or not entry.get("path"):
            return None
        root = Path(entry["path"])
        limit = model_context_length(info_config(root), info_config(root, "tokenizer_config.json"))
        if limit:
            return limit
        try:
            info = inspect_model(root)
            for tensor in info.inputs:
                if tensor.name in {"input_ids", "inputs_embeds"} and len(tensor.shape) >= 2:
                    value = tensor.shape[1]
                    if isinstance(value, int) and value > 1:
                        return value
        except Exception:
            pass
        return None

    def doctor(self) -> None:
        self.doctor_row(True, "Veyra", f"v{__version__} on Python {platform.python_version()}")
        try:
            ort_version = importlib.metadata.version("onnxruntime")
        except importlib.metadata.PackageNotFoundError:
            try:
                ort_version = importlib.metadata.version("onnxruntime-directml")
            except importlib.metadata.PackageNotFoundError:
                try:
                    ort_version = importlib.metadata.version("onnxruntime-openvino")
                except importlib.metadata.PackageNotFoundError:
                    ort_version = None
        self.doctor_row(bool(ort_version), "ONNX Runtime", ort_version or "not installed")
        for label, path in (("config", CONFIG_PATH.parent), ("models", MODELS_DIR), ("chats", CHATS_DIR), ("history", HISTORY_PATH.parent)):
            self.doctor_row(path.exists() and os.access(path, os.W_OK), label, str(path))
        device = normalize_device(self.config.get("device"))
        self.doctor_row(device in available_devices(), "device", f"{device} ({provider_for_device(device)})")
        name, entry = current_model_entry(self.config)
        if not entry:
            self.doctor_row(False, "model", "none selected")
            return
        try:
            info = inspect_model(entry["path"])
            self.doctor_row(info.tokenizer_path.exists(), "tokenizer", str(info.tokenizer_path))
            self.doctor_row(info.supported, "ONNX graph", "supported" if info.supported else "unsupported")
            limit = self.current_model_limit()
            self.doctor_row(bool(limit), "context", str(limit or "missing from model metadata"), warning=not bool(limit))
            output_budget = int(self.config.get("defaults", {}).get("max_new_tokens", 128))
            budget_ok = not limit or output_budget < limit
            self.doctor_row(
                budget_ok,
                "token budget",
                f"{output_budget} max new / {limit or 'unknown'} context",
            )
            if self.runner is None:
                OnnxCausalLMRunner(entry["path"], device=device)
            self.doctor_row(True, "model load", str(name))
        except Exception as exc:
            self.doctor_row(False, "model load", str(exc))

    def doctor_row(self, ok: bool, label: str, detail: str, warning: bool = False) -> None:
        symbol = "!" if warning else ("OK" if ok else "FAIL")
        role = "warning" if warning else ("success" if ok else "error")
        print(self.theme.text(role, symbol.ljust(5)) + self.theme.text("label", label.ljust(14)) + self.theme.text("value", detail))

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

    def print_generation_stats(self, start: float, first_token_at: float | None, generated_tokens: int) -> None:
        end = time.perf_counter()
        elapsed = max(0.0, end - start)
        ttft = 0.0 if first_token_at is None else max(0.0, first_token_at - start)
        speed = generated_tokens / elapsed if elapsed > 0 and generated_tokens else 0.0
        device = normalize_device(self.config.get("device"))
        text = f"stats: {generated_tokens} tokens | {speed:.2f} tok/s | first token {ttft:.2f}s | total {elapsed:.2f}s | device {device}"
        print(self.theme.text("muted", text))

    def assistant_name(self) -> str:
        return str(self.config.get("assistant_name") or "Veyra")

    def current_entry(self) -> dict | None:
        name = self.config.get("current_model")
        return models(self.config).get(name) if name else None

    def generation_summary(self, entry: dict | None = None) -> str:
        defaults = dict(self.config.get("defaults", {}))
        profile = (entry or {}).get("profile", {})
        if isinstance(profile, dict) and isinstance(profile.get("generation"), dict):
            defaults.update(profile["generation"])
        seed = defaults.get("seed")
        context = defaults.get("context_length")
        limit = self.model_limit_for_entry(entry) if entry else self.current_model_limit()
        context_text = str(context) if context is not None else f"auto ({limit or 'unknown'})"
        return (
            f"tokens={defaults.get('max_new_tokens')} temp={defaults.get('temperature')} "
            f"top-k={defaults.get('top_k')} top-p={defaults.get('top_p')} "
            f"repeat={defaults.get('repetition_penalty')} seed={seed if seed is not None else 'random'} "
            f"context={context_text}"
        )

    def apply_model_profile(self, entry: dict) -> None:
        profile = entry.get("profile") if isinstance(entry.get("profile"), dict) else {}
        if profile.get("assistant_name"):
            self.config["assistant_name"] = profile["assistant_name"]
        if profile.get("mode"):
            self.config["current_mode"] = normalize_mode(profile["mode"])
        generation = profile.get("generation")
        if isinstance(generation, dict):
            for key in self.config["defaults"]:
                if key in generation:
                    self.config["defaults"][key] = generation[key]
        else:
            profile["generation"] = dict(self.config.get("defaults", {}))
            entry["profile"] = profile
        limit = self.runner.max_context_length if self.runner else None
        context = self.config["defaults"].get("context_length")
        if limit and context and context > limit:
            self.config["defaults"]["context_length"] = None
            profile["generation"]["context_length"] = None
        output_budget = int(self.config["defaults"].get("max_new_tokens", 128))
        if limit and output_budget >= limit:
            safe_budget = min(128, max(1, limit - 1))
            self.config["defaults"]["max_new_tokens"] = safe_budget
            profile["generation"]["max_new_tokens"] = safe_budget
        save_config(self.config)

    def double_tab_stop_requested(self) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\t":
                    now = time.perf_counter()
                    if now - self._last_tab_at <= 0.6:
                        self._last_tab_at = 0.0
                        return True
                    self._last_tab_at = now
        except Exception:
            return False
        return False

    def clear_visible_screen(self, force: bool = False) -> None:
        if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
            print("\033[2J\033[H", end="")


def update_message(theme=None) -> None:
    def style(role: str, value: str) -> str:
        return theme.text(role, value) if theme else value

    print(style("muted", "Install the latest Veyra CLI from GitHub with one of:"))
    print("  " + style("command", "uv tool install git+https://github.com/Jdudeo5972/veyra-cli.git"))
    print("  " + style("command", "pipx install git+https://github.com/Jdudeo5972/veyra-cli.git"))


def _prefer_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def info_config(model_dir: Path, name: str = "config.json") -> dict:
    try:
        with (model_dir / name).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def make_local_entry(info) -> dict:
    mode = infer_prompt_mode(info_config(info.model_dir), info_config(info.model_dir, "tokenizer_config.json"))
    return {
        "source": "local",
        "repo_id": None,
        "revision": None,
        "downloaded_commit": None,
        "path": str(info.model_dir),
        "runtime": "onnx",
        "architecture": info.model_type or info.architecture or "unknown",
        "mode": mode,
        "profile": {"mode": mode, "assistant_name": "Veyra"},
        "quantized": "int8" in info.onnx_path.name.lower() or "q4" in info.onnx_path.name.lower(),
    }


def find_model_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for path in [root, *[p for p in root.iterdir() if p.is_dir()]]:
        if (path / "tokenizer.json").exists() and (list(path.glob("*.onnx")) or list(path.rglob("*.onnx"))):
            candidates.append(path)
    return candidates
