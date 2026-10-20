from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.theme import Theme as TextualTheme
from textual.widgets import Button, Footer, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from . import __version__
from .completion import VeyraCompleter
from .prompts import PROMPT_MODES
from .registry import HISTORY_PATH, models
from .runner import device_rows, model_context_length
from .shell import VeyraShell, info_config
from .theme import THEMES, get_theme


class Picker(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, items: list[tuple[str, str]]) -> None:
        super().__init__()
        self.heading = title
        self.items = items
        self.filtered = items

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.heading, classes="dialog-title")
            yield Input(placeholder="Search", id="picker-search")
            yield OptionList(id="picker-options")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.filter_items("")
        self.query_one(Input).focus()

    def filter_items(self, text: str) -> None:
        self.filtered = [item for item in self.items if text.casefold() in item[0].casefold()]
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options(Option(Text(label), id=str(index)) for index, (label, _) in enumerate(self.filtered))
        if self.filtered:
            options.highlighted = 0

    @on(Input.Changed, "#picker-search")
    def search_changed(self, event: Input.Changed) -> None:
        self.filter_items(event.value)

    @on(Input.Submitted, "#picker-search")
    def select_first(self) -> None:
        index = self.query_one(OptionList).highlighted
        if index is not None and self.filtered:
            self.dismiss(self.filtered[index][1])

    @on(OptionList.OptionSelected)
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.filtered[event.option_index][1])

    def on_key(self, event) -> None:
        if isinstance(self.focused, Input) and event.key in {"up", "down"}:
            options = self.query_one(OptionList)
            if event.key == "up":
                options.action_cursor_up()
            else:
                options.action_cursor_down()
            event.stop()
            event.prevent_default()

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class Question(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "", secret: bool = False) -> None:
        super().__init__()
        self.heading, self.initial_value, self.secret = title, value, secret

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog question"):
            yield Label(self.heading, classes="dialog-title")
            yield Input(value=self.initial_value, password=self.secret, id="answer")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    @on(Button.Pressed, "#save")
    def save(self) -> None:
        self.dismiss(self.query_one(Input).value)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatMessage(Vertical):
    def __init__(self, role: str, content: str = "", streaming: bool = False) -> None:
        super().__init__(classes="chat-message")
        self.role, self.content, self.streaming = role, content, streaming

    def compose(self) -> ComposeResult:
        yield Static(Text(self.role + " \u203a"), classes="message-role")
        if self.streaming or self.role == "You":
            yield Static(Text(self.content), classes="message-body")
        else:
            yield Markdown(self.content, open_links=False, classes="message-body")

    def append_text(self, text: str) -> None:
        self.content += text
        self.query_one(".message-body", Static).update(Text(self.content))

    async def finish(self) -> None:
        self.streaming = False
        await self.query_one(".message-body").remove()
        await self.mount(Markdown(self.content, open_links=False, classes="message-body"))


class Composer(Input):
    BINDINGS = [
        Binding("tab", "complete", "Complete", priority=True),
        Binding("up", "previous", "Previous", show=False, priority=True),
        Binding("down", "next", "Next", show=False, priority=True),
        Binding("escape", "stop", "Stop", show=False, priority=True),
    ]

    def action_complete(self) -> None:
        self.app.complete_command()

    def action_previous(self) -> None:
        self.app.navigate_input(-1)

    def action_next(self) -> None:
        self.app.navigate_input(1)

    def action_stop(self) -> None:
        self.app.action_stop()


class HistorySuggester(Suggester):
    def __init__(self, history: list[str]) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self.history = history

    async def get_suggestion(self, value: str) -> str | None:
        if value and not value.startswith("/"):
            return next((line for line in reversed(self.history) if line.startswith(value) and line != value), None)
        return None


class TuiShell(VeyraShell):
    """Reuse shell commands while routing output and questions to the UI thread."""

    def __init__(self, args: argparse.Namespace, events: queue.SimpleQueue, stop: threading.Event) -> None:
        self.events, self.stop_requested = events, stop
        super().__init__(args, interactive=False)

    def write_output(self, *values, sep=" ", end="\n", **kwargs) -> None:
        self.events.put(("output", sep.join(str(value) for value in values) + end))

    def ask(self, prompt: str) -> str:
        return self._question(prompt, secret=False)

    def ask_secret(self, prompt: str) -> str:
        return self._question(prompt, secret=True)

    def _question(self, prompt: str, secret: bool) -> str:
        answer: queue.Queue = queue.Queue(maxsize=1)
        self.events.put(("question", (prompt, secret, answer)))
        while not self.stop_requested.is_set():
            try:
                value = answer.get(timeout=0.1)
            except queue.Empty:
                continue
            if value is None:
                raise KeyboardInterrupt
            return value
        raise KeyboardInterrupt

    def banner(self, state: str) -> None:
        self.events.put(("state", state))

    def clear_visible_screen(self, force: bool = False) -> None:
        if force:
            self.events.put(("clear", None))

    def load_current_model(self, quiet: bool = False) -> bool:
        if not self.current_entry():
            return False
        self.banner("loading")
        loaded = super().load_current_model(quiet=quiet)
        self.banner("ready" if loaded else "failed")
        return loaded

    def generate_response(self, text: str, history: list[dict[str, str]], append_user: bool) -> None:
        if self.stop_requested.is_set():
            return
        defaults = dict(self.config.get("defaults", {}))
        prompt, dropped = self.prompt_with_sliding_window(text, history, defaults)
        if dropped:
            self.warn(f"Context window: dropped {dropped} oldest messages.")
        if append_user and self.chat:
            self.chat.message("user", text)
        if append_user:
            self.events.put(("user", text))
        else:
            self.events.put(("conversation", self.chat.history() if self.chat else []))
        self.events.put(("assistant", self.assistant_name()))
        chunks: list[str] = []
        start = time.perf_counter()
        first: float | None = None
        iterator = self.runner.generate(prompt, **defaults)
        last_metrics = start
        self.banner("generating")
        try:
            while not self.stop_requested.is_set():
                try:
                    delta = next(iterator)
                except StopIteration:
                    break
                if first is None:
                    first = time.perf_counter()
                chunks.append(delta)
                self.events.put(("delta", delta))
                now = time.perf_counter()
                if now - last_metrics >= 0.25:
                    self.events.put(("metrics", f"streaming | first text {first - start:.2f}s | total {now - start:.1f}s"))
                    last_metrics = now
        finally:
            iterator.close()
            if chunks and self.chat:
                self.chat.message("assistant", "".join(chunks))
            self.events.put(("response_done", None))
            if self.config.get("stats", False):
                self.print_generation_stats(start, first, self.runner.token_count("".join(chunks)))
            self.events.put(("metrics", f"total {time.perf_counter() - start:.2f}s | device {self.config.get('device', 'cpu')}"))
            if self.stop_requested.is_set():
                self.warn("Generation stopped.")
            self.banner("ready")


class VeyraApp(App):
    TITLE = "Veyra"
    CSS_PATH = "tui.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+q", "request_quit", "Quit", priority=True),
        Binding("ctrl+d", "request_quit", "Quit", show=False, priority=True),
        Binding("ctrl+c", "stop", "Stop", priority=True),
        Binding("f2", "models", "Models"),
        Binding("f3", "chats", "Chats"),
        Binding("f4", "settings", "Settings"),
        Binding("ctrl+n", "new_chat", "New chat"),
    ]

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        super().__init__()
        self.args = args or argparse.Namespace(no_load=False, continue_chat=False)
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.stop_requested = threading.Event()
        self.backend = TuiShell(self.args, self.events, self.stop_requested)
        self.busy = False
        self.exit_when_done = False
        self.state = "unloaded" if self.backend.current_entry() else "no model"
        self.response: ChatMessage | None = None
        self.completions: list = []
        self.history = list(reversed(list(FileHistory(str(HISTORY_PATH)).load_history_strings())))
        self.history_index = len(self.history)
        self.draft = ""
        self.last_tab_at = 0.0
        self.completer = VeyraCompleter(lambda: sorted(models(self.backend.config)), self.backend.chat_names)

    def compose(self) -> ComposeResult:
        yield Static(id="heading", markup=False)
        yield Static(id="status", markup=False)
        with Horizontal(id="toolbar"):
            yield Button("Models", id="models")
            yield Button("Chats", id="chats")
            yield Button("Settings", id="settings")
            yield Button("New chat", id="new-chat")
            yield Button("Stop", id="stop", disabled=True)
        yield VerticalScroll(id="conversation")
        yield Static("", id="metrics", markup=False)
        yield OptionList(id="completions")
        with Horizontal(id="composer-row"):
            yield Label("You \u203a", id="user-label")
            yield Composer(id="composer", select_on_focus=False, suggester=HistorySuggester(self.history))
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(0.05, self.drain_events)
        self.apply_theme()
        await self.show_conversation(self.backend.chat.history())
        self.refresh_status()
        self.query_one(Composer).focus()
        if not models(self.backend.config):
            await self.add_output("No model installed yet. Use /model fetch or /model add PATH.")
        if not self.args.no_load and self.backend.config.get("autoload", True) and self.backend.current_entry():
            self.start_operation(None)

    def apply_theme(self) -> None:
        selected = get_theme(self.backend.config.get("theme"))
        accents = {
            "veyra": "#e03971", "warm": "#CE562F", "red": "#ef5350",
            "pink": "#e03971", "lime": "#b5dc60", "green": "#63b776",
            "blue": "#69b9ee", "cyan": "#65d5d1", "purple": "#ba8ee0",
            "orange": "#ed9d56", "gray": "#b4b4b4", "rainbow": "#e88dd0",
        }
        accent = accents.get(selected.name, "#dedede") if selected.enabled else "#dedede"
        theme_name = "veyra-ui-" + (selected.name if selected.enabled else "mono")
        self.register_theme(TextualTheme(
            name=theme_name, primary=accent,
            accent="#efc678" if selected.enabled and selected.name == "rainbow" else accent,
            secondary="#6adbd1" if selected.enabled and selected.name == "rainbow" else accent,
            foreground="#eeeeee", background="#101012", surface="#19191d", panel="#202025",
            success=accent, warning="#ddbc76" if selected.enabled else "#eeeeee",
            error="#ef7777" if selected.enabled else "#eeeeee",
        ))
        self.theme = theme_name

    def refresh_status(self) -> None:
        config = self.backend.config
        defaults = config.get("defaults", {})
        self.query_one("#heading", Static).update(Text(f"Veyra v{__version__}   /   {self.state}"))
        entry = self.backend.current_entry()
        if self.backend.runner:
            limit = self.backend.runner.max_context_length
        elif entry:
            root = Path(entry["path"])
            limit = model_context_length(info_config(root), info_config(root, "tokenizer_config.json"))
        else:
            limit = None
        context = defaults.get("context_length") or limit or "unknown"
        auto = " (auto)" if defaults.get("context_length") is None else ""
        rows = [
            f"Model: {config.get('current_model') or 'none'}   Mode: {config.get('current_mode', 'chatml')}",
            f"{self.backend.current_runtime_label()} / {config.get('device', 'cpu')}   "
            f"output:{defaults.get('max_new_tokens')}  temp:{defaults.get('temperature')}  "
            f"top-k:{defaults.get('top_k')}  top-p:{defaults.get('top_p')}  repeat:{defaults.get('repetition_penalty')}",
            f"context:{context}{auto}   autoload:{'on' if config.get('autoload', True) else 'off'}   "
            f"chat:{self.backend.chat.path.stem if self.backend.chat else 'none'}",
        ]
        self.query_one("#status", Static).update(Text("\n".join(rows)))

    async def add_output(self, value: str) -> None:
        view = self.query_one("#conversation", VerticalScroll)
        at_end = view.is_vertical_scroll_end
        await view.mount(Static(Text.from_ansi(value.rstrip()), classes="command-output"))
        if at_end:
            view.scroll_end(animate=False)

    async def show_conversation(self, history: list[dict[str, str]]) -> None:
        view = self.query_one("#conversation", VerticalScroll)
        await view.remove_children()
        for message in history:
            role = "You" if message["role"] == "user" else self.backend.assistant_name()
            await view.mount(ChatMessage(role, message["content"]))
        view.scroll_end(animate=False)

    async def drain_events(self) -> None:
        pending = ""
        for _ in range(400):
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "delta":
                pending += data
                continue
            if pending and self.response:
                self.append_response(pending)
                pending = ""
            if kind == "output":
                await self.add_output(data)
            elif kind == "conversation":
                await self.show_conversation(data)
            elif kind == "user":
                await self.query_one("#conversation").mount(ChatMessage("You", data))
            elif kind == "assistant":
                self.response = ChatMessage(data, streaming=True)
                await self.query_one("#conversation").mount(self.response)
            elif kind == "response_done" and self.response:
                await self.response.finish()
                self.response = None
            elif kind == "metrics":
                self.query_one("#metrics", Static).update(Text(data))
            elif kind == "state":
                self.state = data
                self.refresh_status()
            elif kind == "clear":
                await self.query_one("#conversation").remove_children()
            elif kind == "question":
                prompt, secret, answer = data
                if not self.stop_requested.is_set():
                    self.push_screen(Question(prompt, secret=secret), lambda result, answer=answer: answer.put(result))
            elif kind == "finished":
                self.set_busy(False)
                self.apply_theme()
                self.state = "ready" if self.backend.runner else ("failed" if self.backend.load_error else "unloaded")
                if not self.backend.current_entry():
                    self.state = "no model"
                self.refresh_status()
                if self.exit_when_done:
                    self.exit()
            elif kind == "quit":
                self.exit_when_done = True
        if pending and self.response:
            self.append_response(pending)

    def append_response(self, text: str) -> None:
        view = self.query_one("#conversation", VerticalScroll)
        at_end = view.is_vertical_scroll_end
        self.response.append_text(text)
        if at_end:
            view.scroll_end(animate=False)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        if busy:
            self.query_one("#completions").display = False
        for button in self.query("#toolbar Button"):
            button.disabled = not busy if button.id == "stop" else busy

    def start_operation(self, text: str | None) -> None:
        if self.busy:
            self.notify("An operation is still running.", severity="warning")
            return
        self.stop_requested.clear()
        self.set_busy(True)
        self.query_one("#metrics", Static).update("")
        self.execute(text)

    @work(thread=True, exit_on_error=False)
    def execute(self, text: str | None) -> None:
        try:
            if text is None:
                self.backend.load_current_model()
            elif text.startswith("/"):
                if text.startswith("/chat load ") or text == "/chat new":
                    self.backend.handle_command(text)
                    self.events.put(("conversation", self.backend.chat.history()))
                elif self.backend.handle_command(text):
                    self.events.put(("quit", None))
            else:
                self.backend.handle_prompt(text)
        except KeyboardInterrupt:
            self.backend.warn("Cancelled.")
        except Exception as exc:
            self.backend.error(f"Operation failed: {exc}")
        finally:
            self.events.put(("finished", None))

    @on(Input.Changed, "#composer")
    def composer_changed(self) -> None:
        field = self.query_one(Composer)
        document = Document(field.value, cursor_position=field.cursor_position)
        self.completions = list(self.completer.get_completions(document, None))[:30]
        options = self.query_one("#completions", OptionList)
        options.clear_options()
        options.display = bool(self.completions) and not self.busy
        for item in self.completions:
            label = Text(item.text)
            label.append("  " + item.display_meta_text, style="dim")
            options.add_option(Option(label))
        if self.completions:
            options.highlighted = 0

    @on(Input.Submitted, "#composer")
    def submit(self, event: Input.Submitted) -> None:
        if self.query_one("#completions").display:
            self.complete_command()
            return
        text = event.value.strip()
        if not text:
            return
        if text in {"/exit", "/quit"}:
            self.action_request_quit()
            return
        if self.busy:
            self.notify("Stop the current operation before sending another message.", severity="warning")
            return
        field = self.query_one(Composer)
        field.value = ""
        self.history.append(text)
        self.history_index = len(self.history)
        FileHistory(str(HISTORY_PATH)).append_string(text)
        if text == "/model":
            self.action_models()
        elif text == "/chat":
            self.action_chats()
        else:
            self.start_operation(text)

    @on(OptionList.OptionSelected, "#completions")
    def completion_selected(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#completions", OptionList).highlighted = event.option_index
        self.complete_command()

    def complete_command(self) -> None:
        if self.busy:
            now = time.perf_counter()
            if now - self.last_tab_at < 0.6:
                self.action_stop()
            self.last_tab_at = now
            return
        options = self.query_one("#completions", OptionList)
        field = self.query_one(Composer)
        if not options.display or not self.completions:
            if field.value and not field.value.startswith("/") and field.cursor_position == len(field.value):
                suggestion = next((line for line in reversed(self.history) if line.startswith(field.value)), None)
                if suggestion:
                    field.value = suggestion
                    field.cursor_position = len(suggestion)
            return
        item = self.completions[options.highlighted or 0]
        cursor = field.cursor_position
        field.value = field.value[:cursor + item.start_position] + item.text + " " + field.value[cursor:]
        field.cursor_position = cursor + item.start_position + len(item.text) + 1
        field.focus()

    def navigate_input(self, direction: int) -> None:
        options = self.query_one("#completions", OptionList)
        if options.display:
            if direction < 0:
                options.action_cursor_up()
            else:
                options.action_cursor_down()
            return
        field = self.query_one(Composer)
        if self.history_index == len(self.history):
            self.draft = field.value
        self.history_index = min(len(self.history), max(0, self.history_index + direction))
        field.value = self.history[self.history_index] if self.history_index < len(self.history) else self.draft
        field.cursor_position = len(field.value)

    def action_stop(self) -> None:
        if self.busy:
            self.stop_requested.set()
            if isinstance(self.screen, Question):
                self.screen.dismiss(None)
            self.query_one("#metrics", Static).update("Stopping after the current model call...")
        else:
            self.query_one("#completions").display = False

    def action_request_quit(self) -> None:
        if self.busy:
            self.exit_when_done = True
            self.action_stop()
        else:
            self.exit()

    def on_unmount(self) -> None:
        self.stop_requested.set()

    def action_models(self) -> None:
        if self.busy:
            return
        installed = models(self.backend.config)
        items = [(f"{name}  /  {entry.get('runtime', 'onnx')}", f"/model use {name}") for name, entry in installed.items()]
        items.append(("Fetch from Hugging Face", "/model fetch"))
        self.push_screen(Picker("Models", items), self.run_selection)

    def action_chats(self) -> None:
        if self.busy:
            return
        items = [(path.stem, f"/chat load {path.stem}") for path in self.backend.chat.list()]
        items.insert(0, ("New chat", "/chat new"))
        self.push_screen(Picker("Chats", items), self.run_selection)

    def action_new_chat(self) -> None:
        if not self.busy:
            self.start_operation("/chat new")

    def run_selection(self, command: str | None) -> None:
        if command:
            self.start_operation(command)

    def action_settings(self) -> None:
        if self.busy:
            return
        defaults = self.backend.config["defaults"]
        items = [(f"{label}: {defaults.get(key) if defaults.get(key) is not None else 'auto'}", command) for label, key, command in (
            ("Output tokens", "max_new_tokens", "/tokens"), ("Temperature", "temperature", "/temp"),
            ("Top-k", "top_k", "/topk"), ("Top-p", "top_p", "/topp"),
            ("Repetition penalty", "repetition_penalty", "/repetition"),
            ("Context", "context_length", "/context"), ("Seed", "seed", "/seed"),
        )]
        items += [("Mode", "/mode"), ("Device", "/device"), ("Theme", "/theme"),
                  ("Assistant name", "/profile name"), ("System prompt", "/system"),
                  ("Response stats", "/stats"), ("Autoload", "/autoload")]
        self.push_screen(Picker("Generation & settings", items), self.edit_setting)

    def edit_setting(self, command: str | None) -> None:
        if not command:
            return
        choices: list[str] = []
        if command == "/mode":
            choices = list(PROMPT_MODES)
        elif command == "/theme":
            choices = list(THEMES)
        elif command == "/device":
            choices = [name for name, _, available in device_rows() if available]
        elif command in {"/stats", "/autoload"}:
            choices = ["on", "off"]
        if choices:
            self.push_screen(Picker(command[1:], [(value, f"{command} {value}") for value in choices]), self.run_selection)
        else:
            self.push_screen(Question(command[1:]), lambda value: self.run_selection(f"{command} {value}") if value is not None else None)

    @on(Button.Pressed)
    def toolbar_pressed(self, event: Button.Pressed) -> None:
        actions = {"models": self.action_models, "chats": self.action_chats, "settings": self.action_settings,
                   "new-chat": self.action_new_chat, "stop": self.action_stop}
        action = actions.get(event.button.id)
        if action:
            action()
