from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completion
from prompt_toolkit.history import FileHistory
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme as TextualTheme
from textual.widgets import Button, Input, Label, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from . import __version__
from .completion import TOP, VeyraCompleter
from .clipboard import read_clipboard, write_clipboard
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
    def __init__(self, role: str, content: str = "", streaming: bool = False, user: bool = False) -> None:
        super().__init__(classes="chat-message user-message" if user else "chat-message assistant-message")
        self.role, self.content, self.streaming = role, content, streaming
        self.user = user

    def compose(self) -> ComposeResult:
        yield Static(Text(self.role + " \u203a"), classes="message-role")
        if self.streaming or self.user:
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


class Composer(TextArea):
    class Submitted(Message):
        def __init__(self, composer: Composer) -> None:
            super().__init__()
            self.composer = composer
            self.value = composer.value

        @property
        def control(self) -> Composer:
            return self.composer

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter,ctrl+j", "newline", "New line", show=False, priority=True),
        Binding("tab", "complete", "Complete", priority=True),
        Binding("up", "previous", "Previous", show=False, priority=True),
        Binding("down", "next", "Next", show=False, priority=True),
        Binding("escape", "stop", "Stop", show=False, priority=True),
    ]

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.load_text(value)

    @property
    def cursor_position(self) -> int:
        return self.document.get_index_from_location(self.cursor_location)

    @cursor_position.setter
    def cursor_position(self, value: int) -> None:
        self.move_cursor(self.document.get_location_from_index(value))

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        self.app.complete_command()

    def action_previous(self) -> None:
        self.app.navigate_input(-1)

    def action_next(self) -> None:
        self.app.navigate_input(1)

    def action_stop(self) -> None:
        self.app.action_stop()


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

    def status(self, show_chat: bool = True) -> None:
        self.write_output(f"model: {self.config.get('current_model') or 'none'} | mode: {self.config.get('current_mode')}")
        self.write_output(f"state: {'ready' if self.runner else 'unloaded'} | device: {self.config.get('device', 'cpu')}")
        self.write_output(self.generation_summary())
        if show_chat and self.chat:
            self.write_output(f"chat: {self.chat.path}")

    def help(self) -> None:
        super().help()
        self.write_output(self.theme.text("command", "/copy") + " Copy the latest response")
        self.write_output(self.theme.text("command", "/commands") + " Open the command menu")

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
        Binding("ctrl+c", "copy_or_stop", "Copy / stop", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("ctrl+v,ctrl+shift+v,shift+insert", "paste", "Paste", show=False, priority=True),
        Binding("ctrl+p", "commands", "Commands", priority=True),
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
        with Vertical(id="workspace", classes="home"):
            yield Static(self.wordmark(), id="wordmark")
            yield VerticalScroll(id="conversation")
            with Horizontal(id="prompt-wrapper"):
                with Vertical(id="prompt-area"):
                    yield OptionList(id="completions")
                    with Vertical(id="composer-panel"):
                        yield Composer(
                            id="composer", show_line_numbers=False, highlight_cursor_line=False,
                            placeholder="Ask anything...", soft_wrap=True,
                        )
                        yield Static(id="model-line", markup=False)
                    with Horizontal(id="input-hints"):
                        yield Static("", id="metrics", markup=False)
                        yield Static("ctrl+p commands", id="shortcuts", markup=False)
        with Horizontal(id="session-bar"):
            yield Static(id="session-info", markup=False)
            yield Static(f"v{__version__}", id="version-info", markup=False)

    @staticmethod
    def wordmark() -> Text:
        letters = [
            ["#   #", "#   #", "#   #", " # # ", "  #  "],
            ["#####", "#    ", "#### ", "#    ", "#####"],
            ["#   #", " # # ", "  #  ", "  #  ", "  #  "],
            ["#### ", "#   #", "#### ", "#  # ", "#   #"],
            [" ### ", "#   #", "#####", "#   #", "#   #"],
        ]
        text = Text()
        for row in range(5):
            if row:
                text.append("\n")
            for index, letter in enumerate(letters):
                text.append(letter[row].replace("#", "\u2588") + ("  " if index < 4 else ""), style="#777777" if index < 2 else "#eeeeee")
        return text

    async def on_mount(self) -> None:
        self.set_interval(0.05, self.drain_events)
        self.apply_theme()
        await self.show_conversation(self.backend.chat.history())
        self.refresh_status()
        self.query_one(Composer).focus()
        if not models(self.backend.config):
            self.notify("No model installed yet. Use /model fetch or /model add PATH.", timeout=7)
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
            foreground="#eeeeee", background="#0b0b0b", surface="#1d1d1d", panel="#151515",
            success=accent, warning="#ddbc76" if selected.enabled else "#eeeeee",
            error="#ef7777" if selected.enabled else "#eeeeee",
        ))
        self.theme = theme_name

    def refresh_status(self) -> None:
        config = self.backend.config
        defaults = config.get("defaults", {})
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
        line = Text()
        line.append(config.get("current_mode", "chatml"), style=self.current_theme.primary)
        line.append("  \u00b7  ", style="dim")
        line.append(config.get("current_model") or "No model selected")
        line.append(f"  {self.backend.current_runtime_label()}", style="dim")
        self.query_one("#model-line", Static).update(line)
        self.query_one("#model-line").tooltip = (
            f"output:{defaults.get('max_new_tokens')}  temp:{defaults.get('temperature')}  "
            f"top-k:{defaults.get('top_k')}  top-p:{defaults.get('top_p')}  "
            f"repeat:{defaults.get('repetition_penalty')}  context:{context}{auto}"
        )
        session = self.backend.chat.path.stem if self.backend.chat else "new chat"
        self.query_one("#session-info", Static).update(Text(f"{self.state}  \u00b7  {config.get('device', 'cpu')}  \u00b7  {session}"))

    def set_conversation_mode(self, active: bool) -> None:
        workspace = self.query_one("#workspace")
        workspace.set_class(not active, "home")
        workspace.set_class(active, "conversation-mode")

    async def add_output(self, value: str) -> None:
        self.set_conversation_mode(True)
        view = self.query_one("#conversation", VerticalScroll)
        at_end = view.is_vertical_scroll_end
        await view.mount(Static(Text.from_ansi(value.rstrip()), classes="command-output"))
        if at_end:
            view.scroll_end(animate=False)

    async def show_conversation(self, history: list[dict[str, str]]) -> None:
        self.set_conversation_mode(bool(history))
        view = self.query_one("#conversation", VerticalScroll)
        await view.remove_children()
        for message in history:
            role = "You" if message["role"] == "user" else self.backend.assistant_name()
            await view.mount(ChatMessage(role, message["content"], user=message["role"] == "user"))
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
                self.set_conversation_mode(True)
                await self.query_one("#conversation").mount(ChatMessage("You", data, user=True))
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
                self.set_conversation_mode(False)
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
        self.query_one("#shortcuts", Static).update("esc interrupt  ctrl+p commands" if busy else "ctrl+p commands")

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

    @on(Composer.Changed, "#composer")
    def composer_changed(self) -> None:
        field = self.query_one(Composer)
        field.styles.height = min(6, max(2, len(field.value.splitlines()) + int(field.value.endswith("\n"))))
        document = Document(field.value, cursor_position=field.cursor_position)
        self.completions = list(self.completer.get_completions(document, None))[:30]
        if field.value.startswith("/") and " " not in field.value:
            for command, description in {"/copy": "Copy the last response", "/commands": "Open commands"}.items():
                if command.startswith(field.value):
                    self.completions.append(Completion(command, start_position=-len(field.value), display_meta=description))
        options = self.query_one("#completions", OptionList)
        options.clear_options()
        options.display = bool(self.completions) and not self.busy
        self.query_one("#workspace").set_class(options.display, "completing")
        for item in self.completions:
            label = Text(item.text)
            label.append("  " + item.display_meta_text, style="dim")
            options.add_option(Option(label))
        if self.completions:
            options.highlighted = 0

    @on(Composer.Submitted, "#composer")
    def submit(self, event: Composer.Submitted) -> None:
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
        elif text == "/copy":
            self.action_copy_response()
        elif text == "/commands":
            self.action_commands()
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
        if "\n" in field.value:
            if direction < 0:
                field.action_cursor_up()
            else:
                field.action_cursor_down()
            return
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

    def copy_to_clipboard(self, text: str) -> None:
        if write_clipboard(text):
            self._clipboard = text
        else:
            super().copy_to_clipboard(text)

    def selected_text(self) -> str:
        if isinstance(self.focused, Input) and self.focused.password:
            return ""
        selected = self.screen.get_selected_text()
        if selected:
            return selected
        focused = self.focused
        if isinstance(focused, (Input, TextArea)):
            return focused.selected_text
        return ""

    def action_copy_selection(self) -> None:
        selected = self.selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied")

    def action_copy_or_stop(self) -> None:
        if self.selected_text():
            self.action_copy_selection()
        else:
            self.action_stop()

    def action_copy_response(self) -> None:
        content = self.response.content if self.response else next(
            (item["content"] for item in reversed(self.backend.chat.history()) if item["role"] == "assistant"), ""
        )
        if content:
            self.copy_to_clipboard(content)
            self.notify("Response copied")
        else:
            self.notify("No response to copy", severity="warning")

    def action_paste(self) -> None:
        focused = self.focused
        if not isinstance(focused, (Input, TextArea)):
            focused = self.query_one(Composer)
            focused.focus()
        text = read_clipboard()
        if text is None:
            text = self.clipboard
        if not text:
            return
        if isinstance(focused, TextArea):
            result = focused.replace(text.replace("\r\n", "\n"), *focused.selection)
            focused.move_cursor(result.end_location)
        else:
            focused.replace(text.replace("\r\n", "\n").replace("\n", " "), *focused.selection)

    def on_mouse_down(self, event) -> None:
        if event.button == 3:
            if self.selected_text():
                self.action_copy_selection()
            else:
                self.action_paste()
            event.stop()

    def action_commands(self) -> None:
        items = [("Copy last response", "ui:copy")]
        if self.busy:
            items += [("Stop generation", "ui:stop"), ("Quit", "ui:quit")]
        else:
            items += [("Models", "ui:models"), ("Chats", "ui:chats"), ("Settings", "ui:settings"),
                      ("New chat", "/chat new")]
            items += [(f"{command}  {description}", command) for command, description in TOP.items()]
        self.push_screen(Picker("Commands", items), self.command_selected)

    def command_selected(self, command: str | None) -> None:
        if not command:
            return
        actions = {
            "ui:copy": self.action_copy_response, "ui:stop": self.action_stop,
            "ui:quit": self.action_request_quit, "ui:models": self.action_models,
            "ui:chats": self.action_chats, "ui:settings": self.action_settings,
            "/exit": self.action_request_quit, "/quit": self.action_request_quit,
        }
        if command in actions:
            actions[command]()
        else:
            field = self.query_one(Composer)
            field.value = command + " "
            field.cursor_position = len(field.value)
            field.focus()

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
