import argparse
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, OptionList
from textual.widgets.text_area import Selection as EditorSelection
from textual.selection import Selection
from textual.geometry import Offset
from textual import events

from veyra.chat_store import ChatStore
from veyra.registry import default_config
from veyra.theme import Theme
from veyra.tui import ChatMessage, Composer, Picker, Question, VeyraApp


class FakeRunner:
    max_context_length = 1024
    is_encoder_decoder = False

    def format_conversation(self, text, mode, history, system):
        return text

    def token_count(self, text):
        return len(text.split())

    def generate(self, prompt, **kwargs):
        for chunk in ["Hello", " from", " **Veyra**."]:
            time.sleep(0.03)
            yield chunk


class SlowRunner(FakeRunner):
    def generate(self, prompt, **kwargs):
        for _ in range(100):
            time.sleep(0.03)
            yield "word "


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = default_config()
        self.chat = ChatStore(self.root / "chat.jsonl")
        self.patches = [
            patch("veyra.shell.load_config", return_value=self.config),
            patch("veyra.shell.ChatStore.new", return_value=self.chat),
            patch("veyra.shell.save_config"),
            patch("veyra.tui.HISTORY_PATH", self.root / "history.txt"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def app(self, no_load=True):
        return VeyraApp(argparse.Namespace(no_load=no_load, continue_chat=False))

    async def wait_idle(self, app, pilot):
        for _ in range(100):
            await pilot.pause(0.03)
            if not app.busy:
                return
        self.fail("TUI worker did not finish")

    def install_fake_model(self):
        self.config["current_model"] = "tiny"
        self.config["models"]["tiny"] = {"path": str(self.root), "runtime": "transformers"}

    async def test_empty_start_completion_and_settings(self):
        app = self.app()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            self.assertEqual(app.state, "no model")
            field = app.query_one(Composer)
            field.value = "/mo"
            field.cursor_position = len(field.value)
            await pilot.pause()
            await pilot.press("down", "tab")
            self.assertEqual(field.value, "/mode ")
            field.value = ""
            await pilot.press("f4")
            await pilot.pause()
            self.assertIsInstance(app.screen, Picker)
            app.screen.query_one(Input).value = "Temperature"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, Question)
            app.screen.query_one(Input).value = "0.2"
            await pilot.press("enter")
            await self.wait_idle(app, pilot)
            self.assertEqual(self.config["defaults"]["temperature"], 0.2)

    async def test_streaming_autosave_markdown_and_history(self):
        self.install_fake_model()
        app = self.app()
        app.backend.runner = FakeRunner()
        self.config["stats"] = True
        async with app.run_test(size=(100, 34)) as pilot:
            app.query_one(Composer).value = "hello"
            await pilot.press("enter")
            await self.wait_idle(app, pilot)
            self.assertEqual(self.chat.history(), [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello from **Veyra**."},
            ])
            messages = list(app.query(ChatMessage))
            self.assertEqual(len(messages), 2)
            self.assertFalse(messages[-1].streaming)
            self.assertEqual(app.state, "ready")
            await pilot.press("up")
            self.assertEqual(app.query_one(Composer).value, "hello")
            self.assertIn("stats:", " ".join(str(widget.render()) for widget in app.query(".command-output")))

    async def test_cancel_saves_partial_and_leaves_app_usable(self):
        self.install_fake_model()
        app = self.app()
        app.backend.runner = SlowRunner()
        async with app.run_test(size=(80, 24)) as pilot:
            app.start_operation("keep talking")
            for _ in range(30):
                await pilot.pause(0.03)
                if app.response and app.response.content:
                    break
            self.assertTrue(app.busy)
            await pilot.press("ctrl+c")
            await self.wait_idle(app, pilot)
            history = self.chat.history()
            self.assertEqual(history[0]["content"], "keep talking")
            self.assertGreater(len(history[-1]["content"]), 0)
            self.assertLess(len(history[-1]["content"]), 500)
            app.start_operation("/help")
            await self.wait_idle(app, pilot)
            self.assertFalse(app.busy)

    async def test_load_failure_and_model_picker(self):
        self.install_fake_model()
        with patch("veyra.shell.create_runner", side_effect=RuntimeError("broken model")):
            app = self.app(no_load=False)
            async with app.run_test(size=(80, 24)) as pilot:
                await self.wait_idle(app, pilot)
                self.assertEqual(app.state, "failed")
                self.assertIn("broken model", app.backend.load_error)
                await pilot.press("f2")
                await pilot.pause()
                self.assertIsInstance(app.screen, Picker)
                self.assertEqual(app.screen.query_one(OptionList).option_count, 2)
                await pilot.press("escape")
                self.assertNotIsInstance(app.screen, Picker)

    async def test_worker_question_cancel_does_not_block(self):
        app = self.app()
        with patch.object(app.backend, "remote_list", return_value=[{
            "repo_id": "test/model", "onnx_files": [], "has_transformers": True,
        }]):
            async with app.run_test(size=(80, 24)) as pilot:
                app.start_operation("/model fetch")
                for _ in range(50):
                    await pilot.pause(0.03)
                    if isinstance(app.screen, Question):
                        break
                self.assertIsInstance(app.screen, Question)
                await pilot.press("escape")
                await self.wait_idle(app, pilot)
                self.assertFalse(app.busy)

    async def test_retry_and_resume_render_saved_history(self):
        self.install_fake_model()
        self.chat.message("user", "hello")
        self.chat.message("assistant", "old response")
        app = self.app()
        app.backend.runner = FakeRunner()
        async with app.run_test(size=(100, 34)) as pilot:
            self.assertEqual(len(list(app.query(ChatMessage))), 2)
            app.start_operation("/retry")
            await self.wait_idle(app, pilot)
            self.assertEqual(self.chat.history()[-1]["content"], "Hello from **Veyra**.")
            self.assertEqual(len(list(app.query(ChatMessage))), 2)

    async def test_theme_switch_history_suggestion_and_resize(self):
        app = self.app()
        app.history.append("Hello from history")
        async with app.run_test(size=(80, 24)) as pilot:
            field = app.query_one(Composer)
            field.value = "Hell"
            field.cursor_position = len(field.value)
            await pilot.press("tab")
            self.assertEqual(field.value, "Hello from history")
            with patch("veyra.tui.get_theme", return_value=Theme("warm", enabled=True)):
                app.start_operation("/theme warm")
                await self.wait_idle(app, pilot)
                await pilot.pause()
            self.assertEqual(self.config["theme"], "warm")
            self.assertEqual(app.current_theme.primary, "#CE562F")
            self.assertEqual(app.theme, "veyra-ui-warm")
            color = app.query_one("#composer-panel").styles.border_left[1]
            self.assertLessEqual(abs(color.r - 206) + abs(color.g - 86) + abs(color.b - 47), 3)
            await pilot.resize_terminal(60, 24)
            await pilot.pause()
            self.assertLessEqual(field.region.bottom, app.size.height)
            self.assertLessEqual(field.region.right, app.size.width)
            self.assertGreater(app.query_one("#conversation").size.height, 0)

    async def test_secret_dialog_does_not_record_token(self):
        app = self.app()
        with patch("veyra.shell.login_hf_read_only", return_value={"username": "tester", "role": "read"}) as login:
            async with app.run_test(size=(80, 24)) as pilot:
                app.start_operation("/hf login")
                for _ in range(50):
                    await pilot.pause(0.03)
                    if isinstance(app.screen, Question) and app.screen.is_mounted:
                        break
                await pilot.pause()
                answer = app.screen.query_one(Input)
                self.assertTrue(answer.password)
                answer.value = "test-secret-not-a-real-token"
                await pilot.press("enter")
                await self.wait_idle(app, pilot)
                login.assert_called_once_with("test-secret-not-a-real-token")
                self.assertNotIn("test-secret", " ".join(app.history))
                self.assertFalse((self.root / "history.txt").exists())

    async def test_multiline_system_paste_does_not_submit(self):
        self.install_fake_model()
        app = self.app()
        app.backend.runner = FakeRunner()
        with patch("veyra.tui.read_clipboard", return_value="second\r\nthird"):
            async with app.run_test(size=(100, 34)) as pilot:
                field = app.query_one(Composer)
                field.value = "first"
                field.cursor_position = 5
                await pilot.press("shift+enter", "ctrl+v")
                await pilot.pause()
                self.assertEqual(field.value, "first\nsecond\nthird")
                self.assertFalse(app.busy)
                self.assertEqual(self.chat.history(), [])
                await pilot.press("enter")
                await self.wait_idle(app, pilot)
                self.assertEqual(self.chat.history()[0]["content"], "first\nsecond\nthird")

    async def test_copy_selection_takes_priority_over_stop(self):
        app = self.app()
        with patch("veyra.tui.write_clipboard", return_value=True) as copy:
            async with app.run_test(size=(100, 34)) as pilot:
                field = app.query_one(Composer)
                field.value = "copy this"
                field.selection = EditorSelection((0, 0), (0, 4))
                app.set_busy(True)
                await pilot.press("ctrl+c")
                copy.assert_called_once_with("copy")
                self.assertFalse(app.stop_requested.is_set())
                field.move_cursor((0, 9))
                await pilot.press("ctrl+c")
                self.assertTrue(app.stop_requested.is_set())
                app.set_busy(False)

    async def test_response_selection_and_copy_last_response(self):
        self.chat.message("user", "hello")
        self.chat.message("assistant", "A response with **formatting**.")
        app = self.app()
        with patch("veyra.tui.write_clipboard", return_value=True) as copy:
            async with app.run_test(size=(100, 34)) as pilot:
                body = app.query_one(".user-message .message-body")
                app.screen.selections = {body: Selection(Offset(0, 0), Offset(5, 0))}
                await pilot.press("ctrl+shift+c")
                copy.assert_called_with("hello")
                app.clear_selection()
                app.action_copy_response()
                copy.assert_called_with("A response with **formatting**.")
                await pilot.press("ctrl+p")
                await pilot.pause()
                self.assertIsInstance(app.screen, Picker)
                self.assertEqual(app.screen.heading, "Commands")

    async def test_internal_clipboard_and_terminal_paste_keep_newlines(self):
        app = self.app()
        with patch("veyra.tui.write_clipboard", return_value=False), patch("veyra.tui.read_clipboard", return_value=None):
            async with app.run_test(size=(80, 24)) as pilot:
                field = app.query_one(Composer)
                app.copy_to_clipboard("first\nsecond")
                await pilot.press("ctrl+v")
                self.assertEqual(field.value, "first\nsecond")
                field.post_message(events.Paste("\nthird\nfourth"))
                await pilot.pause()
                self.assertEqual(field.value, "first\nsecond\nthird\nfourth")
                self.assertFalse(app.busy)


if __name__ == "__main__":
    unittest.main()
