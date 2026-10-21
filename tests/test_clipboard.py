import ctypes
import unittest
from unittest.mock import MagicMock, patch

from veyra.clipboard import _windows_clipboard


@unittest.skipUnless(hasattr(ctypes, "WinDLL"), "Win32 clipboard contract")
class WindowsClipboardTests(unittest.TestCase):
    def test_unicode_write_uses_owner_window_and_transfers_memory(self):
        user, kernel = MagicMock(), MagicMock()
        memory = ctypes.create_string_buffer(1024)
        user.CreateWindowExW.return_value = 123
        user.OpenClipboard.return_value = 1
        user.EmptyClipboard.return_value = 1
        user.SetClipboardData.return_value = 456
        kernel.GlobalAlloc.return_value = 456
        kernel.GlobalLock.return_value = ctypes.addressof(memory)
        with patch.object(ctypes, "WinDLL", side_effect=[user, kernel]):
            self.assertTrue(_windows_clipboard("hello\n\u00e9\U0001f600"))
        user.OpenClipboard.assert_called_once_with(123)
        user.SetClipboardData.assert_called_once_with(13, 456)
        user.CloseClipboard.assert_called_once()
        user.DestroyWindow.assert_called_once_with(123)
        kernel.GlobalFree.assert_not_called()
        self.assertTrue(memory.raw.startswith("hello\n\u00e9\U0001f600\0".encode("utf-16-le")))

    def test_read_does_not_clear_or_overwrite_clipboard(self):
        user, kernel = MagicMock(), MagicMock()
        memory = ctypes.create_unicode_buffer("hello\nworld")
        user.OpenClipboard.return_value = 1
        user.GetClipboardData.return_value = 456
        kernel.GlobalLock.return_value = ctypes.addressof(memory)
        with patch.object(ctypes, "WinDLL", side_effect=[user, kernel]):
            self.assertEqual(_windows_clipboard(), "hello\nworld")
        user.EmptyClipboard.assert_not_called()
        user.SetClipboardData.assert_not_called()
        user.CloseClipboard.assert_called_once()
