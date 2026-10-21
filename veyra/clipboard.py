"""System clipboard access with standard-library tools and optional OS utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _windows_clipboard(text: str | None = None) -> str | bool | None:
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user.OpenClipboard.argtypes = [wintypes.HWND]
    user.OpenClipboard.restype = wintypes.BOOL
    user.GetClipboardData.argtypes = [wintypes.UINT]
    user.GetClipboardData.restype = wintypes.HANDLE
    user.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user.SetClipboardData.restype = wintypes.HANDLE
    user.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user.CreateWindowExW.restype = wintypes.HWND
    user.DestroyWindow.argtypes = [wintypes.HWND]
    kernel.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel.GlobalLock.restype = ctypes.c_void_p
    kernel.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel.GlobalFree.argtypes = [wintypes.HGLOBAL]
    # Windows requires an owner window when EmptyClipboard precedes SetClipboardData.
    owner = (
        user.CreateWindowExW(0, "STATIC", "Veyra clipboard", 0, 0, 0, 0, 0, None, None, None, None)
        if text is not None else None
    )
    if text is not None and not owner:
        return False
    if not user.OpenClipboard(owner):
        if owner:
            user.DestroyWindow(owner)
        return None
    try:
        if text is None:
            handle = user.GetClipboardData(13)  # CF_UNICODETEXT
            address = kernel.GlobalLock(handle) if handle else None
            if not address:
                return None
            try:
                return ctypes.wstring_at(address)
            finally:
                kernel.GlobalUnlock(handle)
        data = (text + "\0").encode("utf-16-le")
        handle = kernel.GlobalAlloc(0x0002, len(data))  # Clipboard takes ownership of movable memory.
        if not handle:
            return False
        address = kernel.GlobalLock(handle)
        if not address:
            kernel.GlobalFree(handle)
            return False
        ctypes.memmove(address, data, len(data))
        kernel.GlobalUnlock(handle)
        if not user.EmptyClipboard() or not user.SetClipboardData(13, handle):
            kernel.GlobalFree(handle)
            return False
        return True
    finally:
        user.CloseClipboard()
        if owner:
            user.DestroyWindow(owner)


def _command(paste: bool) -> list[str] | None:
    if sys.platform == "darwin":
        return ["pbpaste" if paste else "pbcopy"]
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste" if paste else "wl-copy"):
        return ["wl-paste", "--no-newline"] if paste else ["wl-copy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard", "-o" if paste else "-i"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--output" if paste else "--input"]
    return None


def read_clipboard() -> str | None:
    try:
        if sys.platform == "win32":
            value = _windows_clipboard()
            return value if isinstance(value, str) else None
        command = _command(paste=True)
        if command:
            return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=1, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def write_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            return _windows_clipboard(text) is True
        command = _command(paste=False)
        if command:
            subprocess.run(
                command, input=text, text=True, encoding="utf-8", stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=1, check=True,
            )
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False
