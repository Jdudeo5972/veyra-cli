from __future__ import annotations

import os
from dataclasses import dataclass


THEMES = ("veyra", "warm", "red", "pink", "lime", "green", "blue", "cyan", "purple", "orange", "gray", "rainbow", "mono")
THEME_ALIASES = {"neon": "blue"}
RESET = "\033[0m"


ANSI = {
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "white": "\033[97m",
    "veyra_pink": "\033[38;2;224;57;113m",
    "warm_orange": "\033[38;2;206;86;47m",
    "lime": "\033[92m",
    "gray": "\033[90m",
    "orange": "\033[38;2;255;140;0m",
}


ROLE_COLORS = {
    "veyra": {
        "title": "veyra_pink",
        "border": "veyra_pink",
        "status_ready": "white",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": "veyra_pink",
        "value": "",
        "muted": "dim",
        "command": "veyra_pink",
        "user_prompt": "white",
        "assistant_prompt": "veyra_pink",
        "success": "white",
        "warning": "yellow",
        "error": "red",
    },
    "warm": {
        "title": "warm_orange",
        "border": "warm_orange",
        "status_ready": "white",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": "warm_orange",
        "value": "",
        "muted": "dim",
        "command": "warm_orange",
        "user_prompt": "white",
        "assistant_prompt": "warm_orange",
        "success": "white",
        "warning": "yellow",
        "error": "red",
    },
    "red": {},
    "pink": {},
    "lime": {},
    "green": {
        "title": "green",
        "border": "green",
        "status_ready": "green",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": "green",
        "value": "",
        "muted": "dim",
        "command": "green",
        "user_prompt": "white",
        "assistant_prompt": "green",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    },
    "blue": {
        "title": "bright_cyan",
        "border": "cyan",
        "status_ready": "white",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": "cyan",
        "value": "",
        "muted": "dim",
        "command": "cyan",
        "user_prompt": "white",
        "assistant_prompt": "cyan",
        "success": "white",
        "warning": "yellow",
        "error": "red",
    },
    "cyan": {},
    "purple": {},
    "orange": {},
    "gray": {},
    "rainbow": {
        "title": "bright_magenta",
        "border": "bright_cyan",
        "status_ready": "white",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": "bright_yellow",
        "value": "",
        "muted": "dim",
        "command": "bright_magenta",
        "user_prompt": "white",
        "assistant_prompt": "bright_cyan",
        "success": "white",
        "warning": "yellow",
        "error": "red",
    },
    "mono": {},
}

for _theme, _color in {
    "red": "bright_red",
    "pink": "veyra_pink",
    "lime": "lime",
    "cyan": "bright_cyan",
    "purple": "bright_magenta",
    "orange": "orange",
    "gray": "gray",
}.items():
    ROLE_COLORS[_theme] = {
        "title": _color,
        "border": _color,
        "status_ready": "white" if _theme != "lime" else "lime",
        "status_loading": "yellow",
        "status_error": "red",
        "status_empty": "dim",
        "label": _color,
        "value": "",
        "muted": "dim",
        "command": _color,
        "user_prompt": "white",
        "assistant_prompt": _color,
        "success": "white" if _theme != "lime" else "lime",
        "warning": "yellow",
        "error": "red",
    }


PT_COLORS = {
    "dim": "ansiblack",
    "red": "ansired",
    "green": "ansigreen",
    "yellow": "ansiyellow",
    "blue": "ansiblue",
    "magenta": "ansimagenta",
    "cyan": "ansicyan",
    "bright_red": "ansibrightred",
    "bright_green": "ansibrightgreen",
    "bright_yellow": "ansibrightyellow",
    "bright_magenta": "ansibrightmagenta",
    "bright_cyan": "ansibrightcyan",
    "white": "ansiwhite",
    "veyra_pink": "#e03971",
    "warm_orange": "#CE562F",
    "lime": "ansibrightgreen",
    "gray": "ansibrightblack",
    "orange": "#ff8c00",
}


@dataclass
class Theme:
    name: str
    enabled: bool = True

    def color_name(self, role: str) -> str:
        if not self.enabled:
            return ""
        return ROLE_COLORS.get(self.name, {}).get(role, "")

    def text(self, role: str, value: str) -> str:
        color = self.color_name(role)
        if not color:
            return value
        return f"{ANSI[color]}{value}{RESET}"

    def prompt(self, role: str, value: str):
        color = self.color_name(role)
        style = PT_COLORS.get(color, "")
        return [(style, value)]


def normalize_theme(name: str | None) -> str:
    selected = THEME_ALIASES.get(str(name or ""), name)
    return selected if selected in THEMES else "veyra"


def get_theme(name: str | None) -> Theme:
    selected = normalize_theme(name)
    enabled = selected != "mono" and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    return Theme(selected, enabled)
