from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import CHATS_DIR, ensure_dirs, safe_model_name


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def new(cls, model: str | None, mode: str) -> "ChatStore":
        ensure_dirs()
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        store = cls(CHATS_DIR / f"{name}.jsonl")
        store.append({"type": "meta", "created_at": now_iso(), "model": model, "mode": mode})
        return store

    @classmethod
    def latest(cls) -> "ChatStore | None":
        ensure_dirs()
        files = sorted(CHATS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cls(files[0]) if files else None

    @classmethod
    def named(cls, name: str) -> "ChatStore | None":
        ensure_dirs()
        target = name if name.endswith(".jsonl") else f"{name}.jsonl"
        path = CHATS_DIR / target
        return cls(path) if path.exists() else None

    @staticmethod
    def list() -> list[Path]:
        ensure_dirs()
        return sorted(CHATS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event.setdefault("created_at", now_iso())
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
            f.write("\n")

    def message(self, role: str, content: str) -> None:
        self.append({"type": "message", "role": role, "content": content})

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def history(self) -> list[dict[str, str]]:
        return [
            {"role": e["role"], "content": e["content"]}
            for e in self.events()
            if e.get("type") == "message" and e.get("role") in {"user", "assistant"}
        ]

    def rename(self, name: str) -> Path:
        target = self.path.with_name(f"{safe_model_name(name)}.jsonl")
        self.path.replace(target)
        self.path = target
        return target

    def export_markdown(self) -> Path:
        out = self.path.with_suffix(".md")
        lines = ["# Veyra Chat", ""]
        for event in self.events():
            if event.get("type") == "message":
                role = event.get("role", "message").title()
                lines.extend([f"## {role}", "", event.get("content", ""), ""])
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
