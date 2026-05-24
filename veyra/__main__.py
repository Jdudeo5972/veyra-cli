from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hf import download_model, list_veyra_models, registry_entry
from .inspect import format_inspection, inspect_model
from .prompts import format_prompt, infer_prompt_mode
from .registry import load_config, register_model, safe_model_name
from .runner import OnnxCausalLMRunner
from .shell import VeyraShell, find_model_dirs, make_local_entry, update_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veyra", description="Run local ONNX causal language models.")
    parser.add_argument("prompt", nargs="?", help="Prompt text to run, or a subcommand.")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    parser.add_argument("--no-load", action="store_true", help="Do not autoload the current model in the shell.")
    parser.add_argument("--continue", dest="continue_chat", action="store_true", help="Resume the most recent chat.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or all(a.startswith("--") for a in argv):
        args = build_parser().parse_args(argv)
        return VeyraShell(args).run()

    command = argv[0]
    if command == "run":
        return run_prompt(" ".join(argv[1:]).strip())
    if command == "models":
        return models_cmd()
    if command == "fetch":
        return fetch_cmd()
    if command == "add":
        return add_cmd(argv[1:])
    if command == "inspect":
        return inspect_cmd(argv[1:])
    if command == "update":
        update_message()
        return 0
    if command.startswith("-"):
        args = build_parser().parse_args(argv)
        return VeyraShell(args).run()
    return run_prompt(" ".join(argv))


def run_prompt(prompt: str) -> int:
    if not prompt:
        print("Usage: veyra run \"prompt text\"")
        return 2
    config = load_config()
    name = config.get("current_model")
    entry = config.get("models", {}).get(name)
    if not entry:
        print("Missing model. Use `veyra fetch` or `veyra add PATH`.")
        return 1
    mode = config.get("current_mode", "chatml")
    formatted = format_prompt(prompt, mode)
    runner = OnnxCausalLMRunner(entry["path"], device=config.get("device", "cpu"))
    for delta in runner.generate(formatted, **config.get("defaults", {})):
        print(delta, end="", flush=True)
    print("")
    return 0


def models_cmd() -> int:
    config = load_config()
    current = config.get("current_model")
    installed = config.get("models", {})
    if not installed:
        print("No models installed.")
        return 0
    for name, entry in installed.items():
        mark = "*" if name == current else " "
        print(f"{mark} {name} ({entry.get('source', 'unknown')})")
    return 0


def fetch_cmd() -> int:
    try:
        choices = list_veyra_models()
    except Exception as exc:
        print(f"Could not query Hugging Face: {exc}")
        return 1
    for idx, item in enumerate(choices, 1):
        print(f"{idx}. {item['repo_id']}")
    if not choices:
        return 0
    raw = input("Select model number: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        print("Cancelled.")
        return 1
    repo_id = choices[int(raw) - 1]["repo_id"]
    path, commit = download_model(repo_id)
    config = load_config()
    register_model(config, safe_model_name(repo_id), registry_entry(repo_id, path, commit=commit))
    print(f"Fetched and selected {repo_id}.")
    return 0


def add_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veyra add")
    parser.add_argument("path")
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    root = Path(args.path).expanduser()
    scanned = find_model_dirs(root)
    if len(scanned) > 1 or (scanned and scanned[0] != root.resolve()):
        config = load_config()
        for candidate in scanned:
            try:
                info = inspect_model(candidate)
                if info.supported:
                    name = safe_model_name(info.model_dir.name)
                    register_model(config, name, make_local_entry(info))
                    print(f"Added {name}.")
            except Exception as exc:
                print(f"Skipping {candidate}: {exc}")
        return 0
    info = inspect_model(args.path)
    if not info.supported:
        print(format_inspection(info))
        return 1
    config = load_config()
    name = args.name or safe_model_name(info.model_dir.name)
    entry = make_local_entry(info)
    register_model(config, name, entry)
    print(f"Added and selected {name}.")
    return 0


def _read_json(path):
    try:
        import json
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def inspect_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veyra inspect")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    print(format_inspection(inspect_model(args.path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
