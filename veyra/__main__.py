from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from .hf import (
    download_model,
    fetched_model_name,
    get_hf_model,
    list_veyra_models,
    recommended_onnx_file,
    registry_entry,
    variant_name,
)
from .auth import hf_auth_status, login_hf_read_only, logout_hf
from .inspect import format_inspection, inspect_model
from .registry import load_config, register_model, safe_model_name
from .runner import create_runner
from .shell import VeyraShell, find_model_dirs, format_transformers_inspection, make_local_model_entry, update_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veyra", description="Run local ONNX and Transformers language models.")
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
        return fetch_cmd(argv[1] if len(argv) >= 2 else None)
    if command == "auth":
        return auth_cmd(argv[1:])
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
    profile = entry.get("profile", {}) if isinstance(entry.get("profile"), dict) else {}
    mode = profile.get("mode", config.get("current_mode", "chatml"))
    try:
        runner = create_runner(entry, device=config.get("device", "cpu"))
        formatted = runner.format_conversation(prompt, mode, [], None)
        defaults = dict(config.get("defaults", {}))
        if isinstance(profile.get("generation"), dict):
            defaults.update(profile["generation"])
        for delta in runner.generate(formatted, **defaults):
            print(delta, end="", flush=True)
        print("")
        return 0
    except KeyboardInterrupt:
        print("\nGeneration stopped.")
        return 130
    except Exception as exc:
        print(f"Generation failed: {exc}")
        return 1


def models_cmd() -> int:
    config = load_config()
    current = config.get("current_model")
    installed = config.get("models", {})
    if not installed:
        print("No models installed.")
        return 0
    for name, entry in installed.items():
        mark = "*" if name == current else " "
        print(f"{mark} {name} ({entry.get('source', 'unknown')}, {entry.get('runtime', 'onnx')})")
    return 0


def auth_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veyra auth")
    parser.add_argument("action", nargs="?", choices=("status", "login", "logout"), default="status")
    args = parser.parse_args(argv)
    try:
        if args.action == "login":
            if not sys.stdin.isatty():
                print("Login requires an interactive terminal so the token can be entered securely.")
                return 2
            print("Create a read or fine-grained read token at https://huggingface.co/settings/tokens")
            status = login_hf_read_only(getpass.getpass("HF read token: "))
            print(f"Signed in to Hugging Face as {status['username']} ({auth_role_label(status)}).")
            return 0
        if args.action == "logout":
            logout_hf()
            print("Hugging Face logout complete.")
            return 0
        status = hf_auth_status()
        if not status["authenticated"]:
            print(f"Hugging Face: not signed in. {status['error']}")
            return 1
        print(f"account: {status['username']}")
        print(f"token: {status['token_name']}")
        print(f"permission: {auth_role_label(status)}")
        print(f"source: {status['source']}")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"Hugging Face authentication failed: {exc}")
        return 1


def auth_role_label(status: dict) -> str:
    role = str(status.get("role") or "unknown")
    if role.casefold() == "read":
        return "read-only"
    if role.casefold() == "finegrained" and status.get("read_only"):
        return "fine-grained read-only"
    return role


def fetch_cmd(repo_id: str | None = None) -> int:
    if repo_id:
        try:
            selected = get_hf_model(repo_id)
        except Exception as exc:
            print(f"Fetch failed: {exc}")
            return 1
    else:
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
        selected = choices[int(raw) - 1]
    repo_id = selected["repo_id"]
    runtime_choice = select_runtime_variant(selected)
    if not runtime_choice:
        return 1
    runtime, onnx_file = runtime_choice
    try:
        path, commit = download_model(repo_id, onnx_file=onnx_file, runtime=runtime)
        config = load_config()
        name = fetched_model_name(repo_id, onnx_file, len(selected["onnx_files"]), runtime=runtime)
        entry = registry_entry(
            repo_id,
            path,
            commit=commit,
            onnx_file=onnx_file,
            runtime=runtime,
        )
        register_model(config, name, entry)
    except Exception as exc:
        print(f"Fetch failed: {exc}")
        return 1
    variant = variant_name(onnx_file) if onnx_file else "Safetensors"
    print(f"Fetched and selected {name} ({runtime}: {variant}).")
    return 0


def select_runtime_variant(model: dict) -> tuple[str, str | None] | None:
    variants = model.get("onnx_files", [])
    choices: list[tuple[str, str | None, str]] = []
    if model.get("has_transformers"):
        choices.append(("transformers", None, "Transformers (Safetensors)"))
    recommended = recommended_onnx_file(variants) if variants else None
    for path in variants:
        suffix = " (recommended lightweight runtime)" if path == recommended else ""
        choices.append(("onnx", path, f"ONNX: {path}{suffix}"))
    if not choices:
        print(f"No supported model files found in {model['repo_id']}.")
        return None
    if len(choices) == 1:
        return choices[0][0], choices[0][1]
    print("Available runtimes and variants:")
    for idx, (_, _, label) in enumerate(choices, 1):
        print(f"{idx}. {label}")
    default = next((idx for idx, item in enumerate(choices, 1) if item[0] == "onnx" and item[1] == recommended), 1)
    raw = input(f"Select runtime number [{default}]: ").strip()
    if not raw:
        return choices[default - 1][0], choices[default - 1][1]
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        print("Cancelled.")
        return None
    chosen = choices[int(raw) - 1]
    return chosen[0], chosen[1]


def add_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veyra add")
    parser.add_argument("path")
    parser.add_argument("--name")
    parser.add_argument("--runtime", choices=("auto", "onnx", "transformers"), default="auto")
    args = parser.parse_args(argv)
    root = Path(args.path).expanduser()
    scanned = find_model_dirs(root)
    if len(scanned) > 1 or (scanned and scanned[0] != root.resolve()):
        config = load_config()
        for candidate in scanned:
            try:
                entry = make_local_model_entry(candidate, runtime=args.runtime)
                name = safe_model_name(candidate.name)
                register_model(config, name, entry)
                print(f"Added {name}.")
            except Exception as exc:
                print(f"Skipping {candidate}: {exc}")
        return 0
    config = load_config()
    model_dir = Path(args.path).expanduser().resolve()
    name = args.name or safe_model_name(model_dir.name)
    entry = make_local_model_entry(model_dir, runtime=args.runtime)
    register_model(config, name, entry)
    print(f"Added and selected {name}.")
    return 0


def inspect_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veyra inspect")
    parser.add_argument("path")
    parser.add_argument("--runtime", choices=("auto", "onnx", "transformers"), default="auto")
    args = parser.parse_args(argv)
    root = Path(args.path).expanduser().resolve()
    has_transformers = bool(list(root.glob("*.safetensors")))
    has_onnx = bool(list(root.glob("*.onnx")) or list(root.rglob("*.onnx")))
    if args.runtime == "transformers" or (args.runtime == "auto" and has_transformers and not has_onnx):
        print(format_transformers_inspection(root))
    elif args.runtime == "onnx" or has_onnx:
        print(format_inspection(inspect_model(root)))
        if args.runtime == "auto" and has_transformers:
            print("\n" + format_transformers_inspection(root))
    else:
        print(f"No .onnx or .safetensors model files found in {root}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
