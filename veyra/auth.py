from __future__ import annotations

import os
from typing import Any


READ_ROLES = {"read"}
WRITE_MARKERS = ("write", "create", "delete", "update", "manage", "admin")
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def hf_auth_status() -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face authentication.") from exc

    token = get_token()
    source = "HF_TOKEN environment" if _has_environment_token() else "Hugging Face token store"
    if not token:
        return {"authenticated": False, "source": source, "error": "No token is configured."}
    try:
        return _status_from_whoami(HfApi().whoami(token=token), source)
    except Exception as exc:
        return {
            "authenticated": False,
            "source": source,
            "error": f"The configured token was rejected by Hugging Face ({type(exc).__name__}).",
        }


def login_hf_read_only(token: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, login
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face authentication.") from exc

    token = token.strip()
    if not token:
        raise RuntimeError("No token entered.")
    try:
        whoami = HfApi().whoami(token=token)
    except Exception as exc:
        raise RuntimeError(f"Hugging Face rejected the token ({type(exc).__name__}).") from exc
    access = _access_token_info(whoami)
    if not _is_read_only(access):
        role = str(access.get("role") or "unknown")
        raise RuntimeError(
            f"Token permission '{role}' is not verifiably read-only. "
            "Create a read token or a fine-grained token with read-only model access."
        )
    if _has_environment_token():
        raise RuntimeError(
            "HF_TOKEN is set in the environment and overrides saved credentials. "
            "Unset or replace that environment variable instead."
        )
    login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
    return _status_from_whoami(whoami, "Hugging Face token store")


def logout_hf() -> None:
    if _has_environment_token():
        raise RuntimeError(
            "HF_TOKEN is set in the environment. Remove it from the current shell or system environment to log out."
        )
    try:
        from huggingface_hub import get_token, logout
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face authentication.") from exc
    if get_token() is None:
        return
    status = hf_auth_status()
    if not status.get("authenticated"):
        raise RuntimeError(
            "The active token is invalid, so Veyra cannot identify it safely without removing other saved tokens. "
            "Use `hf auth logout` to clear Hugging Face's token store."
        )
    logout(token_name=status["token_name"])


def _status_from_whoami(whoami: dict[str, Any], source: str) -> dict[str, Any]:
    access = _access_token_info(whoami)
    return {
        "authenticated": True,
        "username": whoami.get("name") or "unknown",
        "role": access.get("role") or "unknown",
        "token_name": access.get("displayName") or "unnamed",
        "read_only": _is_read_only(access),
        "source": source,
    }


def _access_token_info(whoami: dict[str, Any]) -> dict[str, Any]:
    auth = whoami.get("auth")
    if not isinstance(auth, dict):
        return {}
    access = auth.get("accessToken")
    return access if isinstance(access, dict) else {}


def _is_read_only(access: dict[str, Any]) -> bool:
    role = str(access.get("role") or "").casefold()
    if role in READ_ROLES:
        return True
    if role != "finegrained":
        return False
    permissions = [value.casefold() for value in _permission_strings(access.get("fineGrained"))]
    if not permissions:
        return False
    if any(marker in permission for permission in permissions for marker in WRITE_MARKERS):
        return False
    return any("read" in permission for permission in permissions)


def _permission_strings(value: Any, key: str = "") -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for child_key, child in value.items():
            found.extend(_permission_strings(child, str(child_key)))
        return found
    if isinstance(value, (list, tuple, set)):
        found = []
        for child in value:
            found.extend(_permission_strings(child, key))
        return found
    if isinstance(value, str) and ("permission" in key.casefold() or "." in value):
        return [value]
    if value is True and "read" in key.casefold():
        return [key]
    return []


def _has_environment_token() -> bool:
    return any(os.environ.get(name) for name in HF_TOKEN_ENV_VARS)
