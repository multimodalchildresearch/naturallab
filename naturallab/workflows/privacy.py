"""Portable path identities and bounded diagnostics for workflow metadata."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any


ERROR_MESSAGE_MAX_LENGTH = 320
ERROR_TYPE_MAX_LENGTH = 80

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_QUOTED_ABSOLUTE_PATH_RE = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:/|~/|[A-Za-z]:[\\/]|\\\\).+?)(?P=quote)"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s,;!?()\[\]{}<>\"']+"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:~?/)[^\s,;!?()\[\]{}<>\"']+"
)
_URL_RE = re.compile(
    r"(?i)\b(?:file|ftp|https?|rtsps?|sftp|ssh|wss?)://"
    r"[^\s<>{}\[\]\"']+"
)
_SCP_ENDPOINT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])"
    r"(?:[A-Za-z0-9_.-]+@)"
    r"(?:localhost|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|"
    r"(?:\d{1,3}\.){3}\d{1,3}|[A-F0-9:]+)"
    r"(?::[^\s,;]+)?"
)
_HOST_ENDPOINT_RE = re.compile(
    r"(?i)\b(?:localhost(?::\d{1,5})?|"
    r"(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?|"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+:\d{1,5})\b"
)
_NAMED_ENDPOINT_RE = re.compile(
    r"(?i)\b(?:endpoint|host|hostname|server|url|uri)\b"
    r"\s*(?::|=|\bis\b)\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>[A-Za-z0-9_]*(?:api[_-]?key|access[_-]?key|"
    r"authorization|credential|password|passwd|private[_-]?key|secret|"
    r"token)[A-Za-z0-9_]*)\b"
    r"\s*(?::|=|\bis\b|\s)\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_FLAG_RE = re.compile(
    r"(?i)--(?:api[_-]?key|access[_-]?key|authorization|credential|"
    r"password|passwd|private[_-]?key|secret|token)"
    r"(?:=|\s+)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTH_HEADER_RE = re.compile(
    r"(?i)\bauthorization\b\s*(?::|=)\s*"
    r"(?:(?:basic|bearer)\s+)?"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTH_VALUE_RE = re.compile(
    r"(?i)\b(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]+"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:github_pat_[A-Za-z0-9_]+|"
    r"gh[pousr]_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]+|"
    r"AKIA[A-Z0-9]{16})(?![A-Za-z0-9])"
)
_LONG_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{40,}(?![A-Za-z0-9])"
)
_PEM_RE = re.compile(
    r"-----BEGIN [^-]+-----.*?-----END [^-]+-----",
    flags=re.IGNORECASE,
)


def _bounded(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    suffix = "... [truncated]"
    return value[: maximum - len(suffix)].rstrip() + suffix


def _safe_basename(value: str) -> str:
    windows_name = PureWindowsPath(value).name
    posix_name = Path(value).name
    name = windows_name if "\\" in value else posix_name
    name = _SAFE_NAME_RE.sub("_", name).strip("._-")
    return _bounded(name, 80) if name else "item"


def _path_placeholder(match: re.Match[str]) -> str:
    value = match.groupdict().get("path") or match.group(0)
    return f"<path:{_safe_basename(value)}>"


def portable_path_identity(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> str:
    """Return a useful path label without exposing an absolute location.

    Paths inside ``base_dir`` are represented relative to that directory.
    External absolute paths use only a sanitized basename and an opaque hash so
    that two same-named inputs can still be distinguished in a report.
    """

    raw = os.fspath(path)
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() and os.name != "nt":
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{_safe_basename(raw)} [path-id:{digest}]"

    candidate = Path(raw).expanduser()
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        resolved = candidate.absolute()

    if root is not None:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            pass
        else:
            label = relative.as_posix()
            return label if label and label != "." else _safe_basename(raw)

    if not resolved.is_absolute():
        label = resolved.as_posix()
        return label if label and label != "." else _safe_basename(raw)

    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{_safe_basename(raw)} [path-id:{digest}]"


def sanitize_error_message(message: Any) -> str:
    """Return one bounded diagnostic line with common sensitive data removed."""

    try:
        text = str(message)
    except Exception:
        text = "operation failed with an unreadable error message"
    text = " ".join(text.split())
    text = _PEM_RE.sub("<redacted-secret>", text)
    text = _URL_RE.sub("<endpoint>", text)
    text = _SCP_ENDPOINT_RE.sub("<endpoint>", text)
    text = _NAMED_ENDPOINT_RE.sub("endpoint=<endpoint>", text)
    text = _AUTH_HEADER_RE.sub("authorization=<redacted-secret>", text)
    text = _AUTH_VALUE_RE.sub("authorization=<redacted-secret>", text)
    text = _SECRET_FLAG_RE.sub("--credential=<redacted-secret>", text)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}=<redacted-secret>",
        text,
    )
    text = _KNOWN_TOKEN_RE.sub("<redacted-secret>", text)
    text = _HOST_ENDPOINT_RE.sub("<endpoint>", text)
    text = _QUOTED_ABSOLUTE_PATH_RE.sub(_path_placeholder, text)
    text = _WINDOWS_ABSOLUTE_PATH_RE.sub(_path_placeholder, text)
    text = _POSIX_ABSOLUTE_PATH_RE.sub(_path_placeholder, text)
    text = _LONG_TOKEN_RE.sub("<redacted-secret>", text)
    text = " ".join(text.split()) or "operation failed"
    return _bounded(text, ERROR_MESSAGE_MAX_LENGTH)


def sanitize_error_type(error_type: Any) -> str:
    """Return a bounded non-sensitive exception type label."""

    try:
        value = str(error_type).strip()
    except Exception:
        return "Error"
    if not _SAFE_ERROR_TYPE_RE.fullmatch(value):
        return "Error"
    return _bounded(value, ERROR_TYPE_MAX_LENGTH)
