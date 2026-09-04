"""Small, dependency-light runtime provenance helpers.

Generated research reports should identify the software environment that made
them without making report generation depend on a Git checkout. Git metadata
is therefore best-effort, while package and interpreter versions are always
recorded. Caller-supplied parameters are recursively sanitized before storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import platform
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from naturallab import __version__


_SECRET_KEY = re.compile(
    r"(?:password|passwd|passphrase|token|api[_-]?key|secret|credential)",
    re.IGNORECASE,
)
_LOCAL_PATH_KEY = re.compile(
    r"(?:^|_)(?:path|file|directory|dir|video|manifest|bundle|intrinsics|floor)"
    r"(?:$|_)|^(?:input|output)$",
    re.IGNORECASE,
)


def _safe_value(value: Any, *, key: str = "") -> Any:
    if key and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Path) or (
        key and _LOCAL_PATH_KEY.search(key) and isinstance(value, str)
    ):
        return "<redacted-local-path>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _git_output(arguments: Sequence[str], *, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_provenance(*, cwd: Path | str | None = None) -> Mapping[str, Any]:
    """Return the containing Git revision and dirty state when available."""

    location = Path(cwd or Path.cwd()).expanduser().resolve()
    root = _git_output(["rev-parse", "--show-toplevel"], cwd=location)
    if not root:
        return {"available": False, "revision": None, "dirty": None}

    repository = Path(root)
    revision = _git_output(["rev-parse", "HEAD"], cwd=repository)
    status = _git_output(
        ["status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
    )
    return {
        "available": revision is not None,
        "revision": revision,
        "dirty": None if status is None else bool(status),
    }


def runtime_provenance(
    *,
    operation: str,
    parameters: Mapping[str, Any] | None = None,
    cwd: Path | str | None = None,
) -> Mapping[str, Any]:
    """Build JSON-safe provenance for a generated NaturalLab report."""

    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")

    return {
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "operation": operation.strip(),
        "parameters": _safe_value(parameters or {}),
        "software": {
            "naturallab": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": dict(git_provenance(cwd=cwd)),
    }


__all__ = ["git_provenance", "runtime_provenance"]
