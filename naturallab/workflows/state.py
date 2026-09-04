"""Atomic run state and content fingerprints for resumable workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


RUN_STATE_SCHEMA_VERSION = "1.0"


class RunStateError(RuntimeError):
    """Raised when persisted run state is malformed or incompatible."""


class StepStatus(str, Enum):
    """Persisted lifecycle states for one workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fingerprint_path(path: Path | str) -> str:
    """Return a SHA-256 content fingerprint for a file or directory.

    Directory fingerprints include sorted relative paths, entry kinds, and file
    content.  This makes changes to names, nesting, empty directories, symlinks,
    or file contents visible to the resume check.
    """

    target = Path(path)
    active_directories: set[tuple[int, int]] = set()

    def canonical_digest(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def file_content_digest(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def node_digest(node: Path) -> str:
        if node.is_symlink():
            link_text = os.readlink(node)
            try:
                resolved = node.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise FileNotFoundError(
                    f"declared symlink cannot be resolved: {node}"
                ) from exc
            if not resolved.is_file() and not resolved.is_dir():
                raise FileNotFoundError(
                    "declared symlink target is not a file or directory: "
                    f"{node}"
                )
            target_identity = None
            if resolved.is_dir():
                stat_result = resolved.stat()
                target_identity = (
                    stat_result.st_dev,
                    stat_result.st_ino,
                )
            target_digest = (
                canonical_digest(["directory-cycle"])
                if target_identity in active_directories
                else node_digest(resolved)
            )
            return canonical_digest(
                ["symlink", link_text, target_digest]
            )

        if not node.exists():
            raise FileNotFoundError(
                f"declared path does not exist: {node}"
            )
        if node.is_file():
            return canonical_digest(
                ["file", file_content_digest(node)]
            )
        if not node.is_dir():
            raise FileNotFoundError(
                "declared path is not a regular file or directory: "
                f"{node}"
            )

        stat_result = node.stat()
        identity = (stat_result.st_dev, stat_result.st_ino)
        if identity in active_directories:
            return canonical_digest(["directory-cycle"])
        active_directories.add(identity)
        try:
            children = [
                [child.name, node_digest(child)]
                for child in sorted(
                    node.iterdir(),
                    key=lambda child: child.name,
                )
            ]
        finally:
            active_directories.remove(identity)
        return canonical_digest(["directory", children])

    return node_digest(target)


@dataclass
class StepRunState:
    """Mutable persisted state for one step."""

    status: StepStatus
    attempts: int = 0
    config_fingerprint: Optional[str] = None
    input_fingerprints: Dict[str, str] = field(default_factory=dict)
    output_fingerprints: Dict[str, str] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "attempts": self.attempts,
            "config_fingerprint": self.config_fingerprint,
            "input_fingerprints": dict(sorted(self.input_fingerprints.items())),
            "output_fingerprints": dict(
                sorted(self.output_fingerprints.items())
            ),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        step_name: str,
    ) -> "StepRunState":
        if not isinstance(data, Mapping):
            raise RunStateError(f"state for step {step_name!r} must be a mapping")
        try:
            status = StepStatus(data["status"])
            attempts = data.get("attempts", 0)
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
            ):
                raise ValueError("attempts must be a non-negative integer")
            input_fingerprints = _fingerprint_mapping(
                data.get("input_fingerprints", {}),
                f"steps.{step_name}.input_fingerprints",
            )
            output_fingerprints = _fingerprint_mapping(
                data.get("output_fingerprints", {}),
                f"steps.{step_name}.output_fingerprints",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunStateError(
                f"invalid state for step {step_name!r}: {exc}"
            ) from exc

        return cls(
            status=status,
            attempts=attempts,
            config_fingerprint=_optional_string(
                data.get("config_fingerprint"),
                f"steps.{step_name}.config_fingerprint",
            ),
            input_fingerprints=input_fingerprints,
            output_fingerprints=output_fingerprints,
            started_at=_optional_string(
                data.get("started_at"),
                f"steps.{step_name}.started_at",
            ),
            completed_at=_optional_string(
                data.get("completed_at"),
                f"steps.{step_name}.completed_at",
            ),
            error=_optional_string(
                data.get("error"),
                f"steps.{step_name}.error",
            ),
        )


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunStateError(f"{field_name} must be a string or null")
    return value


def _fingerprint_mapping(value: Any, field_name: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise RunStateError(f"{field_name} must be a mapping")
    fingerprints: Dict[str, str] = {}
    for key, fingerprint in value.items():
        if not isinstance(key, str) or not isinstance(fingerprint, str):
            raise RunStateError(
                f"{field_name} must map string names to SHA-256 strings"
            )
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise RunStateError(
                f"{field_name}.{key} is not a lowercase SHA-256 fingerprint"
            )
        fingerprints[key] = fingerprint
    return fingerprints


@dataclass
class RunState:
    """Schema-versioned state persisted beside workflow results."""

    study_id: str
    session_id: str
    manifest_fingerprint: str
    steps: Dict[str, StepRunState]
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = field(
        default=RUN_STATE_SCHEMA_VERSION,
        init=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "session_id": self.session_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "updated_at": self.updated_at,
            "steps": {
                name: self.steps[name].to_dict()
                for name in sorted(self.steps)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunState":
        if not isinstance(data, Mapping):
            raise RunStateError("run state must be a mapping")
        version = data.get("schema_version")
        if str(version) != RUN_STATE_SCHEMA_VERSION:
            raise RunStateError(
                "Unsupported run-state schema_version "
                f"{version!r}; expected {RUN_STATE_SCHEMA_VERSION!r}"
            )
        try:
            study_id = data["study_id"]
            session_id = data["session_id"]
            manifest_fingerprint = data["manifest_fingerprint"]
            step_values = data["steps"]
        except KeyError as exc:
            raise RunStateError(f"run state is missing field {exc.args[0]!r}") from exc
        if not all(
            isinstance(value, str)
            for value in (study_id, session_id, manifest_fingerprint)
        ):
            raise RunStateError(
                "run-state IDs and manifest_fingerprint must be strings"
            )
        if not isinstance(step_values, Mapping):
            raise RunStateError("run-state steps must be a mapping")
        steps = {
            name: StepRunState.from_dict(value, step_name=name)
            for name, value in step_values.items()
            if isinstance(name, str)
        }
        if len(steps) != len(step_values):
            raise RunStateError("run-state step names must be strings")
        return cls(
            study_id=study_id,
            session_id=session_id,
            manifest_fingerprint=manifest_fingerprint,
            steps=steps,
            updated_at=_optional_string(
                data.get("updated_at"),
                "updated_at",
            )
            or utc_now(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "RunState":
        state_path = Path(path)
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RunStateError(
                f"could not read run state {state_path}: {exc}"
            ) from exc
        return cls.from_dict(data)

    def write_atomic(self, path: Path | str) -> None:
        """Durably replace a JSON run-state file in one atomic rename."""

        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        payload = (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=state_path.parent,
                prefix=f".{state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, state_path)
            temporary_name = None
            try:
                directory_fd = os.open(state_path.parent, os.O_RDONLY)
            except OSError:
                return
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
