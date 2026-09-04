"""Privacy contracts for portable workflow metadata and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from naturallab.workflows.privacy import (
    ERROR_MESSAGE_MAX_LENGTH,
    portable_path_identity,
    sanitize_error_message,
    sanitize_error_type,
)


@pytest.mark.parametrize(
    ("message", "private_values"),
    [
        (
            "failed /Users/researcher/private/input.csv token=hunter2",
            ("/Users/", "researcher", "hunter2"),
        ),
        (
            r"failed 'C:\Users\Researcher Name\input.csv' "
            "Authorization: Bearer top-secret-token",
            (r"C:\Users", "Researcher Name", "top-secret-token"),
        ),
        (
            "connection failed at "
            "https://user:password@login.example.org:8443/private",
            ("user", "password", "login.example.org"),
        ),
        (
            "ssh failed user@login.internal:/private/workspace",
            ("user", "login.internal", "/private/workspace"),
        ),
    ],
)
def test_sanitized_errors_remove_sensitive_values(
    message: str,
    private_values: tuple[str, ...],
) -> None:
    sanitized = sanitize_error_message(message)

    assert sanitized
    assert all(value not in sanitized for value in private_values)


def test_sanitized_errors_are_single_line_and_bounded() -> None:
    sanitized = sanitize_error_message("failure\n" + "detail " * 1000)

    assert "\n" not in sanitized
    assert len(sanitized) <= ERROR_MESSAGE_MAX_LENGTH
    assert sanitized.endswith("... [truncated]")


def test_error_type_accepts_exception_names_but_not_path_like_values() -> None:
    assert sanitize_error_type("RuntimeError") == "RuntimeError"
    assert sanitize_error_type("/Users/researcher/PrivateError") == "Error"
    assert sanitize_error_type(r"C:\Users\Researcher\PrivateError") == "Error"


def test_portable_path_identity_uses_relative_or_opaque_labels(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "study"
    study_root.mkdir()
    external = tmp_path / "researcher-private" / "config.yaml"

    assert portable_path_identity(
        study_root / "derived" / "tracks.json",
        base_dir=study_root,
    ) == "derived/tracks.json"
    assert portable_path_identity(
        external,
        base_dir=study_root,
    ).startswith("config.yaml [path-id:")
    assert portable_path_identity(
        r"C:\Users\Researcher Name\private-config.yaml",
        base_dir=study_root,
    ).startswith("private-config.yaml [path-id:")
