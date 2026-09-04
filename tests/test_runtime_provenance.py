from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from naturallab.provenance import git_provenance, runtime_provenance


def test_runtime_provenance_is_json_safe_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    provenance = runtime_provenance(
        operation="calibrate.verify",
        parameters={
            "video": tmp_path / "verification.mp4",
            "nested": {"api_key": "do-not-store", "count": 3},
            "password": "also-do-not-store",
        },
        cwd=tmp_path,
    )

    rendered = json.dumps(provenance, sort_keys=True)
    assert "do-not-store" not in rendered
    assert "also-do-not-store" not in rendered
    assert provenance["operation"] == "calibrate.verify"
    assert provenance["parameters"]["password"] == "<redacted>"
    assert provenance["parameters"]["nested"]["api_key"] == "<redacted>"
    assert provenance["software"]["naturallab"]
    assert provenance["software"]["python"]
    assert provenance["software"]["numpy"]
    assert provenance["software"]["opencv"]
    assert provenance["generated_at_utc"].endswith("Z")


def test_git_provenance_degrades_cleanly_outside_a_repository(
    tmp_path: Path,
) -> None:
    provenance = git_provenance(cwd=tmp_path)

    assert provenance == {
        "available": False,
        "revision": None,
        "dirty": None,
    }


def test_runtime_provenance_is_strict_json_with_non_finite_parameters(
    tmp_path: Path,
) -> None:
    provenance = runtime_provenance(
        operation="calibrate.verify",
        parameters={"residual": math.nan, "limit": math.inf},
        cwd=tmp_path,
    )

    rendered = json.dumps(provenance, allow_nan=False)
    assert '"residual": "nan"' in rendered
    assert '"limit": "inf"' in rendered


@pytest.mark.parametrize("operation", ["", "   ", None])
def test_runtime_provenance_requires_named_operation(
    tmp_path: Path,
    operation: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        runtime_provenance(operation=operation, cwd=tmp_path)  # type: ignore[arg-type]
