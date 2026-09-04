"""Tests for read-only study manifest CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml  # type: ignore[import-untyped]

from naturallab.cli import main
from naturallab.workflows import (
    RunState,
    StepRunState,
    StepStatus,
    load_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _manifest_data() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": "example-study",
        "session_id": "session-007",
        "views": {
            "room-left": {
                "media": "raw/left.mp4",
                "calibration": {
                    "intrinsics": "calibration/left.intrinsics.json",
                },
            },
            "wearable": {
                "media": "raw/wearable.mp4",
                "gaze_input": "raw/gaze.csv",
            },
        },
        "steps": {
            "tracking": {
                "selected": True,
                "depends_on": [],
                "inputs": [],
                "outputs": ["derived/tracks.json"],
                "config": {"preset": "qwen36_27b_quality"},
            },
            "align": {
                "selected": True,
                "depends_on": ["tracking"],
                "inputs": [],
                "outputs": ["derived/aligned.json"],
                "config": {"tolerance_seconds": 0.05},
            },
            "fusion": {
                "selected": False,
                "depends_on": ["tracking"],
                "inputs": [],
                "outputs": ["derived/fused.json"],
                "config": {},
            },
        },
    }


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "session.yaml"
    path.write_text(
        yaml.safe_dump(_manifest_data(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_checked_in_example_manifest_matches_current_schema() -> None:
    manifest = load_manifest(
        REPOSITORY_ROOT / "examples" / "study_manifest.yaml"
    )

    assert manifest.schema_version == "1.0"
    assert tuple(manifest.views) == (
        "room_left",
        "room_right",
        "wearable",
    )
    assert manifest.steps["shared_room_fusion"].selected is False
    assert manifest.steps["track_people"].config["model_id"] == (
        "Qwen/Qwen3.6-27B"
    )


def test_study_validate_json_reports_validated_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(tmp_path)

    exit_code = main(["study", "validate", str(manifest_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["valid"] is True
    assert payload["study_id"] == "example-study"
    assert payload["views"] == ["room-left", "wearable"]
    assert payload["selected_steps"] == ["tracking", "align"]
    assert payload["skipped_steps"] == ["fusion"]
    assert len(payload["manifest_fingerprint"]) == 64
    assert payload["manifest"] == "session.yaml"
    assert str(tmp_path) not in captured.out


def test_study_plan_uses_dependency_order_without_writing_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(tmp_path)
    state_path = manifest_path.with_suffix(".run-state.json")

    exit_code = main(["study", "plan", str(manifest_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [step["name"] for step in payload["selected_steps"]] == [
        "tracking",
        "align",
    ]
    assert payload["selected_steps"][1]["depends_on"] == ["tracking"]
    assert payload["state_path"] == "session.run-state.json"
    assert payload["manifest"] == "session.yaml"
    assert str(tmp_path) not in json.dumps(payload)
    assert not state_path.exists()


def test_study_status_without_state_shows_initial_status_and_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(tmp_path)
    state_path = manifest_path.with_suffix(".run-state.json")

    exit_code = main(["study", "status", str(manifest_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    statuses = {step["name"]: step["status"] for step in payload["steps"]}
    assert exit_code == 0
    assert payload["state_exists"] is False
    assert payload["manifest_fingerprint_matches"] is None
    assert statuses == {
        "tracking": "pending",
        "align": "pending",
        "fusion": "skipped",
    }
    assert payload["state_path"] == "session.run-state.json"
    assert str(tmp_path) not in json.dumps(payload)
    assert not state_path.exists()


def test_study_status_reads_explicit_state_without_modifying_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    state_path = tmp_path / "state" / "custom.json"
    state = RunState(
        study_id=manifest.study_id,
        session_id=manifest.session_id,
        manifest_fingerprint=manifest.fingerprint,
        steps={
            "tracking": StepRunState(
                status=StepStatus.COMPLETED,
                attempts=1,
            ),
            "align": StepRunState(
                status=StepStatus.FAILED,
                attempts=2,
                error="missing gaze timestamps",
            ),
            "fusion": StepRunState(status=StepStatus.SKIPPED),
            "removed-step": StepRunState(status=StepStatus.COMPLETED),
        },
    )
    state.write_atomic(state_path)
    before = state_path.read_bytes()

    exit_code = main(
        [
            "study",
            "status",
            str(manifest_path),
            "--state",
            str(state_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    statuses = {step["name"]: step["status"] for step in payload["steps"]}
    assert exit_code == 0
    assert payload["state_exists"] is True
    assert payload["manifest_fingerprint_matches"] is True
    assert statuses["tracking"] == "completed"
    assert statuses["align"] == "failed"
    assert payload["steps"][1]["error"] == {
        "type": "Error",
        "message": "missing gaze timestamps",
    }
    assert payload["manifest"] == "session.yaml"
    assert payload["state_path"] == "state/custom.json"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["unexpected_state_steps"] == ["removed-step"]
    assert state_path.read_bytes() == before


def test_study_validation_error_is_concise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: '1.0'\n", encoding="utf-8")

    exit_code = main(["study", "validate", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "manifest.study_id is required" in captured.err
    assert "Traceback" not in captured.err


def test_study_plan_redacts_external_absolute_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_dir = tmp_path / "manifest-directory"
    manifest_dir.mkdir()
    path = manifest_dir / "session.yaml"
    data = _manifest_data()
    private_input = tmp_path / "researcher-private" / "tracking.yaml"
    windows_input = r"C:\Users\Researcher Name\private-tracking.yaml"
    data["steps"]["tracking"]["inputs"] = [
        str(private_input),
        windows_input,
    ]
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(["study", "plan", str(path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    identities = payload["selected_steps"][0]["inputs"]
    assert exit_code == 0
    assert captured.err == ""
    assert identities[0].startswith("tracking.yaml [path-id:")
    assert identities[1].startswith("private-tracking.yaml [path-id:")
    assert str(tmp_path) not in captured.out
    assert "researcher-private" not in captured.out
    assert windows_input not in captured.out
    assert "\\Users\\" not in captured.out


def test_study_load_error_redacts_requested_absolute_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "Private Researcher Name" / "missing.yaml"

    exit_code = main(["study", "validate", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "ManifestError" in captured.err
    assert "<path:missing.yaml>" in captured.err
    assert str(tmp_path) not in captured.err
    assert "Private Researcher Name" not in captured.err
