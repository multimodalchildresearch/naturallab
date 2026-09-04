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
    assert payload["state_path"] == str(state_path)
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
