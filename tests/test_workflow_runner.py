"""Tests for conservative, model-independent workflow resumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from naturallab.workflows import (
    RunState,
    RunStateError,
    StepExecutionContext,
    StepStatus,
    StudyManifest,
    WorkflowExecutionError,
    WorkflowRunner,
    fingerprint_path,
)


def workflow_data(
    *,
    detector: str = "qwen",
    site: str = "lab-a",
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": "study-01",
        "session_id": "session-01",
        "views": {
            "main": {
                "media": "raw/video.bin",
                "calibration": {
                    "intrinsics": "calibration/intrinsics.json",
                },
                "object_input": "raw/object-labels.json",
            },
        },
        "steps": {
            "track": {
                "selected": True,
                "depends_on": [],
                "inputs": ["config/track.json"],
                "outputs": ["derived/tracks.json"],
                "config": {"detector": detector},
            },
            "summarize": {
                "selected": True,
                "depends_on": ["track"],
                "inputs": [],
                "outputs": ["derived/summary.json"],
                "config": {},
            },
            "optional": {
                "selected": False,
                "depends_on": [],
                "inputs": [],
                "outputs": [],
                "config": {},
            },
        },
        "metadata": {"site": site},
    }


def create_inputs(tmp_path: Path) -> None:
    for relative, content in (
        ("raw/video.bin", b"frame-data-v1"),
        ("calibration/intrinsics.json", b'{"camera":"main"}'),
        ("raw/object-labels.json", b'{"objects":["toy"]}'),
        ("config/track.json", b'{"threshold":0.5}'),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def make_runner(
    tmp_path: Path,
    *,
    detector: str = "qwen",
    site: str = "lab-a",
) -> WorkflowRunner:
    manifest = StudyManifest.from_dict(
        workflow_data(detector=detector, site=site),
        base_dir=tmp_path,
    )
    return WorkflowRunner(manifest, tmp_path / "run-state.json")


def content_executor(
    calls: list[str],
) -> Dict[str, Any]:
    def track(context: StepExecutionContext) -> None:
        calls.append(context.step.name)
        context.output_paths[0].parent.mkdir(parents=True, exist_ok=True)
        source = context.manifest.resolve_path("raw/video.bin").read_bytes()
        context.output_paths[0].write_bytes(b"tracks:" + source)

    def summarize(context: StepExecutionContext) -> None:
        calls.append(context.step.name)
        tracks = context.manifest.resolve_path(
            "derived/tracks.json"
        ).read_bytes()
        context.output_paths[0].write_bytes(b"summary:" + tracks)

    return {"track": track, "summarize": summarize}


def test_runner_executes_in_dependency_order_and_resumes_exact_outputs(
    tmp_path: Path,
) -> None:
    create_inputs(tmp_path)
    runner = make_runner(tmp_path)
    calls: list[str] = []

    first = runner.run(content_executor(calls))

    assert calls == ["track", "summarize"]
    assert first.executed_steps == ("track", "summarize")
    assert first.reused_steps == ()
    assert first.skipped_steps == ("optional",)
    assert first.state.steps["track"].status is StepStatus.COMPLETED
    assert first.state.steps["summarize"].status is StepStatus.COMPLETED
    assert first.state.steps["optional"].status is StepStatus.SKIPPED
    assert first.state.steps["track"].attempts == 1

    def should_not_run(context: StepExecutionContext) -> None:
        raise AssertionError(f"unexpected execution of {context.step.name}")

    second = runner.run(should_not_run)

    assert second.executed_steps == ()
    assert second.reused_steps == ("track", "summarize")
    assert second.state.steps["track"].attempts == 1
    assert not list(tmp_path.glob(".run-state.json.*.tmp"))
    assert json.loads((tmp_path / "run-state.json").read_text())[
        "schema_version"
    ] == "1.0"


@pytest.mark.parametrize("change", ["input", "config", "output"])
def test_runner_reruns_when_input_config_or_output_fingerprint_changes(
    tmp_path: Path,
    change: str,
) -> None:
    create_inputs(tmp_path)
    calls: list[str] = []
    runner = make_runner(tmp_path)
    runner.run(content_executor(calls))
    calls.clear()

    if change == "input":
        (tmp_path / "raw/video.bin").write_bytes(b"frame-data-v2")
    elif change == "config":
        runner = make_runner(tmp_path, detector="yolo")
    else:
        (tmp_path / "derived/tracks.json").write_bytes(b"tampered")

    result = runner.run(content_executor(calls))

    assert "track" in result.executed_steps
    assert calls[0] == "track"
    assert result.state.steps["track"].attempts == 2
    assert result.state.steps["track"].status is StepStatus.COMPLETED


def test_runner_reruns_when_manifest_metadata_changes(
    tmp_path: Path,
) -> None:
    create_inputs(tmp_path)
    calls: list[str] = []
    make_runner(tmp_path, site="lab-a").run(content_executor(calls))
    calls.clear()

    result = make_runner(tmp_path, site="lab-b").run(
        content_executor(calls)
    )

    assert result.executed_steps == ("track", "summarize")
    assert calls == ["track", "summarize"]


def test_executor_return_without_outputs_is_failed_not_completed(
    tmp_path: Path,
) -> None:
    create_inputs(tmp_path)
    runner = make_runner(tmp_path)

    with pytest.raises(
        WorkflowExecutionError,
        match="declared path does not exist",
    ):
        runner.run(lambda context: None)

    failed = RunState.load(tmp_path / "run-state.json")
    assert failed.steps["track"].status is StepStatus.FAILED
    assert failed.steps["track"].output_fingerprints == {}
    assert failed.steps["track"].completed_at is None
    assert failed.steps["track"].attempts == 1

    calls: list[str] = []
    recovered = runner.run(content_executor(calls))
    assert recovered.state.steps["track"].status is StepStatus.COMPLETED
    assert recovered.state.steps["track"].attempts == 2


def test_executor_exception_is_persisted_as_failed_and_can_resume(
    tmp_path: Path,
) -> None:
    create_inputs(tmp_path)
    runner = make_runner(tmp_path)

    def fail(context: StepExecutionContext) -> None:
        raise RuntimeError("synthetic model failure")

    with pytest.raises(WorkflowExecutionError, match="synthetic model failure"):
        runner.run({"track": fail})

    state = RunState.load(tmp_path / "run-state.json")
    assert state.steps["track"].status is StepStatus.FAILED
    assert state.steps["track"].error == "synthetic model failure"
    assert state.steps["summarize"].status is StepStatus.PENDING


def test_stale_completion_is_invalidated_when_executor_is_missing(
    tmp_path: Path,
) -> None:
    create_inputs(tmp_path)
    runner = make_runner(tmp_path)
    runner.run(content_executor([]))
    (tmp_path / "derived/tracks.json").write_bytes(b"tampered")

    with pytest.raises(WorkflowExecutionError, match="no executor"):
        runner.run({"summarize": lambda context: None})

    state = RunState.load(tmp_path / "run-state.json")
    assert state.steps["track"].status is StepStatus.FAILED
    assert state.steps["track"].output_fingerprints == {}
    assert state.steps["track"].completed_at is None


def test_missing_declared_input_fails_before_executor(tmp_path: Path) -> None:
    create_inputs(tmp_path)
    (tmp_path / "raw/object-labels.json").unlink()
    runner = make_runner(tmp_path)
    called = False

    def executor(context: StepExecutionContext) -> None:
        nonlocal called
        called = True

    with pytest.raises(WorkflowExecutionError, match="fingerprint inputs"):
        runner.run(executor)

    assert not called
    state = RunState.load(tmp_path / "run-state.json")
    assert state.steps["track"].status is StepStatus.FAILED


def test_run_state_from_another_session_is_never_reused(tmp_path: Path) -> None:
    create_inputs(tmp_path)
    runner = make_runner(tmp_path)
    state = runner._initial_state()
    state.session_id = "different-session"
    state.write_atomic(tmp_path / "run-state.json")

    with pytest.raises(RunStateError, match="different-session"):
        runner.run(lambda context: None)


@pytest.mark.parametrize(
    "state_path",
    [
        "derived/tracks.json",
        "derived/tracks.json/run-state.json",
        "raw/video.bin/run-state.json",
        "session.yaml",
    ],
)
def test_runner_rejects_state_path_overlapping_declared_artifacts(
    tmp_path: Path,
    state_path: str,
) -> None:
    manifest = StudyManifest.from_dict(
        workflow_data(),
        base_dir=tmp_path,
        source_path=tmp_path / "session.yaml",
    )

    with pytest.raises(ValueError, match="state_path.*overlaps declared"):
        WorkflowRunner(manifest, tmp_path / state_path)


def test_file_and_directory_fingerprints_track_content_and_names(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    first = directory / "a.txt"
    first.write_text("same", encoding="utf-8")
    initial = fingerprint_path(directory)

    first.rename(directory / "b.txt")
    renamed = fingerprint_path(directory)
    assert renamed != initial

    (directory / "b.txt").write_text("changed", encoding="utf-8")
    assert fingerprint_path(directory) != renamed
    assert fingerprint_path(directory / "b.txt") == fingerprint_path(
        directory / "b.txt"
    )


def test_directory_fingerprint_frames_file_entries_unambiguously(
    tmp_path: Path,
) -> None:
    first_tree = tmp_path / "first"
    second_tree = tmp_path / "second"
    first_tree.mkdir()
    second_tree.mkdir()
    (first_tree / "a").write_bytes(b"F\0b\0x")
    (second_tree / "a").write_bytes(b"")
    (second_tree / "b").write_bytes(b"x")

    assert fingerprint_path(first_tree) != fingerprint_path(second_tree)


def test_directory_fingerprint_handles_symlink_to_ancestor(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "dataset"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    payload = directory / "payload.bin"
    payload.write_bytes(b"v1")
    try:
        (nested / "ancestor").symlink_to(directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    initial = fingerprint_path(directory)
    assert len(initial) == 64

    payload.write_bytes(b"v2")
    assert fingerprint_path(directory) != initial
