"""Command-line entry point for NaturalLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

from naturallab import __version__
from naturallab.doctor import PROFILE_NAMES, format_human, run_doctor
from naturallab.spatial_tracking.calibration.commands import (
    add_calibration_commands,
    run_calibration_command,
)
from naturallab.workflows import (
    ManifestError,
    RunState,
    RunStateError,
    StepStatus,
    StudyManifest,
    load_manifest,
)


def _add_study_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "manifest",
        help="Path to a schema-versioned study manifest (YAML or JSON).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a machine-readable JSON report.",
    )


def _default_state_path(manifest: StudyManifest) -> Path:
    if manifest.source_path is None:
        raise ManifestError("a file-backed manifest is required")
    return manifest.source_path.with_suffix(".run-state.json")


def _validation_report(manifest: StudyManifest) -> Dict[str, Any]:
    selected = [step.name for step in manifest.selected_steps()]
    return {
        "valid": True,
        "manifest": str(manifest.source_path),
        "schema_version": manifest.schema_version,
        "study_id": manifest.study_id,
        "session_id": manifest.session_id,
        "manifest_fingerprint": manifest.fingerprint,
        "view_count": len(manifest.views),
        "views": list(manifest.views),
        "selected_steps": selected,
        "skipped_steps": [
            name for name, step in manifest.steps.items() if not step.selected
        ],
    }


def _plan_report(manifest: StudyManifest) -> Dict[str, Any]:
    return {
        "manifest": str(manifest.source_path),
        "study_id": manifest.study_id,
        "session_id": manifest.session_id,
        "state_path": str(_default_state_path(manifest)),
        "selected_steps": [
            {
                "name": step.name,
                "depends_on": list(step.depends_on),
                "inputs": list(step.inputs),
                "outputs": list(step.outputs),
                "config_fingerprint": (
                    manifest.step_config_fingerprint(step.name)
                ),
            }
            for step in manifest.selected_steps()
        ],
        "skipped_steps": [
            name for name, step in manifest.steps.items() if not step.selected
        ],
    }


def _status_report(
    manifest: StudyManifest,
    state_path: Path,
) -> Dict[str, Any]:
    state_exists = state_path.exists()
    state = RunState.load(state_path) if state_exists else None
    if state is not None and (
        state.study_id != manifest.study_id
        or state.session_id != manifest.session_id
    ):
        raise RunStateError(
            "run state belongs to "
            f"{state.study_id}/{state.session_id}, not "
            f"{manifest.study_id}/{manifest.session_id}"
        )

    steps: list[Dict[str, Any]] = []
    for name, step in manifest.steps.items():
        persisted = None if state is None else state.steps.get(name)
        default_status = (
            StepStatus.PENDING if step.selected else StepStatus.SKIPPED
        )
        steps.append(
            {
                "name": name,
                "selected": step.selected,
                "status": (
                    persisted.status.value
                    if persisted is not None
                    else default_status.value
                ),
                "attempts": 0 if persisted is None else persisted.attempts,
                "started_at": (
                    None if persisted is None else persisted.started_at
                ),
                "completed_at": (
                    None if persisted is None else persisted.completed_at
                ),
                "error": None if persisted is None else persisted.error,
            }
        )

    counts = {status.value: 0 for status in StepStatus}
    for step_status in steps:
        status_name = str(step_status["status"])
        counts[status_name] += 1

    return {
        "manifest": str(manifest.source_path),
        "study_id": manifest.study_id,
        "session_id": manifest.session_id,
        "state_path": str(state_path),
        "state_exists": state_exists,
        "manifest_fingerprint_matches": (
            None
            if state is None
            else state.manifest_fingerprint == manifest.fingerprint
        ),
        "state_updated_at": None if state is None else state.updated_at,
        "status_counts": counts,
        "steps": steps,
        "unexpected_state_steps": (
            []
            if state is None
            else sorted(set(state.steps) - set(manifest.steps))
        ),
    }


def _format_validation(report: Dict[str, Any]) -> str:
    selected = ", ".join(report["selected_steps"]) or "none"
    skipped = ", ".join(report["skipped_steps"]) or "none"
    return "\n".join(
        (
            "Valid NaturalLab study manifest",
            f"Study/session: {report['study_id']}/{report['session_id']}",
            (
                f"Views ({report['view_count']}): "
                + ", ".join(report["views"])
            ),
            f"Selected steps: {selected}",
            f"Skipped steps: {skipped}",
            f"Manifest SHA-256: {report['manifest_fingerprint']}",
        )
    )


def _format_plan(report: Dict[str, Any]) -> str:
    lines = [
        f"Study plan: {report['study_id']}/{report['session_id']}",
        "Selected steps (dependency order):",
    ]
    for index, step in enumerate(report["selected_steps"], start=1):
        dependencies = ", ".join(step["depends_on"]) or "none"
        inputs = ", ".join(step["inputs"]) or "none"
        outputs = ", ".join(step["outputs"]) or "none"
        lines.extend(
            (
                f"  {index}. {step['name']} (depends on: {dependencies})",
                f"     inputs: {inputs}",
                f"     outputs: {outputs}",
            )
        )
    if not report["selected_steps"]:
        lines.append("  none")
    skipped = ", ".join(report["skipped_steps"]) or "none"
    lines.extend(
        (
            f"Skipped steps: {skipped}",
            f"Run-state path: {report['state_path']}",
        )
    )
    return "\n".join(lines)


def _format_status(report: Dict[str, Any]) -> str:
    if report["state_exists"]:
        fingerprint_state = (
            "matches"
            if report["manifest_fingerprint_matches"]
            else "differs"
        )
        state_line = (
            f"Run state: {report['state_path']} "
            f"(manifest fingerprint {fingerprint_state})"
        )
    else:
        state_line = f"Run state: not created ({report['state_path']})"
    lines = [
        f"Study status: {report['study_id']}/{report['session_id']}",
        state_line,
    ]
    for step in report["steps"]:
        attempts = (
            f", attempts={step['attempts']}" if step["attempts"] else ""
        )
        lines.append(f"  {step['name']}: {step['status']}{attempts}")
        if step["error"]:
            lines.append(f"    error: {step['error']}")
    if report["unexpected_state_steps"]:
        lines.append(
            "State-only steps: "
            + ", ".join(report["unexpected_state_steps"])
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="naturallab",
        description="NaturalLab research video and sensor analysis tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local environment without changing it.",
        description=(
            "Run read-only checks for NaturalLab and an optional workflow profile. "
            "This command never downloads models or contacts configured services."
        ),
    )
    doctor_parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="core",
        help="Capability group to inspect (default: core).",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a machine-readable JSON report.",
    )

    subparsers.add_parser(
        "record",
        help="Open the guided recording window.",
        description=(
            "Open the NaturalLab recording window for configuring cameras and "
            "optional sensors, starting LSL streams, and launching LabRecorder."
        ),
    )

    study_parser = subparsers.add_parser(
        "study",
        help="Inspect a study/session manifest without running its steps.",
        description=(
            "Validate and inspect study manifests and persisted run state. "
            "These commands do not execute workflow steps or write files."
        ),
    )
    study_subparsers = study_parser.add_subparsers(dest="study_command")

    validate_parser = study_subparsers.add_parser(
        "validate",
        help="Validate manifest structure and dependencies.",
    )
    _add_study_output_arguments(validate_parser)

    plan_parser = study_subparsers.add_parser(
        "plan",
        help="Show selected steps in dependency order.",
    )
    _add_study_output_arguments(plan_parser)

    status_parser = study_subparsers.add_parser(
        "status",
        help="Read persisted status, or show the initial state.",
    )
    _add_study_output_arguments(status_parser)
    status_parser.add_argument(
        "--state",
        help=(
            "Run-state JSON path. Defaults to MANIFEST with the suffix "
            "'.run-state.json'."
        ),
    )

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="Run click-free camera, floor, and shared-room calibration.",
        description=(
            "Automatically detect a chessboard and produce schema-versioned "
            "intrinsic, floor-plane, multi-camera registration, and "
            "verification outputs."
        ),
    )
    add_calibration_commands(calibration_parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "doctor":
        doctor_report = run_doctor(profile=args.profile)
        if args.json_output:
            print(
                json.dumps(
                    doctor_report.to_dict(),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(format_human(doctor_report))
        return doctor_report.exit_code

    if args.command == "record":
        from naturallab.acquisition.recording_gui import main as run_recorder

        result = run_recorder()
        return 0 if result is None else int(result)

    if args.command == "study":
        if args.study_command is None:
            parser.print_help()
            return 0
        try:
            manifest = load_manifest(args.manifest)
            if args.study_command == "validate":
                study_report = _validation_report(manifest)
                human_report = _format_validation(study_report)
            elif args.study_command == "plan":
                study_report = _plan_report(manifest)
                human_report = _format_plan(study_report)
            elif args.study_command == "status":
                state_path = (
                    Path(args.state).expanduser().resolve()
                    if args.state
                    else _default_state_path(manifest)
                )
                study_report = _status_report(manifest, state_path)
                human_report = _format_status(study_report)
            else:
                parser.error(
                    f"Unknown study command: {args.study_command}"
                )
                return 2
        except (ManifestError, RunStateError) as exc:
            print(f"naturallab: error: {exc}", file=sys.stderr)
            return 2

        if args.json_output:
            print(json.dumps(study_report, indent=2, sort_keys=True))
        else:
            print(human_report)
        return 0

    if args.command == "calibrate":
        if args.calibration_command is None:
            parser.print_help()
            return 0
        return run_calibration_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
