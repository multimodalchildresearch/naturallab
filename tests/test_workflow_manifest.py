"""Tests for researcher-facing study/session manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from naturallab.workflows import ManifestError, StudyManifest, load_manifest


def valid_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": "example-study",
        "session_id": "session-001",
        "views": {
            "ceiling": {
                "media": "raw/ceiling.mp4",
                "calibration": {
                    "intrinsics": "calibration/ceiling.intrinsics.json",
                    "floor_plane": "calibration/ceiling.floor.json",
                    "registration": "calibration/ceiling.room.json",
                },
                "role_input": "annotations/ceiling-roles.json",
                "object_input": "annotations/ceiling-objects.json",
                "metadata": {"camera_serial": "CAM-01"},
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
                "inputs": ["config/tracking.yaml"],
                "outputs": ["derived/tracks.json"],
                "config": {
                    "detector": "qwen",
                    "model": "Qwen/Qwen3.6-27B",
                },
            },
            "multimodal": {
                "selected": True,
                "depends_on": ["tracking"],
                "inputs": [],
                "outputs": ["derived/multimodal.json"],
                "config": {},
            },
            "diagnostics": {
                "selected": False,
                "depends_on": [],
                "inputs": [],
                "outputs": [],
            },
        },
        "metadata": {"site": "lab-a"},
    }


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_manifest_loads_yaml_and_json_with_arbitrary_named_views(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"session{suffix}"
    data = valid_manifest()
    if suffix == ".json":
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.schema_version == "1.0"
    assert manifest.study_id == "example-study"
    assert manifest.session_id == "session-001"
    assert tuple(manifest.views) == ("ceiling", "wearable")
    assert manifest.views["ceiling"].calibration is not None
    assert (
        manifest.views["ceiling"].calibration.registration
        == "calibration/ceiling.room.json"
    )
    assert manifest.views["wearable"].gaze_input == "raw/gaze.csv"
    assert [step.name for step in manifest.selected_steps()] == [
        "tracking",
        "multimodal",
    ]
    assert manifest.resolve_path("raw/ceiling.mp4") == (
        tmp_path / "raw/ceiling.mp4"
    )
    canonical = manifest.to_dict()
    assert canonical["views"] == data["views"]
    assert canonical["steps"]["tracking"] == data["steps"]["tracking"]
    # Optional mappings are serialized explicitly so the canonical form has a
    # stable fingerprint even when researchers omit them in YAML.
    assert canonical["steps"]["diagnostics"]["config"] == {}


def test_manifest_fingerprints_are_canonical_and_step_specific(
    tmp_path: Path,
) -> None:
    first_data = valid_manifest()
    first = StudyManifest.from_dict(first_data, base_dir=tmp_path)
    reordered = StudyManifest.from_dict(
        json.loads(json.dumps(first_data, sort_keys=True)),
        base_dir=tmp_path,
    )

    assert first.fingerprint == reordered.fingerprint
    assert first.step_config_fingerprint("tracking") == (
        reordered.step_config_fingerprint("tracking")
    )

    changed_data = valid_manifest()
    changed_data["steps"]["tracking"]["config"]["detector"] = "yolo"
    changed = StudyManifest.from_dict(changed_data, base_dir=tmp_path)

    assert changed.fingerprint != first.fingerprint
    assert changed.step_config_fingerprint("tracking") != (
        first.step_config_fingerprint("tracking")
    )
    # Unrelated step configuration is intentionally absent from this step's
    # fingerprint.
    assert changed.step_config_fingerprint("multimodal") == (
        first.step_config_fingerprint("multimodal")
    )

    changed_metadata = valid_manifest()
    changed_metadata["metadata"]["site"] = "lab-b"
    metadata_manifest = StudyManifest.from_dict(
        changed_metadata,
        base_dir=tmp_path,
    )
    assert metadata_manifest.step_config_fingerprint("tracking") != (
        first.step_config_fingerprint("tracking")
    )
    assert metadata_manifest.step_config_fingerprint("multimodal") != (
        first.step_config_fingerprint("multimodal")
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda data: data["steps"]["tracking"].pop("selected"),
            "selected must explicitly",
        ),
        (
            lambda data: data["steps"]["tracking"].pop("depends_on"),
            "depends_on must explicitly",
        ),
        (
            lambda data: data["steps"]["tracking"].update(outputs=[]),
            "declares no outputs",
        ),
        (
            lambda data: data["steps"]["tracking"].update(typo=True),
            "unknown field",
        ),
        (
            lambda data: data.update(schema_version="2.0"),
            "Unsupported manifest",
        ),
    ],
)
def test_manifest_rejects_ambiguous_or_unverifiable_steps(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    data = valid_manifest()
    mutate(data)

    with pytest.raises(ManifestError, match=match):
        StudyManifest.from_dict(data, base_dir=tmp_path)


def test_selected_dependency_must_exist_and_be_selected(tmp_path: Path) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["selected"] = False

    with pytest.raises(ManifestError, match="unselected step 'tracking'"):
        StudyManifest.from_dict(data, base_dir=tmp_path)

    data = valid_manifest()
    data["steps"]["multimodal"]["depends_on"] = ["missing"]
    with pytest.raises(ManifestError, match="unknown step 'missing'"):
        StudyManifest.from_dict(data, base_dir=tmp_path)


def test_manifest_rejects_dependency_cycles(tmp_path: Path) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["depends_on"] = ["multimodal"]

    with pytest.raises(ManifestError, match="contain a cycle"):
        StudyManifest.from_dict(data, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("first_output", "second_output"),
    [
        ("derived/shared", "derived/shared"),
        ("derived/shared", "derived/shared/result.json"),
    ],
)
def test_manifest_rejects_overlapping_selected_step_outputs(
    tmp_path: Path,
    first_output: str,
    second_output: str,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["outputs"] = [first_output]
    data["steps"]["multimodal"]["outputs"] = [second_output]

    with pytest.raises(ManifestError, match="overlapping outputs"):
        StudyManifest.from_dict(data, base_dir=tmp_path)


def test_manifest_rejects_selected_output_overlapping_view_input(
    tmp_path: Path,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["outputs"] = ["raw"]

    with pytest.raises(ManifestError, match="declared inputs"):
        StudyManifest.from_dict(data, base_dir=tmp_path)


def test_manifest_rejects_selected_output_overlapping_own_step_input(
    tmp_path: Path,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["outputs"] = ["config"]

    with pytest.raises(ManifestError, match="declared inputs"):
        StudyManifest.from_dict(data, base_dir=tmp_path)


def test_manifest_rejects_selected_output_overlapping_manifest(
    tmp_path: Path,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["outputs"] = ["session.yaml"]

    with pytest.raises(ManifestError, match="manifest"):
        StudyManifest.from_dict(
            data,
            base_dir=tmp_path,
            source_path=tmp_path / "session.yaml",
        )


def test_manifest_allows_one_step_to_fingerprint_nested_outputs(
    tmp_path: Path,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["outputs"] = [
        "derived/tracking",
        "derived/tracking/summary.json",
    ]

    manifest = StudyManifest.from_dict(data, base_dir=tmp_path)

    assert manifest.steps["tracking"].outputs == (
        "derived/tracking",
        "derived/tracking/summary.json",
    )


def test_manifest_rejects_non_json_config_and_bad_identifiers(
    tmp_path: Path,
) -> None:
    data = valid_manifest()
    data["steps"]["tracking"]["config"] = {"when": object()}
    with pytest.raises(ManifestError, match="JSON-compatible"):
        StudyManifest.from_dict(data, base_dir=tmp_path)

    data = valid_manifest()
    data["study_id"] = "has spaces"
    with pytest.raises(ManifestError, match="study_id"):
        StudyManifest.from_dict(data, base_dir=tmp_path)
