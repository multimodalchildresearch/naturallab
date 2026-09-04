"""Tests for the lightweight NaturalLab command and environment doctor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from naturallab import __version__
from naturallab import doctor
from naturallab.cli import main
from naturallab.spatial_tracking.pipeline import DEFAULT_REID_FILENAME
from naturallab.spatial_tracking.pipeline import reid as reid_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "naturallab.cli", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_version_flag() -> None:
    result = run_cli("--version")

    assert result.returncode == 0
    assert result.stdout.strip() == f"naturallab {__version__}"
    assert result.stderr == ""


def test_qwen_extra_declares_the_quality_tracker_dependencies() -> None:
    project_text = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    qwen_extra = project_text.split("qwen = [", 1)[1].split("]", 1)[0]

    assert '"filterpy>=1.4.5"' in qwen_extra
    assert '"scipy>=1.5.0"' in qwen_extra
    assert '"torch>=1.13.0"' in qwen_extra
    assert "torchreid" not in qwen_extra


def test_yolo_has_a_dedicated_strict_doctor_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available = {requirement.module for requirement in doctor.MODULES["core"]}
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available,
    )
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda distribution: "test",
    )

    report = doctor.run_doctor(profile="yolo", cwd=tmp_path)
    yolo_check = next(
        check for check in report.checks if check.details.get("profile") == "yolo"
    )

    assert yolo_check.status == "fail"
    assert yolo_check.details["required"] is True
    assert report.exit_code == 1


def test_doctor_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["doctor", "--profile", "core", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == payload["exit_code"]
    assert payload["application"] == "naturallab"
    assert payload["profile"] == "core"
    assert payload["version"] == __version__
    assert {"pass", "warning", "fail", "skip"} == set(payload["summary"])
    assert payload["checks"]


def test_doctor_does_not_expose_the_working_directory(tmp_path: Path) -> None:
    report = doctor.run_doctor(profile="core", cwd=tmp_path)

    assert str(tmp_path) not in json.dumps(report.to_dict())


def test_selected_profile_missing_modules_are_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available_core = {requirement.module for requirement in doctor.MODULES["core"]}
    available_core.add("torch")
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available_core,
    )
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution: "test")

    report = doctor.run_doctor(profile="acquisition", cwd=tmp_path)
    acquisition_checks = [
        check
        for check in report.checks
        if check.details.get("profile") == "acquisition"
    ]

    assert acquisition_checks
    assert all(check.status == "fail" for check in acquisition_checks)
    assert all(check.details["required"] is True for check in acquisition_checks)
    assert report.exit_code == 1


def test_unwritable_working_directory_is_a_core_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_access = doctor.os.access

    def fake_access(path: str, mode: int) -> bool:
        if Path(path) == tmp_path and mode == os.W_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(doctor.os, "access", fake_access)
    report = doctor.run_doctor(profile="core", cwd=tmp_path)

    cwd_check = next(
        check for check in report.checks if check.check_id == "working_directory"
    )
    assert cwd_check.status == "fail"
    assert report.exit_code == 1


def test_qwen_environment_values_are_never_exposed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint_secret = "https://user:secret@example.invalid/v1?token=hidden"
    api_secret = "super-secret-api-key"
    environment = {
        doctor.VLM_BASE_URL_ENV: endpoint_secret,
        doctor.VLM_API_KEY_ENV: api_secret,
        "NATURALLAB_REID_CACHE_DIR": str(tmp_path),
    }
    available_core = {requirement.module for requirement in doctor.MODULES["core"]}
    available_core.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available_core,
    )
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution: "test")
    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ=environment,
    )
    serialized = json.dumps(report.to_dict())
    service_check = next(
        check for check in report.checks if check.check_id == "vlm_environment"
    )

    assert endpoint_secret not in serialized
    assert api_secret not in serialized
    assert service_check.details["endpoint_configured"] is True
    assert service_check.details["credential_configured"] is True
    assert service_check.details["values_redacted"] is True


def test_qwen_without_endpoint_is_a_nonblocking_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available_core = {requirement.module for requirement in doctor.MODULES["core"]}
    available_core.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available_core,
    )
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution: "test")
    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ={"NATURALLAB_REID_CACHE_DIR": str(tmp_path)},
    )
    service_check = next(
        check for check in report.checks if check.check_id == "vlm_environment"
    )

    assert service_check.status == "warning"
    assert report.exit_code == 0


def test_qwen_profile_requires_pytorch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available = {requirement.module for requirement in doctor.MODULES["core"]}
    available.update(requirement.module for requirement in doctor.MODULES["qwen"])
    available.remove("torch")
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available,
    )
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda distribution: "test",
    )
    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ={"NATURALLAB_REID_CACHE_DIR": str(tmp_path)},
    )
    torch_check = next(
        check
        for check in report.checks
        if check.check_id == "module_qwen_torch"
    )

    assert torch_check.status == "fail"
    assert torch_check.details["required"] is True
    assert report.exit_code == 1


def test_qwen_checkpoint_override_is_checked_without_exposing_its_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available = {requirement.module for requirement in doctor.MODULES["core"]}
    available.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available,
    )
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda distribution: "test",
    )
    secret_directory = tmp_path / "private-study-secret"
    secret_directory.mkdir()
    checkpoint = secret_directory / DEFAULT_REID_FILENAME
    checkpoint.write_bytes(b"checkpoint")
    environment = {doctor.REID_MODEL_PATH_ENV: str(checkpoint)}
    monkeypatch.setattr(
        reid_module,
        "verify_reid_checkpoint",
        lambda path, model: None,
    )

    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ=environment,
    )
    checkpoint_check = next(
        check
        for check in report.checks
        if check.check_id == "qwen_reid_checkpoint"
    )
    serialized = json.dumps(report.to_dict())

    assert checkpoint_check.status == "pass"
    assert checkpoint_check.details["checked_without_loading"] is True
    assert checkpoint_check.details["downloads_attempted"] is False
    assert checkpoint_check.details["path_value_redacted"] is True
    assert str(secret_directory) not in serialized


def test_qwen_checkpoint_override_rejects_unverified_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available = {requirement.module for requirement in doctor.MODULES["core"]}
    available.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available,
    )
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda distribution: "test",
    )
    secret_directory = tmp_path / "private-study-secret"
    secret_directory.mkdir()
    checkpoint = secret_directory / DEFAULT_REID_FILENAME
    checkpoint.write_bytes(b"not the pinned checkpoint")

    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ={doctor.REID_MODEL_PATH_ENV: str(checkpoint)},
    )
    checkpoint_check = next(
        check
        for check in report.checks
        if check.check_id == "qwen_reid_checkpoint"
    )

    assert checkpoint_check.status == "fail"
    assert checkpoint_check.details["integrity_verified"] is False
    assert checkpoint_check.details["failure_category"] == "size-mismatch"
    assert str(secret_directory) not in json.dumps(report.to_dict())
    assert report.exit_code == 1


def test_qwen_checkpoint_missing_is_a_read_only_download_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available = {requirement.module for requirement in doctor.MODULES["core"]}
    available.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available,
    )
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda distribution: "test",
    )
    before = tuple(tmp_path.iterdir())

    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ={"NATURALLAB_REID_CACHE_DIR": str(tmp_path)},
    )
    checkpoint_check = next(
        check
        for check in report.checks
        if check.check_id == "qwen_reid_checkpoint"
    )

    assert checkpoint_check.status == "warning"
    assert checkpoint_check.details["failure_category"] == "missing"
    assert checkpoint_check.details["downloads_attempted"] is False
    assert tuple(tmp_path.iterdir()) == before
    assert report.exit_code == 0


def test_invalid_qwen_endpoint_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_secret_endpoint = "not-a-url-with-secret-token"
    available_core = {requirement.module for requirement in doctor.MODULES["core"]}
    available_core.update(requirement.module for requirement in doctor.MODULES["qwen"])
    monkeypatch.setattr(
        doctor,
        "_module_available",
        lambda module: module in available_core,
    )
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution: "test")
    report = doctor.run_doctor(
        profile="qwen",
        cwd=tmp_path,
        environ={
            doctor.VLM_BASE_URL_ENV: invalid_secret_endpoint,
            "NATURALLAB_REID_CACHE_DIR": str(tmp_path),
        },
    )
    serialized = json.dumps(report.to_dict())
    service_check = next(
        check for check in report.checks if check.check_id == "vlm_environment"
    )

    assert invalid_secret_endpoint not in serialized
    assert service_check.status == "warning"
    assert service_check.details["endpoint_valid"] is False
    assert report.exit_code == 0
