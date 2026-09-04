"""Read-only environment diagnostics for NaturalLab.

The doctor deliberately performs no downloads, network requests, or filesystem
writes. It reports optional capabilities as warnings and treats prerequisites
of an explicitly selected workflow profile as failures.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from naturallab import __version__


MINIMUM_PYTHON = (3, 10)
PROFILE_NAMES = (
    "core",
    "spatial",
    "yolo",
    "gaze",
    "acquisition",
    "qwen",
    "all",
)
VLM_BASE_URL_ENV = "NATURALLAB_VLM_BASE_URL"
VLM_API_KEY_ENV = "NATURALLAB_VLM_API_KEY"
REID_MODEL_PATH_ENV = "NATURALLAB_REID_MODEL_PATH"


@dataclass(frozen=True)
class ModuleRequirement:
    module: str
    distribution: str
    purpose: str


MODULES: Dict[str, Tuple[ModuleRequirement, ...]] = {
    "core": (
        ModuleRequirement("numpy", "numpy", "numerical arrays"),
        ModuleRequirement("pandas", "pandas", "tabular results"),
        ModuleRequirement("cv2", "opencv-python", "video and image input"),
        ModuleRequirement("PIL", "Pillow", "image handling"),
        ModuleRequirement("yaml", "PyYAML", "configuration files"),
        ModuleRequirement("tqdm", "tqdm", "progress reporting"),
    ),
    "spatial": (
        ModuleRequirement("torch", "torch", "model inference"),
        ModuleRequirement("torchvision", "torchvision", "vision operations"),
        ModuleRequirement("filterpy", "filterpy", "Kalman filtering"),
        ModuleRequirement("scipy", "scipy", "scientific calculations"),
        ModuleRequirement("transformers", "transformers", "model adapters"),
    ),
    "yolo": (
        ModuleRequirement("ultralytics", "ultralytics", "YOLO detection"),
    ),
    "gaze": (
        ModuleRequirement("torch", "torch", "model inference"),
        ModuleRequirement("torchvision", "torchvision", "vision operations"),
        ModuleRequirement("transformers", "transformers", "object detection"),
        ModuleRequirement("h5py", "h5py", "prototype storage"),
        ModuleRequirement("matplotlib", "matplotlib", "diagnostic plots"),
    ),
    "acquisition": (
        ModuleRequirement("av", "av", "audio/video stream decoding"),
        ModuleRequirement("pylsl", "pylsl", "Lab Streaming Layer"),
        ModuleRequirement(
            "pupil_labs",
            "pupil-labs-realtime-api",
            "Pupil Labs streaming",
        ),
        ModuleRequirement("pyxdf", "pyxdf", "XDF recording import"),
        ModuleRequirement("scipy", "scipy", "audio export"),
        ModuleRequirement("soundfile", "soundfile", "audio export"),
    ),
    # Qwen is accessed through an OpenAI-compatible institutional service. The
    # quality preset also requires the bundled DeepSORT/OSNet tracking path.
    "qwen": (
        ModuleRequirement("filterpy", "filterpy", "Kalman filtering"),
        ModuleRequirement("scipy", "scipy", "appearance matching"),
        ModuleRequirement("torch", "torch", "OSNet inference"),
    ),
}


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic result suitable for both terminal and JSON output."""

    check_id: str
    label: str
    status: str
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    """Complete result from a profile-aware diagnostic run."""

    profile: str
    checks: Tuple[CheckResult, ...]

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "fail" for check in self.checks) else 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def summary(self) -> Dict[str, int]:
        counts = {"pass": 0, "warning": 0, "fail": 0, "skip": 0}
        for check in self.checks:
            counts[check.status] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application": "naturallab",
            "version": __version__,
            "profile": self.profile,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "summary": self.summary(),
            "checks": [check.to_dict() for check in self.checks],
        }


def _python_check() -> CheckResult:
    current = sys.version_info[:3]
    minimum_text = ".".join(str(part) for part in MINIMUM_PYTHON)
    current_text = ".".join(str(part) for part in current)
    if current < MINIMUM_PYTHON:
        return CheckResult(
            "python",
            "Python",
            "fail",
            f"Python {current_text} is unsupported; {minimum_text}+ is required.",
            {"version": current_text, "minimum": minimum_text},
        )
    return CheckResult(
        "python",
        "Python",
        "pass",
        f"Python {current_text} meets the {minimum_text}+ requirement.",
        {"version": current_text, "minimum": minimum_text},
    )


def _working_directory_check(cwd: Optional[Path] = None) -> CheckResult:
    try:
        directory = Path.cwd() if cwd is None else Path(cwd)
        exists = directory.exists()
        is_directory = directory.is_dir()
        writable = exists and is_directory and os.access(str(directory), os.W_OK)
    except (OSError, RuntimeError) as error:
        return CheckResult(
            "working_directory",
            "Working directory",
            "fail",
            f"The working directory cannot be inspected: {type(error).__name__}.",
            {"exists": False, "is_directory": False, "writable": False},
        )

    details = {
        "exists": exists,
        "is_directory": is_directory,
        "writable": writable,
        "method": "access-check-only",
    }
    if not exists:
        message = "The working directory does not exist."
    elif not is_directory:
        message = "The working path is not a directory."
    elif not writable:
        message = "The working directory is not writable."
    else:
        return CheckResult(
            "working_directory",
            "Working directory",
            "pass",
            "The working directory is writable (checked without creating files).",
            details,
        )
    return CheckResult(
        "working_directory",
        "Working directory",
        "fail",
        message,
        details,
    )


def _ffmpeg_check() -> CheckResult:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return CheckResult(
            "ffmpeg",
            "FFmpeg",
            "warning",
            "FFmpeg is not on PATH; video conversion features may be unavailable.",
            {"available": False},
        )
    return CheckResult(
        "ffmpeg",
        "FFmpeg",
        "pass",
        "FFmpeg is available.",
        {"available": True, "command": Path(executable).name},
    )


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_check(
    requirement: ModuleRequirement,
    *,
    required: bool,
    profile: str,
) -> CheckResult:
    available = _module_available(requirement.module)
    version = (
        _distribution_version(requirement.distribution) if available else None
    )
    details = {
        "module": requirement.module,
        "distribution": requirement.distribution,
        "version": version,
        "available": available,
        "required": required,
        "profile": profile,
    }
    check_id = f"module_{profile}_{requirement.module}"
    if available:
        suffix = f" {version}" if version else ""
        return CheckResult(
            check_id,
            requirement.distribution,
            "pass",
            f"{requirement.distribution}{suffix} is available for "
            f"{requirement.purpose}.",
            details,
        )

    if required:
        return CheckResult(
            check_id,
            requirement.distribution,
            "fail",
            f"{requirement.distribution} is required by the selected profile "
            "but cannot be located.",
            details,
        )
    return CheckResult(
        check_id,
        requirement.distribution,
        "warning",
        f"{requirement.distribution} is not installed; "
        f"{requirement.purpose} is unavailable.",
        details,
    )


def _selected_profiles(profile: str) -> Tuple[str, ...]:
    if profile == "all":
        return ("core", "spatial", "yolo", "gaze", "acquisition", "qwen")
    if profile == "core":
        return ("core",)
    return ("core", profile)


def _module_checks(profile: str) -> Iterable[CheckResult]:
    selected_requirements: Dict[
        str,
        Tuple[ModuleRequirement, bool, str],
    ] = {}
    for selected_profile in _selected_profiles(profile):
        for requirement in MODULES[selected_profile]:
            required = True
            existing = selected_requirements.get(requirement.module)
            if existing is None or (required and not existing[1]):
                selected_requirements[requirement.module] = (
                    requirement,
                    required,
                    selected_profile,
                )
    for requirement, required, selected_profile in (
        selected_requirements.values()
    ):
        yield _module_check(
            requirement,
            required=required,
            profile=selected_profile,
        )


def _cuda_check() -> CheckResult:
    if not _module_available("torch"):
        return CheckResult(
            "cuda",
            "CUDA",
            "skip",
            "CUDA was not checked because PyTorch is not installed.",
            {"torch_available": False, "cuda_available": None},
        )

    # Import PyTorch in an isolated interpreter. Native library incompatibilities
    # can terminate a process instead of raising Python exceptions; they should
    # make this optional check warn, not take down the doctor command.
    probe = """
import json
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
import torch
available = bool(torch.cuda.is_available())
print(json.dumps({
    "cuda_available": available,
    "device_count": int(torch.cuda.device_count()) if available else 0,
    "torch_cuda_version": getattr(torch.version, "cuda", None),
}))
"""
    probe_environment = os.environ.copy()
    probe_environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=probe_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CheckResult(
            "cuda",
            "CUDA",
            "warning",
            f"The isolated CUDA probe could not run: {type(error).__name__}.",
            {
                "torch_available": True,
                "cuda_available": None,
                "error_type": type(error).__name__,
            },
        )

    if completed.returncode != 0:
        return CheckResult(
            "cuda",
            "CUDA",
            "warning",
            "The installed PyTorch build could not complete the isolated CUDA "
            "probe.",
            {
                "torch_available": True,
                "cuda_available": None,
                "probe_return_code": completed.returncode,
            },
        )

    try:
        probe_result = json.loads(completed.stdout)
        available = bool(probe_result["cuda_available"])
        device_count = int(probe_result["device_count"])
        cuda_version = probe_result.get("torch_cuda_version")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return CheckResult(
            "cuda",
            "CUDA",
            "warning",
            "The isolated CUDA probe returned an unreadable result.",
            {
                "torch_available": True,
                "cuda_available": None,
                "error_type": type(error).__name__,
            },
        )

    details = {
        "torch_available": True,
        "cuda_available": available,
        "device_count": device_count,
        "torch_cuda_version": cuda_version,
    }
    if not available:
        return CheckResult(
            "cuda",
            "CUDA",
            "warning",
            "PyTorch is installed but CUDA is unavailable; GPU workflows will "
            "fall back to CPU or need a cluster service.",
            details,
        )
    return CheckResult(
        "cuda",
        "CUDA",
        "pass",
        f"CUDA is available to PyTorch with {device_count} device(s).",
        details,
    )


def _vlm_environment_check(environ: Mapping[str, str]) -> CheckResult:
    """Check service configuration without returning either environment value."""

    endpoint_value = environ.get(VLM_BASE_URL_ENV, "").strip()
    endpoint_configured = bool(endpoint_value)
    credential_configured = bool(environ.get(VLM_API_KEY_ENV, "").strip())
    try:
        parsed_endpoint = urlsplit(endpoint_value)
        parsed_endpoint.port
        endpoint_valid = (
            endpoint_configured
            and parsed_endpoint.scheme in {"http", "https"}
            and bool(parsed_endpoint.hostname)
        )
    except ValueError:
        endpoint_valid = False
    details = {
        "endpoint_variable": VLM_BASE_URL_ENV,
        "endpoint_configured": endpoint_configured,
        "endpoint_valid": endpoint_valid,
        "credential_variable": VLM_API_KEY_ENV,
        "credential_configured": credential_configured,
        "values_redacted": True,
    }
    if not endpoint_configured:
        return CheckResult(
            "vlm_environment",
            "VLM service",
            "warning",
            f"{VLM_BASE_URL_ENV} is not configured; the Qwen service path is "
            "unavailable.",
            details,
        )

    if not endpoint_valid:
        return CheckResult(
            "vlm_environment",
            "VLM service",
            "warning",
            f"{VLM_BASE_URL_ENV} is configured but is not a valid HTTP(S) URL. "
            "Its value is redacted.",
            details,
        )

    credential_note = (
        " and a credential is configured"
        if credential_configured
        else "; no credential is configured"
    )
    return CheckResult(
        "vlm_environment",
        "VLM service",
        "pass",
        f"A VLM endpoint is configured{credential_note}. Values are redacted.",
        details,
    )


def _reid_checkpoint_check(
    *,
    cwd: Optional[Path],
    environ: Mapping[str, str],
) -> CheckResult:
    """Verify cached checkpoint integrity without loading or downloading it."""

    override_configured = bool(
        environ.get(REID_MODEL_PATH_ENV, "").strip()
    )
    source = "environment" if override_configured else "naturallab-cache"
    try:
        from naturallab.spatial_tracking.pipeline.config import (
            load_spatial_pipeline_preset,
            resolve_reid_model_path,
        )
        from naturallab.spatial_tracking.pipeline.reid import (
            ReIDCheckpointError,
            verify_reid_checkpoint,
        )

        preset = load_spatial_pipeline_preset()
        model = preset.tracker.reid_model
        checkpoint = resolve_reid_model_path(
            preset,
            environ=environ,
            cwd=cwd,
        )
    except Exception as error:
        return CheckResult(
            "qwen_reid_checkpoint",
            "OSNet-AIN checkpoint",
            "fail",
            "The configured OSNet-AIN checkpoint could not be resolved "
            f"({type(error).__name__}).",
            {
                "source": source,
                "path_value_redacted": override_configured,
                "checked_without_loading": True,
                "downloads_attempted": False,
            },
        )

    details = {
        "source": source,
        "filename": None if override_configured else checkpoint.name,
        "expected_filename": model.filename,
        "architecture": model.architecture,
        "repository": model.repository,
        "revision": model.revision,
        "expected_sha256": model.sha256,
        "expected_size_bytes": model.size_bytes,
        "canonical_extension": ".pth",
        "checked_without_loading": True,
        "downloads_attempted": False,
        "path_value_redacted": override_configured,
    }
    try:
        verify_reid_checkpoint(checkpoint, model)
    except ReIDCheckpointError as error:
        details["integrity_verified"] = False
        details["failure_category"] = error.category
        if override_configured:
            return CheckResult(
                "qwen_reid_checkpoint",
                "OSNet-AIN checkpoint",
                "fail",
                "The checkpoint configured by "
                f"{REID_MODEL_PATH_ENV} is unavailable or does not match the "
                "exact artifact pinned by the quality preset. No download was "
                "attempted.",
                details,
            )
        return CheckResult(
            "qwen_reid_checkpoint",
            "OSNet-AIN checkpoint",
            "warning",
            "The verified OSNet-AIN checkpoint is not ready in the NaturalLab "
            "cache. The quality pipeline will attempt the pinned download at "
            "runtime and will stop if it cannot verify it. No download was "
            "attempted by doctor.",
            details,
        )

    details["integrity_verified"] = True
    details["failure_category"] = None
    return CheckResult(
        "qwen_reid_checkpoint",
        "OSNet-AIN checkpoint",
        "pass",
        "The exact pinned OSNet-AIN checkpoint passed size and SHA-256 "
        "verification. Model weights were not loaded.",
        details,
    )


def run_doctor(
    profile: str = "core",
    *,
    cwd: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> DoctorReport:
    """Run read-only diagnostics for *profile* and return a structured report."""

    if profile not in PROFILE_NAMES:
        choices = ", ".join(PROFILE_NAMES)
        raise ValueError(f"Unknown doctor profile {profile!r}; choose from {choices}.")

    checks: List[CheckResult] = [
        _python_check(),
        _working_directory_check(cwd),
        _ffmpeg_check(),
    ]
    checks.extend(_module_checks(profile))
    if profile in {"spatial", "yolo", "gaze", "all"}:
        checks.append(_cuda_check())
    if profile in {"qwen", "all"}:
        selected_environment = os.environ if environ is None else environ
        checks.append(_vlm_environment_check(selected_environment))
        checks.append(
            _reid_checkpoint_check(
                cwd=cwd,
                environ=selected_environment,
            )
        )
    return DoctorReport(profile=profile, checks=tuple(checks))


def format_human(report: DoctorReport) -> str:
    """Render a compact, colour-free report suitable for any terminal."""

    status_labels = {
        "pass": "PASS",
        "warning": "WARN",
        "fail": "FAIL",
        "skip": "SKIP",
    }
    lines = [
        f"NaturalLab doctor {__version__} (profile: {report.profile})",
        "",
    ]
    for check in report.checks:
        lines.append(
            f"[{status_labels[check.status]}] {check.label}: {check.message}"
        )

    summary = report.summary()
    lines.extend(
        [
            "",
            "Summary: "
            f"{summary['pass']} passed, {summary['warning']} warning(s), "
            f"{summary['fail']} failed, {summary['skip']} skipped.",
            (
                "Required checks passed. Warnings identify optional capabilities."
                if report.ok
                else "Required checks failed; resolve them before processing data."
            ),
        ]
    )
    return "\n".join(lines)
