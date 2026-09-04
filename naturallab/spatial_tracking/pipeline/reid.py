"""Verified acquisition of the preset-pinned person-ReID checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Optional
from urllib.request import Request, urlopen

from naturallab.spatial_tracking.pipeline.config import (
    REID_MODEL_PATH_ENV,
    ReIDModelConfig,
    SpatialPipelinePreset,
    resolve_reid_model_path,
)

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 120.0


def _default_url_opener(request: Request, *, timeout: float) -> Any:
    return urlopen(request, timeout=timeout)


class ReIDCheckpointWarning(RuntimeWarning):
    """Warn that the requested ReID quality path is unavailable."""


class ReIDCheckpointError(RuntimeError):
    """Raised when the pinned ReID checkpoint cannot be acquired or verified."""

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ReIDCheckpointResolution:
    """Resolved checkpoint state recorded in pipeline provenance."""

    path: Path
    source: str
    verified: bool
    downloaded: bool
    fallback_allowed: bool = False
    fallback_used: bool = False
    failure_category: Optional[str] = None

    def provenance(self, model: ReIDModelConfig) -> dict[str, Any]:
        """Return secret-free model, integrity, and fallback metadata."""

        value = model.provenance()
        value.update(
            {
                "checkpoint_source": self.source,
                "checkpoint_verified": self.verified,
                "downloaded_this_run": self.downloaded,
                "fallback_allowed": self.fallback_allowed,
                "fallback_used": self.fallback_used,
                "failure_category": self.failure_category,
                "reid_backend": (
                    "histogram"
                    if self.fallback_used
                    else model.architecture
                ),
            }
        )
        return value


def reid_checkpoint_source(
    *,
    configured_path: Optional[str | os.PathLike[str]],
    environ: Mapping[str, str],
) -> str:
    """Describe how the checkpoint path was selected without exposing it."""

    if configured_path is not None:
        return "argument"
    if environ.get(REID_MODEL_PATH_ENV, "").strip():
        return "environment"
    return "naturallab-cache"


def verify_reid_checkpoint(
    path: Path,
    model: ReIDModelConfig,
) -> None:
    """Require the exact byte count and SHA-256 from the immutable preset."""

    try:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ReIDCheckpointError(
                "the pinned ReID checkpoint is missing or unreadable",
                category="missing",
            )
        size_bytes = path.stat().st_size
    except ReIDCheckpointError:
        raise
    except OSError as error:
        raise ReIDCheckpointError(
            "the pinned ReID checkpoint could not be inspected",
            category="unreadable",
        ) from error

    if size_bytes != model.size_bytes:
        raise ReIDCheckpointError(
            "the ReID checkpoint has the wrong byte size "
            f"(expected {model.size_bytes}, found {size_bytes})",
            category="size-mismatch",
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(_DOWNLOAD_CHUNK_BYTES),
                b"",
            ):
                digest.update(chunk)
    except OSError as error:
        raise ReIDCheckpointError(
            "the ReID checkpoint could not be read for SHA-256 verification",
            category="unreadable",
        ) from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != model.sha256:
        raise ReIDCheckpointError(
            "the ReID checkpoint failed SHA-256 verification "
            f"(expected {model.sha256}, found {actual_sha256})",
            category="sha256-mismatch",
        )


def _download_verified_checkpoint(
    target: Path,
    model: ReIDModelConfig,
    *,
    opener: Callable[..., Any],
    timeout_seconds: float,
) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ReIDCheckpointError(
            "the NaturalLab ReID cache directory could not be created",
            category="cache-unwritable",
        ) from error

    request = Request(
        model.download_url,
        headers={"User-Agent": "NaturalLab-ReID/1"},
    )
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{model.filename}.",
            suffix=".download",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            size_bytes = 0
            with opener(request, timeout=timeout_seconds) as response:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > model.size_bytes:
                        raise ReIDCheckpointError(
                            "the downloaded ReID checkpoint is larger than the "
                            "pinned artifact",
                            category="size-mismatch",
                        )
                    digest.update(chunk)
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        if size_bytes != model.size_bytes:
            raise ReIDCheckpointError(
                "the downloaded ReID checkpoint is truncated "
                f"(expected {model.size_bytes} bytes, received {size_bytes})",
                category="size-mismatch",
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != model.sha256:
            raise ReIDCheckpointError(
                "the downloaded ReID checkpoint failed SHA-256 verification",
                category="sha256-mismatch",
            )
        os.replace(temporary_path, target)
        temporary_path = None
    except ReIDCheckpointError:
        raise
    except (OSError, TimeoutError) as error:
        raise ReIDCheckpointError(
            "the pinned ReID checkpoint could not be downloaded or cached",
            category="download-failed",
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def acquire_reid_checkpoint(
    preset: SpatialPipelinePreset,
    *,
    configured_path: Optional[str | os.PathLike[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    opener: Optional[Callable[..., Any]] = None,
    timeout_seconds: float = _DOWNLOAD_TIMEOUT_SECONDS,
) -> ReIDCheckpointResolution:
    """Resolve, verify, and if needed download the exact preset checkpoint."""

    environment = os.environ if environ is None else environ
    source = reid_checkpoint_source(
        configured_path=configured_path,
        environ=environment,
    )
    path = resolve_reid_model_path(
        preset,
        configured_path=configured_path,
        environ=environment,
        cwd=cwd,
    )
    model = preset.tracker.reid_model
    try:
        verify_reid_checkpoint(path, model)
    except ReIDCheckpointError:
        if source != "naturallab-cache":
            raise ReIDCheckpointError(
                "the explicitly configured ReID checkpoint is unavailable or "
                "does not match the model pinned by the selected preset",
                category="configured-checkpoint-invalid",
            )
        if not model.auto_download:
            raise
    else:
        return ReIDCheckpointResolution(
            path=path,
            source=source,
            verified=True,
            downloaded=False,
        )

    selected_opener: Callable[..., Any]
    if opener is None:
        selected_opener = _default_url_opener
    else:
        selected_opener = opener
    _download_verified_checkpoint(
        path,
        model,
        opener=selected_opener,
        timeout_seconds=timeout_seconds,
    )
    verify_reid_checkpoint(path, model)
    return ReIDCheckpointResolution(
        path=path,
        source=source,
        verified=True,
        downloaded=True,
    )


def fallback_resolution(
    preset: SpatialPipelinePreset,
    error: ReIDCheckpointError,
    *,
    configured_path: Optional[str | os.PathLike[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> ReIDCheckpointResolution:
    """Create explicit histogram-fallback provenance after acquisition fails."""

    environment = os.environ if environ is None else environ
    return ReIDCheckpointResolution(
        path=resolve_reid_model_path(
            preset,
            configured_path=configured_path,
            environ=environment,
            cwd=cwd,
        ),
        source=reid_checkpoint_source(
            configured_path=configured_path,
            environ=environment,
        ),
        verified=False,
        downloaded=False,
        fallback_allowed=True,
        fallback_used=True,
        failure_category=error.category,
    )


def fallback_guidance(message: str) -> str:
    """Add the explicit opt-in required after a ReID failure."""

    return (
        f"{message}. No appearance fallback was used. Resolve the model/cache "
        "problem and rerun, or explicitly accept lower-quality histogram "
        "features for this run with allow_reid_fallback=True "
        "(CLI: --allow-reid-fallback)"
    )
