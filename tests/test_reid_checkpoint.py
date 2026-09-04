from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from naturallab.spatial_tracking.pipeline import (
    DEFAULT_REID_FILENAME,
    ReIDCheckpointError,
    ReIDModelConfig,
    acquire_reid_checkpoint,
    load_spatial_pipeline_preset,
    verify_reid_checkpoint,
)
from naturallab.spatial_tracking.pipeline import reid as reid_module


def _tiny_model(payload: bytes) -> ReIDModelConfig:
    return ReIDModelConfig(
        architecture="osnet_ain_x1_0",
        repository="research/model",
        revision="1" * 40,
        filename="tiny.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        auto_download=True,
    )


def test_checkpoint_integrity_requires_exact_size_and_sha256(
    tmp_path: Path,
) -> None:
    payload = b"verified checkpoint bytes"
    model = _tiny_model(payload)
    checkpoint = tmp_path / model.filename
    checkpoint.write_bytes(payload)

    verify_reid_checkpoint(checkpoint, model)

    with pytest.raises(ReIDCheckpointError) as wrong_size:
        verify_reid_checkpoint(
            checkpoint,
            replace(model, size_bytes=len(payload) + 1),
        )
    assert wrong_size.value.category == "size-mismatch"

    with pytest.raises(ReIDCheckpointError) as wrong_hash:
        verify_reid_checkpoint(
            checkpoint,
            replace(model, sha256="0" * 64),
        )
    assert wrong_hash.value.category == "sha256-mismatch"


def test_download_is_pinned_verified_and_atomically_published(
    tmp_path: Path,
) -> None:
    payload = b"one exact model artifact"
    model = _tiny_model(payload)
    target = tmp_path / model.filename
    calls: list[tuple[str, float]] = []

    def opener(request: Any, *, timeout: float) -> io.BytesIO:
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    reid_module._download_verified_checkpoint(
        target,
        model,
        opener=opener,
        timeout_seconds=17.0,
    )

    assert target.read_bytes() == payload
    assert calls == [(model.download_url, 17.0)]
    assert not any(
        path.name.endswith(".download") for path in tmp_path.iterdir()
    )


def test_truncated_download_is_rejected_without_publishing(
    tmp_path: Path,
) -> None:
    payload = b"complete"
    model = _tiny_model(payload)
    target = tmp_path / model.filename

    with pytest.raises(ReIDCheckpointError) as caught:
        reid_module._download_verified_checkpoint(
            target,
            model,
            opener=lambda request, timeout: io.BytesIO(payload[:-1]),
            timeout_seconds=1.0,
        )

    assert caught.value.category == "size-mismatch"
    assert not target.exists()
    assert not any(
        path.name.endswith(".download") for path in tmp_path.iterdir()
    )


def test_cache_hit_does_not_open_the_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preset = load_spatial_pipeline_preset()
    checkpoint = tmp_path / DEFAULT_REID_FILENAME
    checkpoint.write_bytes(b"test seam")
    monkeypatch.setenv("NATURALLAB_REID_CACHE_DIR", str(tmp_path))
    verified: list[Path] = []

    def accept_test_checkpoint(
        path: Path,
        model: ReIDModelConfig,
    ) -> None:
        verified.append(path)

    monkeypatch.setattr(
        reid_module,
        "verify_reid_checkpoint",
        accept_test_checkpoint,
    )

    def network_must_not_run(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("cache hit must not open the network")

    resolution = acquire_reid_checkpoint(
        preset,
        opener=network_must_not_run,
    )

    assert resolution.path == checkpoint
    assert resolution.verified is True
    assert resolution.downloaded is False
    assert verified == [checkpoint]


def test_explicit_model_path_is_verified_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preset = load_spatial_pipeline_preset()
    checkpoint = tmp_path / "explicit.pth"
    checkpoint.write_bytes(b"test seam")
    monkeypatch.setattr(
        reid_module,
        "verify_reid_checkpoint",
        lambda path, model: None,
    )

    resolution = acquire_reid_checkpoint(
        preset,
        configured_path=checkpoint,
        opener=lambda *args, **kwargs: pytest.fail(
            "explicit checkpoint must bypass download"
        ),
    )

    assert resolution.path == checkpoint
    assert resolution.source == "argument"
    assert resolution.downloaded is False
