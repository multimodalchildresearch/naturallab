from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from naturallab.spatial_tracking.tracking.deepsort.feature_extractor import (
    AppearanceFeatureExtractor,
    _OSNetAIN,
    _load_backbone_checkpoint,
)


class _DeterministicEmbeddingModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (inputs.shape[0], 512),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        output[:, 0] = 3.0
        output[:, 1] = 4.0
        return output


class _WrongEmbeddingModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (inputs.shape[0], 16),
            dtype=inputs.dtype,
            device=inputs.device,
        )


def _inject_model(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
) -> None:
    monkeypatch.setattr(
        AppearanceFeatureExtractor,
        "_load_model",
        lambda self, model_path: model,
    )


def test_preprocessing_uses_tall_rgb_osnet_geometry() -> None:
    blue_bgr = np.zeros((40, 20, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255

    processed = AppearanceFeatureExtractor._preprocess_image(blue_bgr)

    assert processed is not None
    assert processed.shape == (3, 256, 128)
    # BGR blue becomes RGB channel 2, while the red channel remains zero.
    assert processed[0, 0, 0] == pytest.approx(-0.485 / 0.229)
    assert processed[2, 0, 0] == pytest.approx(
        (1.0 - 0.406) / 0.225
    )


def test_embeddings_are_l2_normalized_without_implicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_model(monkeypatch, _DeterministicEmbeddingModel())
    extractor = AppearanceFeatureExtractor("unused.pth", device="cpu")

    embedding = extractor.extract_features(
        np.zeros((300, 160, 3), dtype=np.uint8)
    )

    assert extractor.has_model is True
    assert embedding is not None
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert embedding[0] == pytest.approx(0.6)
    assert embedding[1] == pytest.approx(0.8)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)

    extractor.has_model = False
    monkeypatch.setattr(
        extractor,
        "extract_fallback_features",
        lambda image: pytest.fail("fallback must be explicitly selected"),
    )
    assert extractor.extract_features(
        np.zeros((10, 10, 3), dtype=np.uint8)
    ) is None


def test_startup_smoke_rejects_wrong_embedding_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_model(monkeypatch, _WrongEmbeddingModel())

    extractor = AppearanceFeatureExtractor("unused.pth", device="cpu")

    assert extractor.has_model is False
    assert isinstance(extractor.model_error, RuntimeError)
    assert "512-D" in str(extractor.model_error)


def test_explicit_unavailable_cuda_does_not_fall_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    extractor = AppearanceFeatureExtractor("unused.pth", device="cuda")

    assert extractor.has_model is False
    assert extractor.device_str == "cuda"
    assert extractor.device is None
    assert isinstance(extractor.model_error, RuntimeError)
    assert "explicitly requested" in str(extractor.model_error)


def test_model_architecture_must_match_vendored_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_called = False

    def unexpected_load(
        self: AppearanceFeatureExtractor,
        model_path: str,
    ) -> nn.Module:
        del self, model_path
        nonlocal load_called
        load_called = True
        return _DeterministicEmbeddingModel()

    monkeypatch.setattr(
        AppearanceFeatureExtractor,
        "_load_model",
        unexpected_load,
    )

    extractor = AppearanceFeatureExtractor(
        "unused.pth",
        model_architecture="different_architecture",
        device="cpu",
    )

    assert extractor.has_model is False
    assert extractor.model_arch == "different_architecture"
    assert isinstance(extractor.model_error, ValueError)
    assert "expected 'osnet_ain_x1_0'" in str(extractor.model_error)
    assert load_called is False


def _save_prefixed_checkpoint(
    path: Path,
    state_dict: OrderedDict[str, torch.Tensor],
) -> None:
    torch.save(
        {
            "state_dict": OrderedDict(
                (f"module.{key}", value)
                for key, value in state_dict.items()
            )
        },
        path,
    )


def test_checkpoint_requires_the_complete_backbone(
    tmp_path: Path,
) -> None:
    source = _OSNetAIN().state_dict()
    source["classifier.weight"] = torch.zeros((7, 512))
    source["classifier.bias"] = torch.zeros(7)
    valid_path = tmp_path / "valid.pth"
    _save_prefixed_checkpoint(valid_path, source)

    _load_backbone_checkpoint(_OSNetAIN(), valid_path)

    missing = OrderedDict(source)
    missing.pop("conv1.conv.weight")
    missing_path = tmp_path / "missing.pth"
    _save_prefixed_checkpoint(missing_path, missing)
    with pytest.raises(ValueError, match="missing=.*conv1.conv.weight"):
        _load_backbone_checkpoint(_OSNetAIN(), missing_path)

    mismatched = OrderedDict(source)
    mismatched["conv1.conv.weight"] = torch.zeros((63, 3, 7, 7))
    mismatched_path = tmp_path / "mismatched.pth"
    _save_prefixed_checkpoint(mismatched_path, mismatched)
    with pytest.raises(ValueError, match="shape_mismatch"):
        _load_backbone_checkpoint(_OSNetAIN(), mismatched_path)


@pytest.mark.skipif(
    not os.environ.get("NATURALLAB_REID_SMOKE_MODEL"),
    reason="set NATURALLAB_REID_SMOKE_MODEL for the real-checkpoint smoke",
)
def test_real_checkpoint_produces_a_normalized_embedding() -> None:
    extractor = AppearanceFeatureExtractor(
        os.environ["NATURALLAB_REID_SMOKE_MODEL"],
        model_architecture="osnet_ain_x1_0",
        device="cpu",
    )

    embedding = extractor.extract_deep_features(
        np.full((300, 160, 3), 127, dtype=np.uint8)
    )

    assert extractor.has_model is True
    assert embedding is not None
    assert embedding.shape == (512,)
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
