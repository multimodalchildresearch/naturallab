"""Person appearance embeddings for the DeepSORT tracker.

The OSNet-AIN implementation in this module is adapted for inference from:

    Kaiyang Zhou, ``deep-person-reid``
    https://github.com/KaiyangZhou/deep-person-reid
    commit f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
    source: ``torchreid/models/osnet_ain.py``

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_MODEL_ARCHITECTURE = "osnet_ain_x1_0"
_DEFAULT_MODEL_FILENAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_"
    "coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 128
_EMBEDDING_DIMENSION = 512
_PIXEL_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class _ConvLayer(nn.Module):
    """Convolution followed by normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if instance_norm:
            self.bn: nn.Module = nn.InstanceNorm2d(
                out_channels,
                affine=True,
            )
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(inputs)))


class _Conv1x1(nn.Module):
    """1x1 convolution followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(inputs)))


class _Conv1x1Linear(nn.Module):
    """1x1 convolution with optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        if self.bn is not None:
            outputs = self.bn(outputs)
        return outputs


class _LightConv3x3(nn.Module):
    """Pointwise convolution followed by a depthwise 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return self.relu(self.bn(outputs))


class _LightConvStream(nn.Module):
    """A stream of lightweight convolutions at one receptive-field scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("OSNet stream depth must be at least one")
        layers: list[nn.Module] = [
            _LightConv3x3(in_channels, out_channels)
        ]
        layers.extend(
            _LightConv3x3(out_channels, out_channels)
            for _ in range(depth - 1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _ChannelGate(nn.Module):
    """Generate channel-wise gates conditioned on the input tensor."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            in_channels,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.gate_activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gates = self.global_avgpool(inputs)
        gates = self.relu(self.fc1(gates))
        gates = self.gate_activation(self.fc2(gates))
        return inputs * gates


class _OSBlock(nn.Module):
    """Omni-scale feature-learning block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = _Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList(
            _LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, 5)
        )
        self.gate = _ChannelGate(mid_channels)
        self.conv3 = _Conv1x1Linear(mid_channels, out_channels)
        self.downsample = (
            _Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        scale_input = self.conv1(inputs)
        scale_sum = torch.zeros_like(scale_input)
        for stream in self.conv2:
            scale_sum = scale_sum + self.gate(stream(scale_input))
        outputs = self.conv3(scale_sum)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(outputs + identity)


class _OSBlockINin(nn.Module):
    """OSNet block with instance normalization inside the residual."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = _Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList(
            _LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, 5)
        )
        self.gate = _ChannelGate(mid_channels)
        self.conv3 = _Conv1x1Linear(
            mid_channels,
            out_channels,
            batch_norm=False,
        )
        self.downsample = (
            _Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )
        # The uppercase attribute name is retained for checkpoint compatibility.
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        scale_input = self.conv1(inputs)
        scale_sum = torch.zeros_like(scale_input)
        for stream in self.conv2:
            scale_sum = scale_sum + self.gate(stream(scale_input))
        outputs = self.IN(self.conv3(scale_sum))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(outputs + identity)


class _OSNetAIN(nn.Module):
    """Minimal OSNet-AIN x1.0 network needed for inference."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = _EMBEDDING_DIMENSION
        self.conv1 = _ConvLayer(
            3,
            64,
            7,
            stride=2,
            padding=3,
            instance_norm=True,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            _OSBlockINin(64, 256),
            _OSBlockINin(256, 256),
        )
        self.pool2 = nn.Sequential(
            _Conv1x1(256, 256),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            _OSBlock(256, 384),
            _OSBlockINin(384, 384),
        )
        self.pool3 = nn.Sequential(
            _Conv1x1(384, 384),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            _OSBlockINin(384, 512),
            _OSBlock(512, 512),
        )
        self.conv5 = _Conv1x1(512, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, _EMBEDDING_DIMENSION),
            nn.BatchNorm1d(_EMBEDDING_DIMENSION),
            nn.ReLU(),
        )
        # The source-domain identity head is deliberately unused at inference.
        self.classifier = nn.Linear(_EMBEDDING_DIMENSION, 1)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.maxpool(outputs)
        outputs = self.conv2(outputs)
        outputs = self.pool2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.pool3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.global_avgpool(outputs)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return self.fc(outputs)


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("OSNet checkpoint must contain a mapping")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("OSNet checkpoint state_dict must be a mapping")
    return state_dict


def _normalized_checkpoint_keys(
    state_dict: Mapping[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str):
            raise TypeError("OSNet checkpoint keys must be strings")
        key = (
            raw_key.removeprefix("module.")
            if raw_key.startswith("module.")
            else raw_key
        )
        if key in normalized:
            raise ValueError(
                f"OSNet checkpoint contains duplicate key {key!r}"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"OSNet checkpoint value for {raw_key!r} is not a tensor"
            )
        normalized[key] = value
    return normalized


def _load_backbone_checkpoint(
    model: _OSNetAIN,
    model_path: str | Path,
) -> None:
    """Load every non-classifier parameter with exact key and shape checks."""

    checkpoint = torch.load(
        Path(model_path),
        map_location="cpu",
        weights_only=True,
    )
    supplied = _normalized_checkpoint_keys(
        _checkpoint_state_dict(checkpoint)
    )
    expected = model.state_dict()
    expected_backbone = {
        key: value
        for key, value in expected.items()
        if not key.startswith("classifier.")
    }

    missing = sorted(set(expected_backbone).difference(supplied))
    unexpected = sorted(
        key
        for key in supplied
        if key not in expected and not key.startswith("classifier.")
    )
    mismatched = sorted(
        key
        for key, expected_value in expected_backbone.items()
        if key in supplied
        and tuple(supplied[key].shape) != tuple(expected_value.shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatched:
            shape_details = {
                key: {
                    "checkpoint": tuple(supplied[key].shape),
                    "expected": tuple(expected_backbone[key].shape),
                }
                for key in mismatched
            }
            details.append(f"shape_mismatch={shape_details}")
        raise ValueError(
            "OSNet checkpoint does not exactly match the "
            f"{_MODEL_ARCHITECTURE} inference backbone: "
            + "; ".join(details)
        )

    backbone = OrderedDict(
        (key, supplied[key]) for key in expected_backbone
    )
    load_result = model.load_state_dict(backbone, strict=False)
    permitted_missing = {"classifier.weight", "classifier.bias"}
    if (
        set(load_result.missing_keys) != permitted_missing
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "OSNet checkpoint produced an unexpected load result: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("ReID device must be one of: cpu, cuda, mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested for ReID but is unavailable"
        )
    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "MPS was explicitly requested for ReID but is unavailable"
            )
    return torch.device(device)


def _validated_normalized_embedding(
    output: torch.Tensor,
) -> torch.Tensor:
    if output.shape != (1, _EMBEDDING_DIMENSION):
        raise RuntimeError(
            "OSNet-AIN must return a single 512-D embedding; "
            f"received shape {tuple(output.shape)}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("OSNet-AIN returned a non-finite embedding")
    norm = torch.linalg.vector_norm(output, ord=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norm).all().item()) or float(norm.item()) <= 0:
        raise RuntimeError("OSNet-AIN returned a zero-norm embedding")
    return output / norm


def _extract_hsv_histogram(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    resized = cv2.resize(image, (64, 128))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
    return np.concatenate([h_hist, s_hist, v_hist]).flatten()


class HistogramFeatureExtractor:
    """Fixed fallback backend used only after an explicit policy opt-in."""

    has_model = False
    last_inference_error = None

    def __init__(self, model_error: Optional[BaseException] = None) -> None:
        self.model_error = model_error

    @staticmethod
    def extract_fallback_features(
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        return _extract_hsv_histogram(image)


class AppearanceFeatureExtractor:
    """Extract L2-normalized OSNet-AIN person appearance embeddings."""

    image_size = (_IMAGE_HEIGHT, _IMAGE_WIDTH)
    embedding_dimension = _EMBEDDING_DIMENSION

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_FILENAME,
        model_architecture: str = _MODEL_ARCHITECTURE,
        device: str = "cpu",
    ) -> None:
        self.logger = logging.getLogger(
            "naturallab.spatial_tracking.AppearanceFeatureExtractor"
        )
        self.model_arch = model_architecture
        self.device_str = device
        self.device: Optional[torch.device] = None
        self.model: Optional[nn.Module] = None
        self.model_error: Optional[BaseException] = None
        self.last_inference_error: Optional[BaseException] = None
        self.has_model = False

        try:
            if model_architecture != _MODEL_ARCHITECTURE:
                raise ValueError(
                    "Unsupported ReID model architecture "
                    f"{model_architecture!r}; expected "
                    f"{_MODEL_ARCHITECTURE!r}"
                )
            self.device = _resolve_device(device)
            self.model = self._load_model(model_path)
            self.model.to(self.device)
            self.model.float()
            self.model.eval()
            self._run_startup_smoke()
        except Exception as error:
            self.model_error = error
            self.logger.error(
                "Could not initialize %s from %s on %s: %s",
                self.model_arch,
                model_path,
                device,
                error,
            )
            return

        self.has_model = True
        self.logger.info(
            "Loaded %s from %s on %s",
            self.model_arch,
            model_path,
            device,
        )

    def _load_model(self, model_path: str) -> _OSNetAIN:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ReID checkpoint not found: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"ReID checkpoint is empty: {path}")
        model = _OSNetAIN()
        _load_backbone_checkpoint(model, path)
        return model

    def _run_startup_smoke(self) -> None:
        if self.model is None or self.device is None:
            raise RuntimeError("OSNet-AIN model or device was not initialized")
        smoke_input = torch.zeros(
            (1, 3, _IMAGE_HEIGHT, _IMAGE_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            output = self.model(smoke_input)
        _validated_normalized_embedding(output)

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        resized = cv2.resize(image, (_IMAGE_WIDTH, _IMAGE_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - _PIXEL_MEAN) / _PIXEL_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1))

    def extract_deep_features(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return one normalized embedding, or ``None`` after a hard failure."""

        if image is None or image.size == 0:
            return None
        if not self.has_model or self.model is None or self.device is None:
            return None

        try:
            self.last_inference_error = None
            preprocessed = self._preprocess_image(image)
            if preprocessed is None:
                return None
            tensor = (
                torch.from_numpy(preprocessed)
                .unsqueeze(0)
                .to(self.device, dtype=torch.float32)
            )
            with torch.inference_mode():
                output = self.model(tensor)
            embedding = _validated_normalized_embedding(output)
            return embedding[0].detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        except Exception as error:
            self.last_inference_error = error
            self.logger.error("OSNet-AIN feature extraction failed: %s", error)
            return None

    def extract_fallback_features(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Explicitly extract legacy HSV histograms when a caller opts in."""

        return _extract_hsv_histogram(image)

    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract OSNet features without an implicit histogram fallback."""

        return self.extract_deep_features(image)
