"""Generic frame sources used by analysis components.

The analysis pipeline must not depend on NaturalLab having recorded the data.
These contracts deliberately cover files, image sequences, live/custom
iterables, and future XDF or institutional-data adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
import re
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


_NATURAL_NUMBER = re.compile(r"(\d+)")


def natural_path_sort_key(
    path: Union[str, Path],
) -> Tuple[Tuple[int, Union[int, str]], ...]:
    """Sort numbered paths in human temporal order (1, 2, 10)."""
    candidate = Path(path)
    parts: List[Tuple[int, Union[int, str]]] = []
    for token in _NATURAL_NUMBER.split(candidate.name):
        if token.isdigit():
            parts.append((0, int(token)))
        elif token:
            parts.append((1, token.casefold()))
    parts.append((2, candidate.name))
    return tuple(parts)


def _validate_optional_fps(fps: Optional[float]) -> Optional[float]:
    if fps is None:
        return None
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not math.isfinite(float(fps))
        or fps <= 0
    ):
        raise ValueError("fps must be a finite positive number when provided")
    return float(fps)


@dataclass(frozen=True)
class FramePacket:
    """One image and the minimum metadata needed by downstream components."""

    image: Any
    frame_index: int
    source_id: str
    timestamp_ns: Optional[int] = None
    source_timestamp: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative when provided")
        if (
            self.source_timestamp is not None
            and (
                isinstance(self.source_timestamp, bool)
                or not isinstance(self.source_timestamp, Real)
                or not math.isfinite(float(self.source_timestamp))
                or self.source_timestamp < 0
            )
        ):
            raise ValueError(
                "source_timestamp must be a finite non-negative number "
                "when provided"
            )


class FrameSource(ABC):
    """Ordered source of image frames.

    Timestamps are optional for frame-level detection. Workflows that calculate
    durations, assign gaze, or synchronize modalities must require them at
    their own boundary.
    """

    source_id: str

    @abstractmethod
    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frames in processing order."""


class IterableFrameSource(FrameSource):
    """Adapt an arbitrary Python iterable of images or :class:`FramePacket`s."""

    def __init__(
        self,
        frames: Iterable[Any],
        *,
        source_id: str = "iterable",
        fps: Optional[float] = None,
        start_timestamp_ns: int = 0,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id must not be empty")
        if start_timestamp_ns < 0:
            raise ValueError("start_timestamp_ns must be non-negative")
        self._frames = frames
        self.source_id = source_id
        self.fps = _validate_optional_fps(fps)
        self.start_timestamp_ns = start_timestamp_ns

    def __iter__(self) -> Iterator[FramePacket]:
        for index, item in enumerate(self._frames):
            if isinstance(item, FramePacket):
                yield item
                continue
            timestamp_ns = None
            source_timestamp = None
            if self.fps is not None:
                source_timestamp = index / self.fps
                timestamp_ns = self.start_timestamp_ns + round(
                    source_timestamp * 1_000_000_000
                )
            yield FramePacket(
                image=item,
                frame_index=index,
                source_id=self.source_id,
                timestamp_ns=timestamp_ns,
                source_timestamp=source_timestamp,
                metadata={
                    "fps": self.fps,
                    "timestamp_source": (
                        "synthesized_fps"
                        if self.fps is not None
                        else None
                    ),
                },
            )


class ImageDirectorySource(FrameSource):
    """Read an ordered directory of still images as a frame sequence."""

    DEFAULT_SUFFIXES: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        directory: Union[str, Path],
        *,
        source_id: Optional[str] = None,
        fps: Optional[float] = None,
        suffixes: Sequence[str] = DEFAULT_SUFFIXES,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError(f"image directory does not exist: {self.directory}")
        self.source_id = source_id or self.directory.name
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        self.fps = _validate_optional_fps(fps)
        normalized_suffixes = {suffix.lower() for suffix in suffixes}
        self.paths = tuple(
            sorted(
                (
                    path
                    for path in self.directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in normalized_suffixes
                ),
                key=natural_path_sort_key,
            )
        )
        if not self.paths:
            raise ValueError(f"no supported images found in {self.directory}")

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "Pillow is required for image-directory input; install naturallab core"
            ) from exc

        for index, path in enumerate(self.paths):
            timestamp_ns = None
            source_timestamp = None
            if self.fps is not None:
                source_timestamp = index / self.fps
                timestamp_ns = round(source_timestamp * 1_000_000_000)
            with Image.open(path) as image:
                decoded = image.convert("RGB").copy()
            yield FramePacket(
                image=decoded,
                frame_index=index,
                source_id=self.source_id,
                timestamp_ns=timestamp_ns,
                source_timestamp=source_timestamp,
                metadata={
                    "path": str(path),
                    "fps": self.fps,
                    "color_space": "RGB",
                    "timestamp_source": (
                        "synthesized_fps"
                        if self.fps is not None
                        else None
                    ),
                },
            )


class VideoFileSource(FrameSource):
    """Decode a conventional video file without coupling consumers to OpenCV."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        source_id: Optional[str] = None,
        start_frame: int = 0,
        stop_frame: Optional[int] = None,
        step: int = 1,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"video file does not exist: {self.path}")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if stop_frame is not None and stop_frame <= start_frame:
            raise ValueError("stop_frame must be greater than start_frame")
        if step <= 0:
            raise ValueError("step must be positive")
        self.source_id = source_id or self.path.stem
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.step = step

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "OpenCV is required for video-file input; install naturallab core"
            ) from exc

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video file: {self.path}")

        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = (
            reported_fps
            if math.isfinite(reported_fps) and reported_fps > 0
            else None
        )
        if self.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        frame_index = self.start_frame
        try:
            while self.stop_frame is None or frame_index < self.stop_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame_index - self.start_frame) % self.step == 0:
                    timestamp_ns = None
                    source_timestamp = None
                    timestamp_source = None
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if (
                        math.isfinite(position_ms)
                        and position_ms >= 0
                        and (position_ms > 0 or frame_index == 0)
                    ):
                        source_timestamp = position_ms / 1000.0
                        timestamp_source = "container_pts"
                    elif fps is not None:
                        source_timestamp = frame_index / fps
                        timestamp_source = "synthesized_fps"
                    if source_timestamp is not None:
                        timestamp_ns = round(source_timestamp * 1_000_000_000)
                    yield FramePacket(
                        image=frame,
                        frame_index=frame_index,
                        source_id=self.source_id,
                        timestamp_ns=timestamp_ns,
                        source_timestamp=source_timestamp,
                        metadata={
                            "path": str(self.path),
                            "fps": fps,
                            "color_space": "BGR",
                            "timestamp_source": timestamp_source,
                        },
                    )
                frame_index += 1
        finally:
            capture.release()
