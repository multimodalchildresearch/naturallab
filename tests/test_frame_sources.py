from pathlib import Path

import pytest

from naturallab.media import (
    FramePacket,
    ImageDirectorySource,
    IterableFrameSource,
    VideoFileSource,
)


def test_frame_packet_rejects_invalid_identity_and_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        FramePacket(image=object(), frame_index=-1, source_id="camera-1")
    with pytest.raises(ValueError, match="source_id"):
        FramePacket(image=object(), frame_index=0, source_id=" ")
    for timestamp in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="source_timestamp"):
            FramePacket(
                image=object(),
                frame_index=0,
                source_id="camera-1",
                source_timestamp=timestamp,
            )


def test_iterable_source_assigns_stable_indices_and_timestamps() -> None:
    images = [object(), object(), object()]
    packets = list(
        IterableFrameSource(
            images,
            source_id="ceiling-01",
            fps=25.0,
            start_timestamp_ns=1_000,
        )
    )

    assert [packet.frame_index for packet in packets] == [0, 1, 2]
    assert [packet.source_id for packet in packets] == ["ceiling-01"] * 3
    assert [packet.timestamp_ns for packet in packets] == [
        1_000,
        40_001_000,
        80_001_000,
    ]
    assert [packet.source_timestamp for packet in packets] == [
        0.0,
        0.04,
        0.08,
    ]
    assert {
        packet.metadata["timestamp_source"] for packet in packets
    } == {"synthesized_fps"}
    assert [packet.image for packet in packets] == images


def test_iterable_source_preserves_existing_packets() -> None:
    packet = FramePacket(
        image="frame",
        frame_index=17,
        source_id="external-source",
        timestamp_ns=123,
    )
    assert list(IterableFrameSource([packet], source_id="ignored")) == [packet]


def test_video_source_validates_before_importing_opencv(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        VideoFileSource(tmp_path / "missing.mp4")
    video_path = tmp_path / "placeholder.mp4"
    video_path.write_bytes(b"not a real video")
    with pytest.raises(ValueError, match="step"):
        VideoFileSource(video_path, step=0)


@pytest.mark.parametrize(
    "fps",
    [0, -1, float("nan"), float("inf"), float("-inf"), True, "30"],
)
def test_iterable_source_rejects_invalid_fps(fps) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        IterableFrameSource([], fps=fps)


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), True])
def test_image_directory_source_rejects_invalid_fps(
    tmp_path: Path,
    fps,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        ImageDirectorySource(tmp_path, fps=fps)


def test_video_source_falls_back_when_position_timestamp_is_zero_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import numpy as np

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.zeros((2, 3, 3), dtype=np.uint8),
                np.zeros((2, 3, 3), dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 25.0
            if property_id == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return 0.0

        def set(self, property_id: int, value: float) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())
    video_path = tmp_path / "placeholder.mp4"
    video_path.write_bytes(b"placeholder")

    packets = list(VideoFileSource(video_path))

    assert [packet.source_timestamp for packet in packets] == [0.0, 0.04]
    assert [
        packet.metadata["timestamp_source"] for packet in packets
    ] == ["container_pts", "synthesized_fps"]


def test_image_directory_uses_natural_numeric_order(tmp_path: Path) -> None:
    for name in ("frame10.jpg", "frame2.jpg", "frame1.jpg"):
        (tmp_path / name).write_bytes(b"placeholder")

    source = ImageDirectorySource(tmp_path)

    assert [path.name for path in source.paths] == [
        "frame1.jpg",
        "frame2.jpg",
        "frame10.jpg",
    ]
