"""Regression tests for timing-safe Neon audio acquisition."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from naturallab.acquisition import lsl_streams


class FakeDescription:
    def __init__(self):
        self.values = {}

    def append_child_value(self, key, value):
        self.values[key] = value
        return self


class FakeStreamInfo:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.description = FakeDescription()

    def desc(self):
        return self.description


class FakeStreamOutlet:
    def __init__(self, info):
        self.info = info
        self.chunks = []

    def push_chunk(self, samples, timestamp):
        self.chunks.append((samples, timestamp))


class FakeDecodedAudioFrame:
    def __init__(self, *, sample_rate=48_000, channels=2, samples=4):
        self.sample_rate = sample_rate
        self.samples = samples
        self.layout = SimpleNamespace(channels=tuple(range(channels)))
        self.format = SimpleNamespace(name="s16p")
        self._samples = np.arange(
            channels * samples,
            dtype=np.int16,
        ).reshape(channels, samples)

    def to_ndarray(self):
        return self._samples


def _audio_frame(timestamp, **kwargs):
    return SimpleNamespace(
        av_frame=FakeDecodedAudioFrame(**kwargs),
        timestamp_unix_seconds=timestamp,
    )


class FakeDevice:
    def __init__(self, frames, estimates=None):
        self.frames = list(frames)
        self.time_echo_calls = []
        self.estimates = estimates or SimpleNamespace(
            time_offset_ms=SimpleNamespace(mean=10.0, std=0.5),
            roundtrip_duration_ms=SimpleNamespace(mean=2.0, std=0.25),
        )

    def estimate_time_offset(self, **kwargs):
        self.time_echo_calls.append(kwargs)
        return self.estimates

    def receive_audio_frame(self, timeout_seconds):
        assert timeout_seconds == lsl_streams.AUDIO_READ_TIMEOUT_SECONDS
        return self.frames.pop(0) if self.frames else None


def install_fake_audio_runtime(monkeypatch):
    outlets = []

    def create_outlet(info):
        outlet = FakeStreamOutlet(info)
        outlets.append(outlet)
        return outlet

    fake_pylsl = SimpleNamespace(
        StreamInfo=FakeStreamInfo,
        StreamOutlet=create_outlet,
        local_clock=lambda: 100.0,
    )
    monkeypatch.setattr(lsl_streams, "pylsl", fake_pylsl)
    monkeypatch.setattr(lsl_streams.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(lsl_streams, "running", True)
    return outlets


def test_audio_outlet_uses_source_format_rtcp_time_and_time_echo(monkeypatch):
    outlets = install_fake_audio_runtime(monkeypatch)
    device = FakeDevice(
        [
            _audio_frame(1_000.0),
            _audio_frame(1_000.0 + 4 / 48_000),
        ]
    )

    prepared = lsl_streams._prepare_audio_stream(
        device,
        "NeonAudio_Child",
    )

    assert device.time_echo_calls == [
        {"number_of_measurements": lsl_streams.AUDIO_TIME_ECHO_MEASUREMENTS}
    ]
    assert prepared.sample_rate == 48_000
    assert prepared.channel_count == 2
    assert len(outlets) == 1
    info = outlets[0].info
    assert info.kwargs["nominal_srate"] == 48_000
    assert info.kwargs["channel_count"] == 2
    assert "RTCP absolute timestamps" in info.description.values["source_transport"]
    assert info.description.values["time_echo_offset_mean_ms"] == "10.0"
    assert (
        info.description.values["timeline_discontinuity_policy"]
        == "stop stream and report an error"
    )

    with pytest.raises(RuntimeError, match="timed out waiting"):
        lsl_streams._push_prepared_audio_stream(prepared)

    assert len(outlets[0].chunks) == 2
    assert np.asarray(outlets[0].chunks[0][0]).shape == (4, 2)
    assert outlets[0].chunks[0][1] == pytest.approx(
        100.0 + 0.010 + 3 / 48_000
    )
    assert outlets[0].chunks[1][1] == pytest.approx(
        100.0 + 0.010 + 7 / 48_000
    )


def test_audio_refuses_missing_source_timestamps_before_advertising(monkeypatch):
    outlets = install_fake_audio_runtime(monkeypatch)
    device = FakeDevice([_audio_frame(None)])

    with pytest.raises(RuntimeError, match="no absolute source timestamp"):
        lsl_streams._prepare_audio_stream(
            device,
            "NeonAudio_Caregiver",
        )

    assert outlets == []


def test_audio_refuses_missing_time_echo_before_advertising(monkeypatch):
    outlets = install_fake_audio_runtime(monkeypatch)
    device = FakeDevice([_audio_frame(1_000.0)])
    device.estimates = None

    with pytest.raises(RuntimeError, match="did not produce"):
        lsl_streams._prepare_audio_stream(
            device,
            "NeonAudio_Caregiver",
        )

    assert outlets == []
    assert len(device.frames) == 1


def test_audio_stops_instead_of_hiding_a_source_timeline_gap(monkeypatch):
    outlets = install_fake_audio_runtime(monkeypatch)
    device = FakeDevice(
        [
            _audio_frame(1_000.0),
            _audio_frame(1_000.0 + 480 / 48_000),
        ]
    )
    prepared = lsl_streams._prepare_audio_stream(
        device,
        "NeonAudio_Child",
    )

    with pytest.raises(RuntimeError, match="timeline discontinuity"):
        lsl_streams._push_prepared_audio_stream(prepared)

    assert len(outlets[0].chunks) == 1


def test_raw_rtsp_audio_and_vlc_fallback_are_removed():
    source = Path(lsl_streams.__file__).read_text(encoding="utf-8")

    assert "--audio-method" not in source
    assert "start_audio_stream_to_lsl_pyav" not in source
    assert "av.open" not in source
    assert "queue.Queue" not in source
    assert "VLC" not in source


def test_main_does_not_claim_audio_that_failed_validation(monkeypatch, capsys):
    class Device:
        def __init__(self, address, port):
            self.address = address
            self.phone_ip = address
            self.phone_name = "Test Neon"
            self.port = port

    simple_module = SimpleNamespace(Device=Device)
    monkeypatch.setitem(sys.modules, "pupil_labs.realtime_api.simple", simple_module)
    monkeypatch.setattr(lsl_streams, "REALTIME_API_AVAILABLE", True)
    monkeypatch.setattr(lsl_streams, "running", False)
    monkeypatch.setattr(lsl_streams, "pylsl", object())
    monkeypatch.setattr(lsl_streams, "stream_neon_api_to_lsl", lambda *_args: None)
    monkeypatch.setattr(
        lsl_streams,
        "start_audio_stream_to_lsl",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--caregiver-ip",
            "192.0.2.20",
            "--no-realsense",
            "--no-imu",
            "--no-eye-events",
        ],
    )

    assert lsl_streams.main() == 1
    captured = capsys.readouterr()
    assert "NeonAudio_Caregiver" not in captured.out
    assert "Audio did not start for Caregiver" in captured.err
    assert "Fix audio or rerun with --no-audio" in captured.err


def test_main_fails_if_audio_requested_but_realtime_api_unavailable(
    monkeypatch,
    capsys,
):
    class FakeThread:
        daemon = False

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(lsl_streams, "REALTIME_API_AVAILABLE", False)
    monkeypatch.setattr(lsl_streams, "pylsl", object())
    monkeypatch.setattr(lsl_streams.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--caregiver-ip",
            "192.0.2.20",
            "--no-realsense",
            "--no-imu",
            "--no-eye-events",
            "--rtsp-urls",
            "rtsp://192.0.2.30/stream",
        ],
    )

    assert lsl_streams.main() == 1
    captured = capsys.readouterr()
    assert "NeonAudio_Caregiver" in captured.err
    assert "no acquisition source started" not in captured.err
