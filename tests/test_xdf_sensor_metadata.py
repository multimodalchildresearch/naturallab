"""Regression tests for identity-safe IMU and metric depth extraction."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from naturallab.acquisition import lsl_streams, xdf_extract


def _imu_stream(name: str, marker: float) -> dict:
    samples = np.zeros((2, 13), dtype=np.float64)
    samples[:, 0] = marker
    return {
        "info": {
            "name": [name],
            "type": ["IMU"],
            "channel_count": ["13"],
        },
        "time_series": samples,
        "time_stamps": np.array([1.0, 1.1]),
    }


def _depth_stream(*, embedded_scale=None, name="RealSense_Depth") -> dict:
    raw_depth = np.array([[0, 100], [200, 400]], dtype=np.uint16)
    encoded_ok, encoded = cv2.imencode(".png", raw_depth)
    assert encoded_ok
    info = {
        "name": [name],
        "type": ["Depth"],
        "channel_count": ["1"],
    }
    if embedded_scale is not None:
        info["desc"] = [
            {"depth_scale_m_per_unit": [str(embedded_scale)]}
        ]
    return {
        "info": info,
        "time_series": [
            [base64.b64encode(encoded.tobytes()).decode("ascii")]
        ],
        "time_stamps": np.array([10.0]),
    }


def _device_metadata_stream(scale: float) -> dict:
    return {
        "info": {
            "name": ["RealSense_Metadata"],
            "type": ["DeviceInfo"],
            "channel_count": ["1"],
        },
        "time_series": [[json.dumps({"depth_scale": scale})]],
        "time_stamps": np.array([10.0]),
    }


class _VideoWriter:
    def __init__(self, *_args, **_kwargs):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        return None


def test_multiple_imu_streams_use_distinct_deterministic_files(
    tmp_path,
    monkeypatch,
    capsys,
):
    streams = [
        _imu_stream("NeonIMU_Child", 11.0),
        _imu_stream("NeonIMU_Caregiver", 22.0),
    ]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    child_path = output_dir / "imu_neonimu_child.csv"
    caregiver_path = output_dir / "imu_neonimu_caregiver.csv"
    assert child_path.is_file()
    assert caregiver_path.is_file()
    assert not (output_dir / "imu.csv").exists()
    child = pd.read_csv(child_path)
    caregiver = pd.read_csv(caregiver_path)
    assert child.loc[0, "gyro_x [deg/s]"] == 11.0
    assert caregiver.loc[0, "gyro_x [deg/s]"] == 22.0
    assert child["timestamp"].tolist() == [1.0, 1.1]
    assert caregiver["timestamp"].tolist() == [1.0, 1.1]
    assert set(child["timestamp_domain"]) == {"lsl"}
    assert set(caregiver["timestamp_domain"]) == {"lsl"}
    assert "timestamp [ns]" not in child.columns
    assert "datetime" not in child.columns
    summary = capsys.readouterr().out
    assert "NeonIMU_Child -> imu_neonimu_child.csv" in summary
    assert "NeonIMU_Caregiver -> imu_neonimu_caregiver.csv" in summary


def test_single_imu_stream_retains_unambiguous_legacy_filename(
    tmp_path,
    monkeypatch,
):
    streams = [_imu_stream("NeonIMU_Child", 11.0)]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert (output_dir / "imu.csv").is_file()
    assert not (output_dir / "imu_neonimu_child.csv").exists()


def test_multi_role_imu_extraction_fails_if_one_role_is_empty(
    tmp_path,
    monkeypatch,
    capsys,
):
    child = _imu_stream("NeonIMU_Child", 11.0)
    caregiver = _imu_stream("NeonIMU_Caregiver", 22.0)
    caregiver["time_series"] = np.empty((0, 13), dtype=np.float64)
    caregiver["time_stamps"] = np.empty(0, dtype=np.float64)
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: ([child, caregiver], {})),
    )

    output_dir = tmp_path / "extracted"
    with pytest.raises(RuntimeError, match="no IMU samples"):
        xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert (output_dir / "imu_neonimu_child.csv").is_file()
    assert not (output_dir / "imu_neonimu_caregiver.csv").exists()
    assert not list(output_dir.glob("NeonIMU_Caregiver_*"))
    assert "All streams extracted" not in capsys.readouterr().out


def test_multi_role_imu_extraction_fails_on_timestamp_mismatch(
    tmp_path,
    monkeypatch,
    capsys,
):
    child = _imu_stream("NeonIMU_Child", 11.0)
    caregiver = _imu_stream("NeonIMU_Caregiver", 22.0)
    caregiver["time_stamps"] = np.array([1.0])
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: ([child, caregiver], {})),
    )

    output_dir = tmp_path / "extracted"
    with pytest.raises(RuntimeError, match="IMU timestamp/sample mismatch"):
        xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert (output_dir / "imu_neonimu_child.csv").is_file()
    assert not (output_dir / "imu_neonimu_caregiver.csv").exists()
    assert not list(output_dir.glob("NeonIMU_Caregiver_*"))
    assert "All streams extracted" not in capsys.readouterr().out


def test_imu_extraction_rejects_malformed_samples_without_fallbacks(tmp_path):
    stream = _imu_stream("NeonIMU_Child", 11.0)
    stream["time_series"] = [["not-a-number"] * 13] * 2

    with pytest.raises(RuntimeError, match="rectangular numeric array"):
        xdf_extract.extract_imu_stream(stream, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_depth_extraction_uses_recorded_deviceinfo_scale(
    tmp_path,
    monkeypatch,
    capsys,
):
    streams = [_depth_stream(), _device_metadata_stream(0.0025)]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )
    monkeypatch.setattr(cv2, "VideoWriter", _VideoWriter)

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(
        tmp_path / "recording.xdf",
        output_dir,
        depth_interval=1,
        include_csv=True,
    )

    distance = np.loadtxt(
        output_dir / "RealSense_Depth_depth" / "distance_000000.csv",
        delimiter=",",
    )
    assert distance[0, 1] == pytest.approx(0.25)
    assert distance[1, 1] == pytest.approx(1.0)

    depth_metadata = json.loads(
        (output_dir / "RealSense_Depth_depth_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert depth_metadata["raw_value_unit"] == "device_depth_unit"
    assert depth_metadata["depth_scale_m_per_unit"] == 0.0025
    assert depth_metadata["metric_distance_unit"] == "metre"
    assert depth_metadata["distance_csv_unit"] == "metre"
    output = capsys.readouterr().out
    assert "0.0025 metres per raw device unit" in output
    assert "RealSense_Depth (0.0025 m/unit)" in output


def test_depth_extraction_fails_without_recorded_scale(
    tmp_path,
    monkeypatch,
):
    streams = [_depth_stream()]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )

    output_dir = tmp_path / "extracted"
    with pytest.raises(RuntimeError, match="no recorded depth scale"):
        xdf_extract.extract_streams(
            tmp_path / "recording.xdf",
            output_dir,
        )

    assert not (output_dir / "RealSense_Depth_depth").exists()
    assert not (
        output_dir / "RealSense_Depth_depth_metadata.json"
    ).exists()


def test_depth_extraction_rejects_conflicting_recorded_scales(
    tmp_path,
    monkeypatch,
):
    streams = [
        _depth_stream(embedded_scale=0.001),
        _device_metadata_stream(0.002),
    ]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )

    with pytest.raises(RuntimeError, match="conflicting depth scales"):
        xdf_extract.extract_streams(
            tmp_path / "recording.xdf",
            tmp_path / "extracted",
        )


def test_depth_scale_is_not_borrowed_by_an_unmatched_second_sensor():
    realsense_depth = _depth_stream(name="RealSense_Depth")
    other_depth = _depth_stream(name="OtherSensor_Depth")
    metadata = _device_metadata_stream(0.0025)

    scale, source = xdf_extract._resolve_depth_scale(
        realsense_depth,
        [realsense_depth, other_depth, metadata],
    )
    assert scale == 0.0025
    assert "RealSense_Metadata" in source

    with pytest.raises(RuntimeError, match="cannot associate"):
        xdf_extract._resolve_depth_scale(
            other_depth,
            [realsense_depth, other_depth, metadata],
        )


def test_xdf_extraction_rejects_a_nonempty_output_directory(
    tmp_path,
    monkeypatch,
):
    streams = [_imu_stream("NeonIMU_Child", 11.0)]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )
    output_dir = tmp_path / "extracted"
    output_dir.mkdir()
    stale_file = output_dir / "imu.csv"
    stale_file.write_text("stale-result\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output directory is not empty"):
        xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert stale_file.read_text(encoding="utf-8") == "stale-result\n"


def test_gaze_extraction_preserves_lsl_clock_and_ignores_stale_imu(tmp_path):
    pd.DataFrame({"timestamp [ns]": [1_700_000_000_000_000_000]}).to_csv(
        tmp_path / "imu.csv",
        index=False,
    )
    stream = {
        "info": {
            "name": ["NeonGaze_Child"],
            "type": ["Gaze"],
            "channel_count": ["16"],
        },
        "time_series": np.zeros((2, 16), dtype=np.float64),
        "time_stamps": np.array([12.0, 12.1]),
    }

    xdf_extract.extract_gaze_stream(stream, tmp_path)

    gaze = pd.read_csv(tmp_path / "NeonGaze_Child.csv")
    assert gaze["timestamp"].tolist() == [12.0, 12.1]
    assert set(gaze["timestamp_domain"]) == {"lsl"}
    assert "datetime" not in gaze.columns
    assert "lsl_relative_timestamp" not in gaze.columns


def test_audio_extraction_preserves_sample_alignment_and_lsl_clock(
    tmp_path,
    monkeypatch,
):
    def fake_write(path, data, sample_rate):
        assert np.asarray(data).shape == (3, 2)
        assert sample_rate == 8_000
        Path(path).write_bytes(b"fake-wave")

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=fake_write))
    stream = {
        "info": {
            "name": ["NeonAudio_Child"],
            "type": ["Audio"],
            "channel_count": ["2"],
            "nominal_srate": ["8000"],
        },
        "time_series": np.array(
            [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
            dtype=np.float32,
        ),
        "time_stamps": np.array([20.0, 20.000125, 20.00025]),
    }

    output = xdf_extract.extract_audio_stream(stream, tmp_path)

    assert Path(output).read_bytes() == b"fake-wave"
    timestamps = pd.read_csv(tmp_path / "NeonAudio_Child_timestamps.csv")
    assert timestamps["sample_index"].tolist() == [0, 1, 2]
    assert timestamps["timestamp"].tolist() == [20.0, 20.000125, 20.00025]
    assert set(timestamps["timestamp_domain"]) == {"lsl"}
    assert "datetime" not in timestamps.columns


@pytest.mark.parametrize("scale", [0, -0.001, float("nan"), float("inf")])
def test_depth_scale_must_be_finite_and_positive(scale):
    depth_stream = _depth_stream(embedded_scale=scale)

    with pytest.raises(RuntimeError, match="finite positive"):
        xdf_extract._resolve_depth_scale(depth_stream, [depth_stream])


def test_realsense_depth_stream_records_hardware_scale(monkeypatch):
    created_infos = []
    created_outlets = []

    class Description:
        def __init__(self):
            self.values = {}

        def append_child_value(self, key, value):
            self.values[key] = value
            return self

    class StreamInfo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.description = Description()
            created_infos.append(self)

        def desc(self):
            return self.description

    class StreamOutlet:
        def __init__(self, info):
            self.info = info
            self.samples = []
            created_outlets.append(self)

        def push_sample(self, sample, timestamp=None):
            self.samples.append((sample, timestamp))

    class Device:
        def get_info(self, key):
            return {"name": "Test RealSense", "serial": "0001"}[key]

        def first_depth_sensor(self):
            return SimpleNamespace(get_depth_scale=lambda: 0.0025)

    class Pipeline:
        def start(self, _config):
            return SimpleNamespace(get_device=lambda: Device())

        def stop(self):
            return None

    fake_rs = SimpleNamespace(
        pipeline=Pipeline,
        config=lambda: SimpleNamespace(enable_stream=lambda *_args: None),
        align=lambda _stream: object(),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        camera_info=SimpleNamespace(name="name", serial_number="serial"),
    )
    fake_pylsl = SimpleNamespace(
        StreamInfo=StreamInfo,
        StreamOutlet=StreamOutlet,
        local_clock=lambda: 123.5,
    )
    monkeypatch.setattr(lsl_streams, "rs", fake_rs)
    monkeypatch.setattr(lsl_streams, "pylsl", fake_pylsl)
    monkeypatch.setattr(lsl_streams, "REALSENSE_AVAILABLE", True)
    monkeypatch.setattr(lsl_streams, "running", False)

    lsl_streams.stream_realsense_to_lsl()

    depth_info = next(
        info
        for info in created_infos
        if info.kwargs["name"] == "RealSense_Depth"
    )
    assert depth_info.description.values == {
        "content": "raw_depth",
        "depth_format": "uint16_device_units",
        "depth_scale_m_per_unit": "0.0025",
        "metric_unit": "metre",
    }
    metadata_outlet = next(
        outlet
        for outlet in created_outlets
        if outlet.info.kwargs["name"] == "RealSense_Metadata"
    )
    metadata = json.loads(metadata_outlet.samples[0][0][0])
    assert metadata["depth_scale"] == 0.0025
    assert metadata["depth_scale_m_per_unit"] == 0.0025
    assert metadata["raw_depth_unit"] == "device_depth_unit"
    assert metadata["metric_unit"] == "metre"
