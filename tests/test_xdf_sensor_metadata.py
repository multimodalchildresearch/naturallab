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


def _channel_description(labels: list[str]) -> list[dict]:
    return [
        {
            "channels": [
                {
                    "channel": [
                        {"label": [label]}
                        for label in labels
                    ]
                }
            ]
        }
    ]


def _numeric_stream(
    stream_type: str,
    name: str,
    labels: list[str],
    samples,
    timestamps,
) -> dict:
    return {
        "info": {
            "name": [name],
            "type": [stream_type],
            "channel_count": [str(len(labels))],
            "desc": _channel_description(labels),
        },
        "time_series": samples,
        "time_stamps": timestamps,
    }


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
    def __init__(self, path, *_args, **_kwargs):
        self.frames = []
        Path(path).write_bytes(b"fake-video")

    def write(self, frame):
        self.frames.append(frame)

    def isOpened(self):
        return True

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

    child_path = output_dir / "neonimu_child.csv"
    caregiver_path = output_dir / "neonimu_caregiver.csv"
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
    assert "NeonIMU_Child -> neonimu_child.csv" in summary
    assert "NeonIMU_Caregiver -> neonimu_caregiver.csv" in summary


def test_single_imu_stream_uses_its_safe_deterministic_stem(
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

    assert (output_dir / "neonimu_child.csv").is_file()
    assert not (output_dir / "imu.csv").exists()


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

    assert not output_dir.exists()
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

    assert not output_dir.exists()
    assert "All streams extracted" not in capsys.readouterr().out


def test_failed_extraction_preserves_an_existing_empty_target(
    tmp_path,
    monkeypatch,
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
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="IMU timestamp/sample mismatch"):
        xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not list(tmp_path.glob(".extracted.staging-*"))


def test_stream_names_are_sanitized_without_path_traversal(
    tmp_path,
    monkeypatch,
    capsys,
):
    stream = _numeric_stream(
        "Markers",
        "../../Outside Results",
        ["marker"],
        [[7]],
        [1.0],
    )
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: ([stream], {})),
    )

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "outside_results.csv"
    }
    assert not (tmp_path / "Outside Results.csv").exists()
    assert "../../Outside Results" in capsys.readouterr().out


@pytest.mark.parametrize(
    "streams",
    [
        [
            _numeric_stream("Markers", "Room/Camera", ["value"], [[1]], [1.0]),
            _numeric_stream("Markers", "Room Camera", ["value"], [[2]], [2.0]),
        ],
        [
            _numeric_stream("VideoStream", "Camera", ["frame"], [[1]], [1.0]),
            _numeric_stream(
                "Markers",
                "Camera timestamps",
                ["value"],
                [[2]],
                [2.0],
            ),
        ],
    ],
)
def test_ambiguous_stream_output_names_are_rejected_before_writing(
    tmp_path,
    monkeypatch,
    streams,
):
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )
    output_dir = tmp_path / "extracted"

    with pytest.raises(RuntimeError, match="ambiguous XDF stream"):
        xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".extracted.staging-*"))


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
    monkeypatch.setattr(
        xdf_extract,
        "_verify_video_file",
        lambda *_args, **_kwargs: None,
    )

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(
        tmp_path / "recording.xdf",
        output_dir,
        depth_interval=1,
        include_csv=True,
    )

    distance = np.loadtxt(
        output_dir / "realsense_depth_depth" / "distance_000000.csv",
        delimiter=",",
    )
    assert distance[0, 1] == pytest.approx(0.25)
    assert distance[1, 1] == pytest.approx(1.0)

    depth_metadata = json.loads(
        (output_dir / "realsense_depth_depth_metadata.json").read_text(
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

    assert not output_dir.exists()


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


def test_depth_extraction_does_not_skip_a_corrupt_frame(
    tmp_path,
    monkeypatch,
):
    stream = _depth_stream(embedded_scale=0.0025)
    stream["time_series"].append(["not-base64"])
    stream["time_stamps"] = np.array([10.0, 10.1])
    monkeypatch.setattr(cv2, "VideoWriter", _VideoWriter)

    with pytest.raises(RuntimeError, match="could not decode depth frame 1"):
        xdf_extract.extract_depth_stream(
            stream,
            tmp_path,
            save_interval=1,
            include_csv=True,
        )

    assert not (tmp_path / "realsense_depth_timestamps.csv").exists()
    assert not (tmp_path / "realsense_depth_depth_metadata.json").exists()
    assert not (tmp_path / "realsense_depth_depth").exists()
    assert list(tmp_path.iterdir()) == []


def test_depth_extraction_fails_when_raw_png_write_fails(
    tmp_path,
    monkeypatch,
):
    stream = _depth_stream(embedded_scale=0.0025)
    monkeypatch.setattr(cv2, "VideoWriter", _VideoWriter)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="could not write raw depth frame 0"):
        xdf_extract.extract_depth_stream(
            stream,
            tmp_path,
            save_interval=1,
        )

    assert list(tmp_path.iterdir()) == []


def test_depth_extraction_fails_when_metric_csv_write_fails(
    tmp_path,
    monkeypatch,
):
    stream = _depth_stream(embedded_scale=0.0025)
    monkeypatch.setattr(cv2, "VideoWriter", _VideoWriter)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: True)

    def fail_metric_write(*_args, **_kwargs):
        raise OSError("fixture disk failure")

    monkeypatch.setattr(np, "savetxt", fail_metric_write)

    with pytest.raises(OSError, match="fixture disk failure"):
        xdf_extract.extract_depth_stream(
            stream,
            tmp_path,
            save_interval=1,
            include_csv=True,
        )

    assert list(tmp_path.iterdir()) == []


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
            "desc": _channel_description(
                [f"recorded_gaze_channel_{index}" for index in range(16)]
            ),
        },
        "time_series": np.zeros((2, 16), dtype=np.float64),
        "time_stamps": np.array([12.0, 12.1]),
    }

    xdf_extract.extract_gaze_stream(stream, tmp_path)

    gaze = pd.read_csv(tmp_path / "neongaze_child.csv")
    assert gaze["timestamp"].tolist() == [12.0, 12.1]
    assert set(gaze["timestamp_domain"]) == {"lsl"}
    assert "datetime" not in gaze.columns
    assert "lsl_relative_timestamp" not in gaze.columns


def test_gaze_extraction_preserves_recorded_indices_and_requires_labels(tmp_path):
    labels = [
        "frame_index",
        "gaze_x",
        "gaze_y",
        "pupil_diameter_left",
        "pupil_diameter_right",
    ]
    stream = _numeric_stream(
        "Gaze",
        "NeonGaze_Child",
        labels,
        [[42, 10, 20, 3, 4], [44, 11, 21, 3.1, 4.1]],
        [100.0, 100.1],
    )

    xdf_extract.extract_gaze_stream(stream, tmp_path)

    gaze = pd.read_csv(tmp_path / "neongaze_child.csv")
    assert gaze["frame_index"].tolist() == [42.0, 44.0]
    assert gaze["timestamp"].tolist() == [100.0, 100.1]
    assert "original_frame_index" not in gaze.columns
    assert "datetime" not in gaze.columns

    stream["info"].pop("desc")
    with pytest.raises(RuntimeError, match="channel-label group"):
        xdf_extract.extract_gaze_stream(stream, tmp_path / "missing-labels")


@pytest.mark.parametrize(
    ("samples", "timestamps", "message"),
    [
        ([], [], "no samples"),
        ([[1.0, 2.0], [3.0]], [1.0, 2.0], "rectangular numeric"),
        ([[1.0, 2.0]], [1.0, 2.0], "timestamp/sample mismatch"),
        ([[1.0, float("nan")]], [1.0], "non-finite samples"),
        ([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0], "strictly increasing"),
    ],
)
def test_declared_gaze_rejects_incomplete_or_ambiguous_data(
    tmp_path,
    samples,
    timestamps,
    message,
):
    stream = _numeric_stream(
        "Gaze",
        "Gaze",
        ["x", "y"],
        samples,
        timestamps,
    )

    with pytest.raises(RuntimeError, match=message):
        xdf_extract.extract_gaze_stream(stream, tmp_path)

    assert not (tmp_path / "gaze.csv").exists()
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.npy"))


def test_eye_event_extraction_preserves_sensor_time_and_optional_nan(tmp_path):
    fixation_labels = [
        "fixation_id",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "duration_ms",
        "fixation_x_px",
        "fixation_y_px",
        "azimuth_deg",
        "elevation_deg",
    ]
    fixation = _numeric_stream(
        "Fixations",
        "NeonFixations_Child",
        fixation_labels,
        [[7, 1_000_000_000, 1_100_000_000, 100, np.nan, 20, np.nan, 2]],
        [50.25],
    )
    saccade_labels = [
        "saccade_id",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "amplitude_deg",
        "amplitude_px",
        "mean_velocity_px_s",
        "peak_velocity_px_s",
        "duration_ms",
    ]
    saccade = _numeric_stream(
        "Saccades",
        "NeonSaccades_Child",
        saccade_labels,
        [[9, 2_000_000_000, 2_050_000_000, np.nan, 30, np.nan, 70, 50]],
        [51.25],
    )

    xdf_extract.extract_fixations_stream(fixation, tmp_path)
    xdf_extract.extract_saccades_stream(saccade, tmp_path)

    fixations = pd.read_csv(tmp_path / "neonfixations_child.csv")
    saccades = pd.read_csv(tmp_path / "neonsaccades_child.csv")
    assert fixations["start_timestamp_ns"].iloc[0] == 1_000_000_000
    assert saccades["start_timestamp_ns"].iloc[0] == 2_000_000_000
    assert np.isnan(fixations["fixation_x_px"].iloc[0])
    assert np.isnan(saccades["amplitude_deg"].iloc[0])
    assert fixations["timestamp"].iloc[0] == 50.25
    assert saccades["timestamp"].iloc[0] == 51.25
    assert set(fixations["timestamp_domain"]) == {"lsl"}
    assert set(saccades["timestamp_domain"]) == {"lsl"}
    for forbidden in ("section_id", "recording_id", "detected_datetime"):
        assert forbidden not in fixations.columns
        assert forbidden not in saccades.columns


def test_multiple_eye_event_streams_receive_distinct_safe_files(
    tmp_path,
    monkeypatch,
):
    fixation_labels = [
        "fixation_id",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "duration_ms",
    ]
    saccade_labels = [
        "saccade_id",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "duration_ms",
    ]
    streams = [
        _numeric_stream(
            "Fixations",
            "NeonFixations/Child",
            fixation_labels,
            [[1, 10, 20, 10]],
            [1.0],
        ),
        _numeric_stream(
            "Fixations",
            "NeonFixations Caregiver",
            fixation_labels,
            [[2, 30, 40, 10]],
            [2.0],
        ),
        _numeric_stream(
            "Saccades",
            "NeonSaccades/Child",
            saccade_labels,
            [[3, 50, 60, 10]],
            [3.0],
        ),
        _numeric_stream(
            "Saccades",
            "NeonSaccades Caregiver",
            saccade_labels,
            [[4, 70, 80, 10]],
            [4.0],
        ),
    ]
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: (streams, {})),
    )

    output_dir = tmp_path / "extracted"
    xdf_extract.extract_streams(tmp_path / "recording.xdf", output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "neonfixations_child.csv",
        "neonfixations_caregiver.csv",
        "neonsaccades_child.csv",
        "neonsaccades_caregiver.csv",
    }
    assert pd.read_csv(output_dir / "neonfixations_child.csv")[
        "fixation_id"
    ].tolist() == [1.0]
    assert pd.read_csv(output_dir / "neonfixations_caregiver.csv")[
        "fixation_id"
    ].tolist() == [2.0]


def test_eye_event_extraction_rejects_missing_required_fields_without_fallback(
    tmp_path,
):
    stream = _numeric_stream(
        "Fixations",
        "NeonFixations_Child",
        [
            "fixation_id",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "duration_ms",
            "fixation_x_px",
            "fixation_y_px",
            "azimuth_deg",
            "elevation_deg",
        ],
        [[1, 10, 20, float("nan"), 3, 4, 5, 6]],
        [1.0],
    )

    with pytest.raises(RuntimeError, match="duration_ms.*non-finite"):
        xdf_extract.extract_fixations_stream(stream, tmp_path)

    assert not (tmp_path / "neonfixations_child.csv").exists()
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.npy"))


def test_generic_extraction_requires_explicit_labels_and_strict_counts(tmp_path):
    stream = _numeric_stream(
        "Markers",
        "TaskMarkers",
        ["code", "trial"],
        [[1, 4], [2, 4]],
        [5.0, 5.5],
    )
    output = xdf_extract.extract_generic_stream(stream, tmp_path)
    exported = pd.read_csv(output)
    assert exported.columns.tolist() == [
        "code",
        "trial",
        "timestamp",
        "timestamp_domain",
    ]
    assert set(exported["timestamp_domain"]) == {"lsl"}

    stream["info"].pop("desc")
    with pytest.raises(RuntimeError, match="channel-label group"):
        xdf_extract.extract_generic_stream(stream, tmp_path / "no-labels")


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
    timestamps = pd.read_csv(tmp_path / "neonaudio_child_timestamps.csv")
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
