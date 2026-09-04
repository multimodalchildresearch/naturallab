"""Focused regression tests for acquisition configuration and stream metadata."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import runpy
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from naturallab.acquisition import lsl_streams
from naturallab.acquisition import recording_gui
from naturallab.acquisition import xdf_extract


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Button:
    def __init__(self):
        self.states = []

    def config(self, **kwargs):
        self.states.append(kwargs)


def test_recording_gui_module_remains_importable_without_tk(monkeypatch):
    recording_gui_path = Path(recording_gui.__file__)
    real_import = builtins.__import__

    def import_without_tk(name, *args, **kwargs):
        if name == "tkinter":
            raise OSError("libX11.so.6 is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tk)
    namespace = runpy.run_path(
        str(recording_gui_path),
        run_name="naturallab.acquisition._headless_recording_gui",
    )

    assert namespace["tk"] is None
    assert "StorageLocation=" in namespace["build_labrecorder_config"](
        "/tmp/recordings",
        [],
        "acquisition-host",
    )
    with pytest.raises(RuntimeError, match="requires Python Tk support"):
        namespace["_require_tk"]()


def test_legacy_plaintext_password_is_removed_on_load(tmp_path):
    config_path = tmp_path / "recording.json"
    config_path.write_text(
        json.dumps(
            {
                "rtsp_user": "camera-user",
                "rtsp_pass": "legacy-secret",
                "rtsp_stream": "stream1",
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o644)

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.config_file = str(config_path)
    gui._config_migration_notice = ""
    gui._config_security_warning = ""

    loaded = gui.load_config()
    saved_text = config_path.read_text(encoding="utf-8")
    saved = json.loads(saved_text)

    assert loaded["rtsp_user"] == "camera-user"
    assert loaded["delete_after_extract"] is False
    assert "rtsp_pass" not in loaded
    assert "rtsp_pass" not in saved
    assert "legacy-secret" not in saved_text
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert "Removed a legacy plaintext RTSP password" in gui._config_migration_notice


def test_rtsp_credentials_are_encoded_in_launched_urls_and_redacted_from_logs(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "lsl_streams.py"
    script_path.write_text("# test executable path\n", encoding="utf-8")
    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.streaming_active = False
    gui.lsl_process = None
    gui.lsl_script_var = Variable(str(script_path))
    gui.conda_env_var = Variable("")
    gui.caregiver_ip_var = Variable("192.0.2.20")
    gui.child_ip_var = Variable("")
    gui.max_devices_var = Variable(2)
    gui.rtsp_user_var = Variable("lab user@example.org")
    gui.rtsp_pass_var = Variable("p@ss word,#")
    gui.rtsp_stream_var = Variable("main/stream")
    gui.camera_vars = {
        "cam1": {
            "ip": Variable("192.0.2.10:554"),
            "name": Variable("Room camera"),
            "enabled": Variable(True),
        }
    }
    gui.no_realsense_var = Variable(False)
    gui.no_audio_var = Variable(False)
    gui.no_imu_var = Variable(False)
    gui.no_eye_events_var = Variable(False)
    gui.start_lsl_btn = Button()
    gui.stop_lsl_btn = Button()
    gui.read_lsl_output = lambda: None
    messages = []
    gui.log = messages.append

    launched = {}

    class Process:
        pid = 1234

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return Process()

    class Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(recording_gui.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(recording_gui.threading, "Thread", Thread)

    gui.start_lsl_streaming()

    url_argument = launched["command"][
        launched["command"].index("--rtsp-urls") + 1
    ]
    assert url_argument == (
        "rtsp://lab%20user%40example.org:p%40ss%20word%2C%23@"
        "192.0.2.10:554/main/stream"
    )
    combined_log = "\n".join(messages)
    assert "p@ss word,#" not in combined_log
    assert "p%40ss%20word%2C%23" not in combined_log
    assert "lab%20user%40example.org:***@192.0.2.10:554" in combined_log


def test_free_form_process_output_redacts_every_rtsp_password():
    process_line = (
        "python lsl_streams.py --rtsp-urls "
        "rtsp://first:secret-one@camera-1/live,"
        "rtsp://second:secret-two@camera-2/live"
    )

    redacted = recording_gui.redact_log_text(process_line)

    assert "secret-one" not in redacted
    assert "secret-two" not in redacted
    assert "rtsp://first:***@camera-1/live" in redacted
    assert "rtsp://second:***@camera-2/live" in redacted


def test_xdf_stream_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    stream = {
        "info": {
            "name": ["BrokenGaze"],
            "type": ["Gaze"],
            "channel_count": ["2"],
        },
        "time_series": [[0.5, 0.5]],
        "time_stamps": [1.0],
    }
    monkeypatch.setattr(
        xdf_extract,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda _path: ([stream], {})),
    )

    def fail_gaze_extraction(_stream, _output_dir):
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(
        xdf_extract,
        "extract_gaze_stream",
        fail_gaze_extraction,
    )

    with pytest.raises(RuntimeError, match="XDF extraction was incomplete"):
        xdf_extract.extract_streams(
            tmp_path / "recording.xdf",
            tmp_path / "extracted",
        )


def test_save_config_never_persists_rtsp_password(tmp_path):
    config_path = tmp_path / "recording.json"
    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.config_file = str(config_path)
    gui.config = {
        "rtsp_user": "camera-user",
        "rtsp_pass": "must-not-be-written",
        "rtsp_stream": "stream1",
    }
    errors = []
    gui.log = errors.append

    gui.save_config()

    saved_text = config_path.read_text(encoding="utf-8")
    assert "rtsp_pass" not in json.loads(saved_text)
    assert "must-not-be-written" not in saved_text
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert not errors


def test_realsense_lsl_rates_match_configured_capture(monkeypatch):
    created_infos = []
    created_outlets = []

    class Description:
        def append_child_value(self, _key, _value):
            return self

    class StreamInfo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_infos.append(self)

        def desc(self):
            return Description()

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
            return SimpleNamespace(get_depth_scale=lambda: 0.001)

    class Profile:
        def get_device(self):
            return Device()

    class Pipeline:
        def start(self, _config):
            return Profile()

        def stop(self):
            return None

    class Config:
        def __init__(self):
            self.calls = []

        def enable_stream(self, *arguments):
            self.calls.append(arguments)

    config = Config()
    fake_rs = SimpleNamespace(
        pipeline=Pipeline,
        config=lambda: config,
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

    video_rates = {
        info.kwargs["name"]: info.kwargs["nominal_srate"]
        for info in created_infos
        if info.kwargs["name"] in {"RealSense_Color", "RealSense_Depth"}
    }
    assert video_rates == {
        "RealSense_Color": lsl_streams.REALSENSE_FPS,
        "RealSense_Depth": lsl_streams.REALSENSE_FPS,
    }
    assert len(config.calls) == 2
    assert {call[-1] for call in config.calls} == {lsl_streams.REALSENSE_FPS}
    metadata_outlet = next(
        outlet
        for outlet in created_outlets
        if outlet.info.kwargs["name"] == "RealSense_Metadata"
    )
    assert metadata_outlet.samples[0][1] == 123.5


def test_acquisition_sources_use_lsl_clock_domain_and_monotonic_fps():
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        repository_root / "naturallab" / "acquisition" / "lsl_streams.py",
        repository_root / "scripts" / "stream_synchronized_sensors.py",
    )

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        assert "time.time()" not in source, source_path
        assert "pylsl.local_clock()" in source, source_path
        assert "time.monotonic()" in source, source_path


def test_generated_labrecorder_config_uses_supported_key_value_format(tmp_path):
    content = recording_gui.build_labrecorder_config(
        tmp_path / "recordings",
        ["NeonVideo_Child", "NeonGaze_Child", "NeonVideo_Child"],
        "acquisition-host",
    )

    assert content.startswith("; NaturalLab LabRecorder configuration\n")
    assert f"StorageLocation={tmp_path / 'recordings' / 'exp%n' / 'block_%b.xdf'}" in content
    assert 'SessionBlocks="Acceptance","Main"' in content
    assert content.count('"NeonVideo_Child (acquisition-host)"') == 1
    assert '"NeonGaze_Child (acquisition-host)"' in content
    assert "RCSEnabled=1" in content
    assert "RCSPort=22345" in content
    assert "AutoStart=0" in content
    assert "<?xml" not in content
    assert "syncAccuracy" not in content
    assert "enableClockSync" not in content


def test_package_rtsp_stream_uses_reported_or_irregular_rate(monkeypatch):
    created_infos = []
    rates = iter((24.0, 0.0))

    class Description:
        def append_child_value(self, _key, _value):
            return self

    class StreamInfo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_infos.append(self)

        def desc(self):
            return Description()

    class Capture:
        def __init__(self, reported_rate):
            self.reported_rate = reported_rate

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == 3:
                return 1920
            if property_id == 4:
                return 1080
            return self.reported_rate

        def release(self):
            return None

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoCapture=lambda _url: Capture(next(rates)),
    )
    fake_pylsl = SimpleNamespace(
        StreamInfo=StreamInfo,
        StreamOutlet=lambda info: SimpleNamespace(info=info),
    )
    monkeypatch.setattr(lsl_streams, "cv2", fake_cv2)
    monkeypatch.setattr(lsl_streams, "pylsl", fake_pylsl)
    monkeypatch.setattr(lsl_streams, "running", False)

    lsl_streams.stream_rtsp_to_lsl("rtsp://secret@camera/stream", "reported")
    lsl_streams.stream_rtsp_to_lsl("rtsp://secret@camera/stream", "unknown")

    assert [info.kwargs["nominal_srate"] for info in created_infos] == [24.0, 0.0]


def test_root_rtsp_stream_uses_reported_or_irregular_rate(monkeypatch):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stream_synchronized_sensors.py"
    )
    spec = importlib.util.spec_from_file_location("naturallab_root_streamer", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.running = False

    created_infos = []
    rates = iter((29.97, float("nan")))

    class Description:
        def append_child_value(self, _key, _value):
            return self

    class StreamInfo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_infos.append(self)

        def desc(self):
            return Description()

    class Capture:
        def __init__(self, reported_rate):
            self.reported_rate = reported_rate

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == 3:
                return 1280
            if property_id == 4:
                return 720
            return self.reported_rate

        def release(self):
            return None

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoCapture=lambda _url: Capture(next(rates)),
    )
    fake_pylsl = SimpleNamespace(
        StreamInfo=StreamInfo,
        StreamOutlet=lambda info: SimpleNamespace(info=info),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)

    module.stream_rtsp_camera("rtsp://secret@camera/stream", "reported")
    module.stream_rtsp_camera("rtsp://secret@camera/stream", "unknown")

    assert [info.kwargs["nominal_srate"] for info in created_infos] == [29.97, 0.0]


def test_package_rtsp_url_default_is_empty():
    source = Path(lsl_streams.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--rtsp-urls":
            continue
        defaults.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "default"
        )

    assert len(defaults) == 1
    assert ast.literal_eval(defaults[0]) == ""
