"""Focused regression tests for acquisition configuration and stream metadata."""

from __future__ import annotations

import ast
import argparse
import base64
import builtins
import importlib.util
import json
import runpy
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from naturallab.acquisition import lsl_streams
from naturallab.acquisition import recording_gui
from naturallab.acquisition import xdf_extract
from naturallab.cli import main as cli_main
from scripts import stream_synchronized_sensors
from scripts.stream_synchronized_sensors import parse_comma_separated


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


def test_recording_gui_defaults_to_bundled_streamer(tmp_path):
    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.config_file = str(tmp_path / "missing-config.json")
    gui._config_migration_notice = ""
    gui._config_security_warning = ""

    loaded = gui.load_config()

    assert Path(loaded["lsl_script_path"]).resolve() == Path(
        lsl_streams.__file__
    ).resolve()


def test_record_command_launches_recording_window(monkeypatch):
    launches = []
    monkeypatch.setattr(recording_gui, "main", lambda: launches.append(True))

    assert cli_main(["record"]) == 0
    assert launches == [True]


def test_simple_streamer_rejects_empty_comma_separated_entries():
    parser = argparse.ArgumentParser()

    assert parse_comma_separated(
        parser,
        "camera-one, camera-two",
        "--cameras",
    ) == ["camera-one", "camera-two"]
    with pytest.raises(SystemExit):
        parse_comma_separated(parser, "camera-one,,camera-two", "--cameras")


def test_simple_streamer_requires_one_name_per_camera(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream_synchronized_sensors.py",
            "--cameras",
            "rtsp://camera-one/live,rtsp://camera-two/live",
            "--camera-names",
            "camera-one",
        ],
    )

    with pytest.raises(SystemExit):
        stream_synchronized_sensors.main()


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
    gui.child_ip_var = Variable("192.0.2.21")
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
    gui.no_neon_var = Variable(False)
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
    assert launched["command"][0] == sys.executable
    assert "--eye-events" in launched["command"]
    assert url_argument == (
        "rtsp://lab%20user%40example.org:p%40ss%20word%2C%23@"
        "192.0.2.10:554/main/stream"
    )
    combined_log = "\n".join(messages)
    assert "p@ss word,#" not in combined_log
    assert "p%40ss%20word%2C%23" not in combined_log
    assert "lab%20user%40example.org:***@192.0.2.10:554" in combined_log


def test_recorder_ignores_legacy_placeholder_addresses(
    monkeypatch,
):
    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.streaming_active = False
    gui.lsl_process = None
    gui.lsl_script_var = Variable(str(Path(lsl_streams.__file__)))
    gui.conda_env_var = Variable("")
    gui.caregiver_ip_var = Variable("YOUR_IP_ADDRESS")
    gui.child_ip_var = Variable("YOUR_IP_ADDRESS")
    gui.max_devices_var = Variable(2)
    gui.rtsp_user_var = Variable("admin")
    gui.rtsp_pass_var = Variable("")
    gui.rtsp_stream_var = Variable("stream1")
    gui.camera_vars = {
        "cam1": {
            "ip": Variable("YOUR_IP_ADDRESS"),
            "name": Variable("Camera1"),
            "enabled": Variable(True),
        }
    }
    gui.no_neon_var = Variable(True)
    gui.no_realsense_var = Variable(True)
    gui.no_audio_var = Variable(False)
    gui.no_imu_var = Variable(False)
    gui.no_eye_events_var = Variable(True)
    gui.start_lsl_btn = Button()
    gui.stop_lsl_btn = Button()
    gui.read_lsl_output = lambda: None
    gui.log = lambda _message: None

    launched = {}

    class Process:
        pid = 1234

    def fake_popen(command, **_kwargs):
        launched["command"] = command
        return Process()

    class Thread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(recording_gui.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(recording_gui.threading, "Thread", Thread)

    gui.start_lsl_streaming()

    command = launched["command"]
    assert "--no-neon" in command
    assert "--caregiver-ip" not in command
    assert "--child-ip" not in command
    assert "--rtsp-urls" not in command
    assert "--no-eye-events" in command


def test_camera_url_preview_treats_legacy_placeholder_as_unconfigured(
    monkeypatch,
):
    rendered = []

    class Popup:
        def __init__(self, _root):
            pass

        def title(self, _title):
            pass

        def geometry(self, _geometry):
            pass

        def destroy(self):
            pass

    class TextWidget:
        def __init__(self, *_args, **_kwargs):
            pass

        def pack(self, **_kwargs):
            pass

        def insert(self, _index, text):
            rendered.append(text)

        def config(self, **_kwargs):
            pass

    class Widget:
        def pack(self, **_kwargs):
            pass

    monkeypatch.setattr(
        recording_gui,
        "tk",
        SimpleNamespace(
            Toplevel=Popup,
            WORD="word",
            BOTH="both",
            DISABLED="disabled",
        ),
    )
    monkeypatch.setattr(
        recording_gui,
        "scrolledtext",
        SimpleNamespace(ScrolledText=TextWidget),
    )
    monkeypatch.setattr(
        recording_gui,
        "ttk",
        SimpleNamespace(Button=lambda *_args, **_kwargs: Widget()),
    )

    def reject_placeholder(*_args, **_kwargs):
        pytest.fail("the legacy placeholder must not be formatted as an RTSP URL")

    monkeypatch.setattr(recording_gui, "build_rtsp_url", reject_placeholder)

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.root = object()
    gui.rtsp_user_var = Variable("admin")
    gui.rtsp_pass_var = Variable("")
    gui.rtsp_stream_var = Variable("stream1")
    gui.camera_vars = {
        "cam1": {
            "ip": Variable("YOUR_IP_ADDRESS"),
            "name": Variable("Camera1"),
            "enabled": Variable(True),
        }
    }

    gui.show_camera_urls()

    assert len(rendered) == 1
    assert "Camera 1: ⚠️ INCOMPLETE" in rendered[0]
    assert "YOUR_IP_ADDRESS" not in rendered[0]


def test_quick_setup_stops_when_lsl_streaming_does_not_start(monkeypatch):
    logs = []
    labrecorder_starts = []
    success_messages = []

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.streaming_active = False
    gui.labrecorder_process = None
    gui.log = logs.append
    gui.start_lsl_streaming = lambda: None
    gui.start_labrecorder = lambda: labrecorder_starts.append(True)

    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(
            showinfo=lambda *args, **kwargs: success_messages.append(
                (args, kwargs)
            )
        ),
    )
    monkeypatch.setattr(
        recording_gui.time,
        "sleep",
        lambda _seconds: pytest.fail("failed LSL startup must not wait"),
    )

    gui.quick_setup()

    assert not labrecorder_starts
    assert not success_messages
    assert any("LSL streaming did not start" in message for message in logs)


def test_quick_setup_stops_when_lsl_process_exits_during_wait(monkeypatch):
    logs = []
    labrecorder_starts = []
    success_messages = []

    class ExitedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.streaming_active = False
    gui.lsl_process = None
    gui.labrecorder_process = None
    gui.log = logs.append

    def start_then_exit():
        gui.streaming_active = True
        gui.lsl_process = ExitedProcess()

    gui.start_lsl_streaming = start_then_exit
    gui.start_labrecorder = lambda: labrecorder_starts.append(True)

    monkeypatch.setattr(recording_gui.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(
            showinfo=lambda *args, **kwargs: success_messages.append(
                (args, kwargs)
            )
        ),
    )

    gui.quick_setup()

    assert gui.streaming_active is False
    assert not labrecorder_starts
    assert not success_messages
    assert any("exited during startup" in message for message in logs)


def test_eye_event_streaming_requires_an_explicit_child_neon_ip(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "lsl_streams.py"
    script_path.write_text("# test executable path\n", encoding="utf-8")
    errors = []

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.streaming_active = False
    gui.lsl_process = None
    gui.lsl_script_var = Variable(str(script_path))
    gui.conda_env_var = Variable("")
    gui.caregiver_ip_var = Variable("192.0.2.20")
    gui.child_ip_var = Variable("YOUR_IP_ADDRESS")
    gui.no_neon_var = Variable(False)
    gui.no_eye_events_var = Variable(False)

    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(
            showerror=lambda *args, **kwargs: errors.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(
        recording_gui.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid eye-event configuration must not launch a subprocess"
        ),
    )

    gui.start_lsl_streaming()

    assert gui.streaming_active is False
    assert len(errors) == 1
    assert errors[0][0][0] == "Child Neon IP required"
    assert "explicit Child Neon IP" in errors[0][0][1]


def test_disabling_neon_also_blocks_explicit_eye_event_start(monkeypatch):
    eye_event_starts = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--no-neon",
            "--eye-events",
            "--no-realsense",
            "--no-audio",
            "--no-imu",
        ],
    )
    monkeypatch.setattr(lsl_streams, "pylsl", object())
    monkeypatch.setattr(lsl_streams, "REALTIME_API_AVAILABLE", True)
    monkeypatch.setattr(lsl_streams, "running", False)
    monkeypatch.setattr(
        lsl_streams,
        "stream_eye_events_to_lsl",
        lambda *_arguments: eye_event_starts.append(True),
    )

    lsl_streams.main()

    assert eye_event_starts == []


def test_eye_events_receive_the_configured_child_ip(monkeypatch):
    received_addresses = []

    class FinishedThread:
        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--child-ip",
            "192.0.2.42",
            "--eye-events",
            "--no-realsense",
            "--no-audio",
            "--no-imu",
        ],
    )
    monkeypatch.setattr(lsl_streams, "pylsl", object())
    monkeypatch.setattr(lsl_streams, "REALTIME_API_AVAILABLE", True)
    monkeypatch.setattr(lsl_streams, "running", False)
    monkeypatch.setattr(
        lsl_streams,
        "stream_eye_events_to_lsl",
        lambda address: (
            received_addresses.append(address) or FinishedThread()
        ),
    )

    assert lsl_streams.main() == 0
    assert received_addresses == ["192.0.2.42"]


def test_eye_events_require_explicit_child_ip(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--eye-events",
            "--no-realsense",
            "--no-audio",
            "--no-imu",
        ],
    )
    monkeypatch.setattr(lsl_streams, "pylsl", object())

    with pytest.raises(SystemExit):
        lsl_streams.main()


def test_bundled_streamer_fails_when_every_source_is_disabled(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lsl_streams.py",
            "--no-neon",
            "--no-realsense",
            "--no-audio",
            "--no-imu",
        ],
    )
    monkeypatch.setattr(lsl_streams, "pylsl", object())

    assert lsl_streams.main() == 1
    assert "no acquisition source started" in capsys.readouterr().err


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


def test_xdf_video_extraction_rejects_timestamp_length_mismatch(tmp_path):
    stream = {
        "info": {"name": ["camera-01"]},
        "time_series": ["frame-one", "frame-two"],
        "time_stamps": [1.0],
    }

    with pytest.raises(RuntimeError, match="video/timestamp length mismatch"):
        xdf_extract.extract_video_stream(stream, tmp_path)


def test_xdf_video_extraction_does_not_skip_a_corrupt_frame(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    valid_frame = base64.b64encode(encoded.tobytes()).decode("ascii")
    stream = {
        "info": {"name": ["camera-01"]},
        "time_series": [[valid_frame], ["not-base64"]],
        "time_stamps": [1.0, 1.1],
    }

    with pytest.raises(RuntimeError, match="could not decode frame 1"):
        xdf_extract.extract_video_stream(stream, tmp_path)

    assert not (tmp_path / "camera-01_timestamps.csv").exists()
    assert not (tmp_path / "camera-01.mp4").exists()


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


def test_all_device_disable_flags_survive_save_and_reload(tmp_path, monkeypatch):
    config_path = tmp_path / "recording.json"
    flag_values = {
        "no_neon": True,
        "no_realsense": True,
        "no_audio": True,
        "no_imu": True,
        "no_eye_events": False,
    }

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.config_file = str(config_path)
    gui.config = {}
    gui.log = lambda _message: None
    gui.caregiver_ip_var = Variable("192.0.2.20")
    gui.child_ip_var = Variable("192.0.2.21")
    gui.max_devices_var = Variable(2)
    gui.lsl_script_var = Variable(str(Path(lsl_streams.__file__)))
    gui.conda_env_var = Variable("")
    gui.labrecorder_path_var = Variable("/usr/local/bin/LabRecorder")
    gui.labrecorder_config_var = Variable("")
    gui.extraction_drive_var = Variable("/media")
    gui.recording_dir_var = Variable(str(tmp_path / "recordings"))
    gui.auto_extract_var = Variable(True)
    gui.delete_after_extract_var = Variable(False)
    gui.depth_interval_var = Variable(30)
    gui.keep_raw_depth_var = Variable(True)
    gui.rtsp_user_var = Variable("admin")
    gui.rtsp_pass_var = Variable("session-only")
    gui.rtsp_stream_var = Variable("stream1")
    gui.camera_vars = {}
    for key, value in flag_values.items():
        setattr(gui, f"{key}_var", Variable(value))

    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(showinfo=lambda *_args, **_kwargs: None),
    )

    gui.save_current_config()

    reloaded = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    reloaded.config_file = str(config_path)
    reloaded._config_migration_notice = ""
    reloaded._config_security_warning = ""
    reloaded.config = reloaded.load_config()
    reloaded.caregiver_ip_var = Variable(None)
    reloaded.child_ip_var = Variable(None)
    reloaded.max_devices_var = Variable(None)
    reloaded.lsl_script_var = Variable(None)
    reloaded.conda_env_var = Variable(None)
    reloaded.labrecorder_path_var = Variable(None)
    reloaded.labrecorder_config_var = Variable(None)
    reloaded.extraction_drive_var = Variable(None)
    reloaded.recording_dir_var = Variable(None)
    reloaded.file_dir_var = Variable(None)
    reloaded.auto_extract_var = Variable(None)
    reloaded.delete_after_extract_var = Variable(None)
    reloaded.depth_interval_var = Variable(None)
    reloaded.keep_raw_depth_var = Variable(None)
    reloaded.rtsp_user_var = Variable(None)
    reloaded.rtsp_pass_var = Variable(None)
    reloaded.rtsp_stream_var = Variable(None)
    reloaded.camera_vars = {}
    for key in flag_values:
        setattr(reloaded, f"{key}_var", Variable(None))

    reloaded.load_saved_settings()

    assert {
        key: getattr(reloaded, f"{key}_var").get()
        for key in flag_values
    } == flag_values
    assert "session-only" not in config_path.read_text(encoding="utf-8")


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
        assert "pylsl.local_clock()" in source, source_path
        assert "time.monotonic()" in source, source_path

        tree = ast.parse(source)
        wall_clock_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
            and node.func.attr == "time"
        ]
        if source_path.name == "lsl_streams.py":
            clock_mapping = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_lsl_unix_clock_offset"
            )
            assert all(
                clock_mapping.lineno <= call.lineno <= clock_mapping.end_lineno
                for call in wall_clock_calls
            )
        else:
            assert not wall_clock_calls, source_path


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
