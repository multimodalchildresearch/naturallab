"""Safety contracts for recorder processes owned by the desktop window."""

from types import SimpleNamespace

from naturallab.acquisition import recording_gui


class _Button:
    def __init__(self):
        self.states = []

    def config(self, **kwargs):
        self.states.append(kwargs)


def test_force_stop_only_targets_the_process_owned_by_the_window(monkeypatch):
    calls = []

    class Process:
        def poll(self):
            return None

        def kill(self):
            calls.append("kill")

        def wait(self, timeout):
            calls.append(("wait", timeout))

    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.lsl_process = Process()
    gui.streaming_active = True
    gui.start_lsl_btn = _Button()
    gui.stop_lsl_btn = _Button()
    gui.log = lambda message: calls.append(message)

    monkeypatch.setattr(
        recording_gui.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("force-stop must not scan system processes")
        ),
    )
    monkeypatch.setattr(
        recording_gui.os,
        "kill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("force-stop must use only the tracked process handle")
        ),
    )
    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(showinfo=lambda *_args, **_kwargs: None),
    )

    gui.force_stop_lsl()

    assert calls[:2] == ["Force-stopping this window's recorder...", "kill"]
    assert ("wait", 5) in calls
    assert gui.lsl_process is None
    assert gui.streaming_active is False


def test_emergency_stop_uses_only_owned_process_helpers(monkeypatch):
    calls = []
    gui = recording_gui.RecordingGUI.__new__(recording_gui.RecordingGUI)
    gui.lsl_process = object()
    gui.labrecorder_process = object()
    gui.streaming_active = True
    gui.log = lambda message: calls.append(message)
    gui.stop_lsl_streaming = lambda: calls.append("stop recorder")
    gui.close_labrecorder = lambda: calls.append("close LabRecorder")
    monkeypatch.setattr(
        recording_gui.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("emergency stop must not scan system processes")
        ),
    )
    monkeypatch.setattr(
        recording_gui,
        "messagebox",
        SimpleNamespace(showinfo=lambda *_args, **_kwargs: None),
    )

    gui.emergency_stop()

    assert "stop recorder" in calls
    assert "close LabRecorder" in calls
