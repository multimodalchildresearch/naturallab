from pathlib import Path

import pytest

from naturallab.spatial_tracking.pipeline import (
    load_spatial_pipeline_preset,
)
from scripts.track_people_in_video import (
    build_argument_parser,
    build_deepsort_components,
    tracker_run_provenance,
    validate_cli_args,
)


def parse_cli(*arguments: str):
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--input", "input.mp4", "--output", "results", *arguments]
    )
    return validate_cli_args(parser, args)


def test_kalman_remains_the_default_without_reid_opt_in() -> None:
    args = parse_cli()

    assert args.tracker == "kalman"
    assert args.reid_model is None
    assert args.allow_reid_fallback is False
    assert args.min_hits == 3
    assert tracker_run_provenance(args) == {
        "backend": "kalman",
        "parameters": {
            "max_age": 30,
            "min_hits": 3,
        },
        "reid_model": None,
    }


def test_deepsort_reid_options_parse_explicitly() -> None:
    args = parse_cli(
        "--detector",
        "qwen",
        "--tracker",
        "deepsort",
        "--reid-model",
        "models/osnet.pth",
        "--allow-reid-fallback",
    )

    assert args.tracker == "deepsort"
    assert args.reid_model == Path("models/osnet.pth")
    assert args.allow_reid_fallback is True
    assert args.min_hits == 1


@pytest.mark.parametrize(
    "arguments,expected_message",
    [
        (
            ("--allow-reid-fallback",),
            "--allow-reid-fallback is valid only with --tracker deepsort",
        ),
        (
            ("--reid-model", "model.pth"),
            "--reid-model is valid only with --tracker deepsort",
        ),
        (
            ("--tracker", "deepsort", "--max-age", "0"),
            "--max-age must be positive with --tracker deepsort",
        ),
    ],
)
def test_invalid_tracker_combinations_fail_before_runtime(
    arguments: tuple[str, ...],
    expected_message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_cli(*arguments)

    assert caught.value.code == 2
    assert expected_message in capsys.readouterr().err


def test_deepsort_helper_applies_overrides_through_quality_factory() -> None:
    args = parse_cli(
        "--tracker",
        "deepsort",
        "--max-age",
        "17",
        "--min-hits",
        "2",
        "--device",
        "cuda",
        "--reid-model",
        "/models/approved-osnet.pth",
        "--allow-reid-fallback",
    )
    calls = []

    class FakeComponents:
        tracker = object()

        @staticmethod
        def provenance():
            return {
                "tracker_backend": "deepsort",
                "tracker_parameters": {
                    "max_age": 17,
                    "min_hits": 2,
                    "reid_device": "cuda",
                    "allow_reid_fallback": True,
                },
                    "reid_model": {
                        "reid_backend": "histogram",
                        "fallback_allowed": True,
                        "fallback_used": True,
                    },
                    "diagnostics": {
                        "enabled": False,
                        "path_policy": (
                            "explicit_new_or_empty_directory_required"
                        ),
                        "output_directory_name": None,
                        "persisted_content": "none",
                        "persists_images": False,
                    },
            }

    def fake_pipeline_builder(**kwargs):
        calls.append(kwargs)
        return FakeComponents()

    components = build_deepsort_components(
        args,
        pipeline_builder=fake_pipeline_builder,
        preset_loader=load_spatial_pipeline_preset,
    )

    assert isinstance(components, FakeComponents)
    assert len(calls) == 1
    call = calls[0]
    assert call["preset"].tracker.max_age == 17
    assert call["preset"].tracker.min_hits == 2
    assert call["preset"].tracker.reid_device == "cuda"
    assert call["reid_model_path"] == Path(
        "/models/approved-osnet.pth"
    )
    assert call["allow_reid_fallback"] is True
    assert tracker_run_provenance(args, components) == {
        "backend": "deepsort",
        "parameters": {
            "max_age": 17,
            "min_hits": 2,
            "reid_device": "cuda",
            "allow_reid_fallback": True,
        },
            "reid_model": {
                "reid_backend": "histogram",
                "fallback_allowed": True,
                "fallback_used": True,
            },
            "diagnostics": {
                "enabled": False,
                "path_policy": (
                    "explicit_new_or_empty_directory_required"
                ),
                "output_directory_name": None,
                "persisted_content": "none",
                "persists_images": False,
            },
        }


def test_deepsort_helper_keeps_fallback_disabled_by_default() -> None:
    args = parse_cli("--tracker", "deepsort", "--device", "cpu")
    calls = []

    class FakeComponents:
        tracker = object()

    def fake_pipeline_builder(**kwargs):
        calls.append(kwargs)
        return FakeComponents()

    build_deepsort_components(
        args,
        pipeline_builder=fake_pipeline_builder,
        preset_loader=load_spatial_pipeline_preset,
    )

    assert calls[0]["allow_reid_fallback"] is False
    assert calls[0]["preset"].tracker.reid_device == "cpu"
