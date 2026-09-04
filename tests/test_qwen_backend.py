import json
from importlib import resources
from typing import Any, Mapping

import pytest
import yaml

try:
    from importlib.resources import files as resource_files
except ImportError:  # pragma: no cover - Python 3.8 compatibility
    resource_files = None

from naturallab.spatial_tracking.vlm import (
    DEFAULT_QWEN_MODEL_ID,
    EvidenceImage,
    QwenBackendConfig,
    QwenPersonGrounder,
    QwenTrackRoleAssigner,
    VLMResponseError,
    VLMValidationError,
)


class FakeTransport:
    def __init__(self, model_content: str):
        self.model_content = model_content
        self.calls = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": self.model_content,
                    }
                }
            ]
        }


def test_grounding_parses_valid_boxes_without_inventing_confidence():
    transport = FakeTransport(
        json.dumps(
            {
                "detections": [
                    {
                        "bbox": [0.1, 0.2, 0.7, 0.9],
                        "confidence": 0.83,
                        "label": "person",
                    },
                    {
                        "bbox": [0.0, 0.0, 0.25, 0.5],
                        "label": "person",
                    },
                ]
            }
        )
    )
    result = QwenPersonGrounder(transport=transport).ground(
        EvidenceImage(b"frame")
    )

    assert [detection.bbox.as_tuple() for detection in result.detections] == [
        (0.1, 0.2, 0.7, 0.9),
        (0.0, 0.0, 0.25, 0.5),
    ]
    assert result.detections[0].confidence == 0.83
    assert result.detections[1].confidence is None
    assert result.provenance.model_id == DEFAULT_QWEN_MODEL_ID


def test_grounding_normalizes_qwen_relative_integer_boxes():
    transport = FakeTransport(
        json.dumps(
            {
                "detections": [
                    {
                        "bbox": [35, 30, 715, 997],
                        "confidence": None,
                        "label": "person",
                    }
                ]
            }
        )
    )

    result = QwenPersonGrounder(transport=transport).ground(
        EvidenceImage(b"frame")
    )

    assert result.detections[0].bbox.as_tuple() == pytest.approx(
        (0.035, 0.03, 0.715, 0.997)
    )
    assert result.provenance.prompt_version == "qwen-person-grounding/v2"


@pytest.mark.parametrize(
    "bbox",
    [
        [-0.1, 0.1, 0.5, 0.5],
        [0.1, 0.1, 1.1, 0.5],
        [0.6, 0.1, 0.2, 0.5],
        [0.1, 0.6, 0.5, 0.2],
        [0.1, 0.2, 0.5],
        [True, 0.2, 0.5, 0.8],
        [0, 0, 1001, 1000],
    ],
)
def test_grounding_rejects_invalid_normalized_boxes(bbox):
    transport = FakeTransport(
        json.dumps(
            {
                "detections": [
                    {
                        "bbox": bbox,
                        "confidence": None,
                        "label": "person",
                    }
                ]
            }
        )
    )

    with pytest.raises(VLMResponseError, match=r"invalid detections\[0\]"):
        QwenPersonGrounder(transport=transport).ground(EvidenceImage(b"frame"))


def test_grounding_accepts_single_fenced_json_object():
    transport = FakeTransport(
        """```json
{"detections":[{"bbox":[0.1,0.2,0.6,0.8],"confidence":null,"label":"person"}]}
```"""
    )
    result = QwenPersonGrounder(transport=transport).ground(
        EvidenceImage(b"frame")
    )

    assert len(result.detections) == 1
    assert result.detections[0].confidence is None


@pytest.mark.parametrize(
    "model_content",
    [
        "Here is the result: {\"detections\": []}",
        "```json\n{\"detections\": []}\n```\nextra",
        "{\"detections\":",
        "[]",
    ],
)
def test_grounding_rejects_malformed_or_embellished_responses(model_content):
    transport = FakeTransport(model_content)

    with pytest.raises(VLMResponseError):
        QwenPersonGrounder(transport=transport).ground(EvidenceImage(b"frame"))


def test_role_assignment_supports_explicit_abstention():
    transport = FakeTransport(
        json.dumps(
            {
                "track_id": "track-7",
                "role": None,
                "abstain": True,
                "reason": "Evidence views disagree",
            }
        )
    )
    assignment = QwenTrackRoleAssigner(
        roles=("participant", "facilitator"),
        transport=transport,
    ).assign_role(
        "track-7",
        [EvidenceImage(b"first"), EvidenceImage(b"second")],
    )

    assert assignment.track_id == "track-7"
    assert assignment.abstained is True
    assert assignment.role is None
    assert assignment.confidence is None
    assert assignment.reason == "Evidence views disagree"


def test_role_descriptions_are_supplied_without_expanding_the_whitelist():
    transport = FakeTransport(
        json.dumps(
            {
                "track_id": "track-8",
                "role": "facilitator",
                "abstain": False,
            }
        )
    )
    assigner = QwenTrackRoleAssigner(
        roles=("participant", "facilitator"),
        role_descriptions={
            "participant": "the person completing the task",
            "facilitator": "the person presenting the materials",
        },
        transport=transport,
    )

    assignment = assigner.assign_role(
        "track-8",
        [EvidenceImage(b"frame")],
    )

    prompt_text = transport.calls[0]["payload"]["messages"][1]["content"][0][
        "text"
    ]
    assert assignment.role == "facilitator"
    assert "the person completing the task" in prompt_text
    assert "the person presenting the materials" in prompt_text


def test_role_assignment_rejects_role_outside_whitelist():
    transport = FakeTransport(
        json.dumps(
            {
                "track_id": "track-7",
                "role": "researcher",
                "abstain": False,
                "confidence": 0.8,
            }
        )
    )

    with pytest.raises(VLMResponseError, match="whitelisted role"):
        QwenTrackRoleAssigner(
            roles=("participant", "facilitator"),
            transport=transport,
        ).assign_role("track-7", [EvidenceImage(b"frame")])


def test_fake_transport_captures_openai_request_and_secret_free_provenance():
    transport = FakeTransport(
        json.dumps(
            {
                "track_id": "person-a",
                "role": "participant",
                "abstain": False,
                "confidence": 0.91,
            }
        )
    )
    config = QwenBackendConfig(
        base_url="https://user:password@vlm.internal.example/v1?token=secret",
        api_key="service-secret",
        precision="bf16",
        detection_cadence_frames=17,
        timeout_seconds=9,
    )
    assignment = QwenTrackRoleAssigner(
        roles=("participant", "facilitator"),
        config=config,
        transport=transport,
    ).assign_role(
        "person-a",
        [
            EvidenceImage(b"frame-one", label="front"),
            EvidenceImage(b"frame-two", mime_type="image/png"),
        ],
    )

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert (
        call["url"]
        == "https://user:password@vlm.internal.example/v1/chat/completions?token=secret"
    )
    assert call["headers"]["Authorization"] == "Bearer service-secret"
    assert call["payload"]["model"] == "Qwen/Qwen3.6-27B"
    assert call["payload"]["response_format"] == {"type": "json_object"}
    assert call["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert call["timeout_seconds"] == 9.0

    user_content = call["payload"]["messages"][1]["content"]
    image_parts = [part for part in user_content if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert image_parts[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    provenance = assignment.provenance.as_dict()
    assert provenance == {
        "model_id": "Qwen/Qwen3.6-27B",
        "prompt_version": "qwen-track-role-assignment/v2",
        "endpoint_identity": "https://vlm.internal.example/v1",
        "precision": "bf16",
        "detection_cadence_frames": 17,
    }
    assert "service-secret" not in repr(provenance)
    assert "password" not in repr(provenance)
    assert "token=secret" not in repr(provenance)


def test_config_uses_shared_vlm_environment_names(monkeypatch):
    monkeypatch.setenv(
        "NATURALLAB_VLM_BASE_URL",
        "https://cluster-vlm.internal:8080/v1",
    )
    monkeypatch.setenv("NATURALLAB_VLM_API_KEY", "environment-secret")

    config = QwenBackendConfig()

    assert config.base_url == "https://cluster-vlm.internal:8080/v1"
    assert config.api_key == "environment-secret"
    assert "environment-secret" not in repr(config)


def test_config_rejects_a_nonnumeric_url_port():
    with pytest.raises(VLMValidationError, match="numeric port"):
        QwenBackendConfig(base_url="http://localhost:not-a-port/v1")


def test_config_rejects_plain_http_for_remote_service(monkeypatch):
    monkeypatch.delenv("NATURALLAB_ALLOW_INSECURE_VLM_HTTP", raising=False)

    with pytest.raises(VLMValidationError, match="must use HTTPS"):
        QwenBackendConfig(base_url="http://vlm.internal.example/v1")


def test_config_allows_explicit_plain_http_opt_in(monkeypatch):
    monkeypatch.setenv("NATURALLAB_ALLOW_INSECURE_VLM_HTTP", "1")

    config = QwenBackendConfig(base_url="http://vlm.internal.example/v1")

    assert config.base_url == "http://vlm.internal.example/v1"


def test_default_precision_does_not_claim_a_server_deployment_format():
    assert QwenBackendConfig().precision == "unspecified"


def test_quality_preset_selects_the_exact_qwen_model() -> None:
    if resource_files is None:
        preset_text = resources.read_text(
            "naturallab.config.presets",
            "qwen36_27b_quality.yaml",
            encoding="utf-8",
        )
    else:
        preset_text = (
            resource_files("naturallab.config.presets")
            .joinpath("qwen36_27b_quality.yaml")
            .read_text(encoding="utf-8")
        )
    preset = yaml.safe_load(preset_text)

    assert preset["vlm_service"]["model_id"] == "Qwen/Qwen3.6-27B"
    assert preset["vlm_service"]["precision"] == "unspecified"
    assert preset["spatial"]["detector"]["backend"] == (
        "qwen_grounding"
    )
    assert preset["spatial"]["role_assignment"]["backend"] == (
        "qwen_track_role"
    )
    assert "roles" not in preset["spatial"]["role_assignment"]
    assert "role_descriptions" not in preset["spatial"]["role_assignment"]
