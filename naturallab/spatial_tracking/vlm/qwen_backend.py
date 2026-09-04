"""Qwen3.6-27B person grounding and semantic track-role assignment."""

from __future__ import annotations

import json
import ipaddress
import math
import os
import re
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from .prompts import (
    GROUNDING_PROMPT_VERSION,
    GROUNDING_SYSTEM_PROMPT,
    ROLE_ASSIGNMENT_PROMPT_VERSION,
    ROLE_SYSTEM_PROMPT,
)
from .transport import UrllibJSONTransport
from .types import (
    EvidenceImage,
    InferenceProvenance,
    JSONTransport,
    NormalizedXYXY,
    PersonGroundingDetection,
    PersonGroundingResult,
    TrackRoleAssignment,
    VLMValidationError,
)

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_VLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLM_BASE_URL_ENV = "NATURALLAB_VLM_BASE_URL"
VLM_API_KEY_ENV = "NATURALLAB_VLM_API_KEY"
VLM_ALLOW_INSECURE_HTTP_ENV = "NATURALLAB_ALLOW_INSECURE_VLM_HTTP"

_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*(?P<body>\{.*\})\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class VLMResponseError(RuntimeError):
    """Raised when the model response is missing or malformed."""


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VLMValidationError("base_url must be a non-empty HTTP(S) URL")
    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise VLMValidationError("base_url must be a valid HTTP(S) URL")
    try:
        parts.port
    except ValueError as error:
        raise VLMValidationError(
            "base_url must contain a valid numeric port"
        ) from error
    hostname = parts.hostname or ""
    is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if (
        parts.scheme == "http"
        and not is_loopback
        and os.environ.get(VLM_ALLOW_INSECURE_HTTP_ENV) != "1"
    ):
        raise VLMValidationError(
            "non-loopback VLM endpoints must use HTTPS; set "
            f"{VLM_ALLOW_INSECURE_HTTP_ENV}=1 only after explicitly accepting "
            "plaintext transport on a trusted network"
        )
    return value


def _endpoint_identity(base_url: str) -> str:
    """Remove credentials, query parameters, and fragments from provenance."""

    parts = urlsplit(base_url)
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class QwenBackendConfig:
    """Runtime settings shared by Qwen grounding and role assignment."""

    model_id: str = DEFAULT_QWEN_MODEL_ID
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            VLM_BASE_URL_ENV,
            DEFAULT_VLM_BASE_URL,
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get(VLM_API_KEY_ENV),
        repr=False,
    )
    precision: str = "unspecified"
    detection_cadence_frames: int = 10
    timeout_seconds: float = 120.0
    max_tokens: int = 1024
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise VLMValidationError("model_id must be a non-empty string")
        object.__setattr__(self, "base_url", _validate_base_url(self.base_url))
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise VLMValidationError("api_key must be a string or null")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise VLMValidationError("precision must be a non-empty string")
        if (
            isinstance(self.detection_cadence_frames, bool)
            or not isinstance(self.detection_cadence_frames, int)
            or self.detection_cadence_frames < 1
        ):
            raise VLMValidationError(
                "detection_cadence_frames must be a positive integer"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise VLMValidationError("timeout_seconds must be positive")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise VLMValidationError("max_tokens must be a positive integer")
        if not isinstance(self.enable_thinking, bool):
            raise VLMValidationError("enable_thinking must be a boolean")

    @property
    def endpoint_identity(self) -> str:
        return _endpoint_identity(self.base_url)


class OpenAICompatibleChatClient:
    """Construct chat-completion requests without coupling tests to a network."""

    def __init__(
        self,
        config: QwenBackendConfig,
        transport: Optional[JSONTransport] = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJSONTransport()

    @property
    def chat_completions_url(self) -> str:
        parts = urlsplit(self.config.base_url)
        path = parts.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path = f"{path}/chat/completions"
        return urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, "")
        )

    def complete(self, messages: Sequence[Mapping[str, Any]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model_id,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking
            },
        }
        response = self.transport.post_json(
            self.chat_completions_url,
            headers=headers,
            payload=payload,
            timeout_seconds=float(self.config.timeout_seconds),
        )
        return _extract_message_content(response)


def _extract_message_content(response: Mapping[str, Any]) -> str:
    try:
        choices = response["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMResponseError(
            "OpenAI-compatible response is missing choices[0].message.content"
        ) from exc

    if isinstance(content, str):
        if not content.strip():
            raise VLMResponseError("model returned empty message content")
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
            ):
                raise VLMResponseError(
                    "message content contains a non-text response part"
                )
            text_parts.append(part["text"])
        joined = "".join(text_parts)
        if not joined.strip():
            raise VLMResponseError("model returned empty message content")
        return joined
    raise VLMResponseError("message content must be text")


def parse_json_object(content: str) -> Mapping[str, Any]:
    """Parse a bare JSON object or a single fenced JSON object, with no prose."""

    if not isinstance(content, str):
        raise VLMResponseError("model content must be text")
    candidate = content.strip()
    fenced = _JSON_FENCE.fullmatch(candidate)
    if fenced:
        candidate = fenced.group("body")
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VLMResponseError(
            "model content is not a valid bare or fenced JSON object"
        ) from exc
    if not isinstance(result, Mapping):
        raise VLMResponseError("model content must decode to a JSON object")
    return result


def _optional_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    # Reuse the detection type's strict confidence validation.
    placeholder = PersonGroundingDetection(
        bbox=NormalizedXYXY(0.0, 0.0, 1.0, 1.0),
        confidence=value,
    )
    return placeholder.confidence


def _grounding_bbox(value: Any) -> NormalizedXYXY:
    """Normalize either strict [0,1] values or Qwen's integer [0,1000] grid."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise VLMValidationError(
            "bbox must be a four-element [x1, y1, x2, y2] array"
        )
    if any(
        isinstance(item, bool)
        or not isinstance(item, Real)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise VLMValidationError("bbox coordinates must be finite numbers")

    coordinates = tuple(float(item) for item in value)
    if all(0.0 <= item <= 1.0 for item in coordinates):
        return NormalizedXYXY.from_sequence(coordinates)
    if all(
        isinstance(item, Integral)
        and not isinstance(item, bool)
        and 0 <= int(item) <= 1000
        for item in value
    ):
        return NormalizedXYXY.from_sequence(
            tuple(float(item) / 1000.0 for item in value)
        )
    raise VLMValidationError(
        "bbox must use normalized [0,1] coordinates or Qwen integer "
        "[0,1000] relative coordinates"
    )


def _parse_grounding(content: str) -> Tuple[PersonGroundingDetection, ...]:
    value = parse_json_object(content)
    if set(value) != {"detections"}:
        raise VLMResponseError(
            "grounding response must contain only the detections field"
        )
    raw_detections = value["detections"]
    if not isinstance(raw_detections, list):
        raise VLMResponseError("detections must be a JSON array")

    detections: List[PersonGroundingDetection] = []
    for index, raw_detection in enumerate(raw_detections):
        if not isinstance(raw_detection, Mapping):
            raise VLMResponseError(f"detections[{index}] must be a JSON object")
        allowed_fields = {"bbox", "confidence", "label"}
        if "bbox" not in raw_detection or not set(raw_detection).issubset(
            allowed_fields
        ):
            raise VLMResponseError(
                f"detections[{index}] has missing or unsupported fields"
            )
        label = raw_detection.get("label", "person")
        if label != "person":
            raise VLMResponseError(
                f"detections[{index}].label must be exactly 'person'"
            )
        try:
            bbox = _grounding_bbox(raw_detection["bbox"])
            confidence = _optional_confidence(raw_detection.get("confidence"))
        except VLMValidationError as exc:
            raise VLMResponseError(f"invalid detections[{index}]: {exc}") from exc
        detections.append(
            PersonGroundingDetection(
                bbox=bbox,
                confidence=confidence,
            )
        )
    return tuple(detections)


def _role_whitelist(roles: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(roles, (str, bytes)) or not roles:
        raise VLMValidationError("roles must be a non-empty sequence")
    normalized: List[str] = []
    for role in roles:
        if not isinstance(role, str) or not role.strip():
            raise VLMValidationError("every role must be a non-empty string")
        role = role.strip()
        if role in normalized:
            raise VLMValidationError(f"duplicate role in whitelist: {role}")
        normalized.append(role)
    return tuple(normalized)


def _provenance(
    config: QwenBackendConfig,
    prompt_version: str,
) -> InferenceProvenance:
    return InferenceProvenance(
        model_id=config.model_id,
        prompt_version=prompt_version,
        endpoint_identity=config.endpoint_identity,
        precision=config.precision,
        detection_cadence_frames=config.detection_cadence_frames,
    )


class QwenPersonGrounder:
    """Person grounder using Qwen3.6-27B through a configured service."""

    def __init__(
        self,
        config: Optional[QwenBackendConfig] = None,
        transport: Optional[JSONTransport] = None,
    ) -> None:
        self.config = config or QwenBackendConfig()
        self.client = OpenAICompatibleChatClient(self.config, transport)

    def ground(self, image: EvidenceImage) -> PersonGroundingResult:
        if not isinstance(image, EvidenceImage):
            raise VLMValidationError("image must be an EvidenceImage")
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Ground every visible person in this frame. Coordinates "
                    "must describe the supplied image."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image.as_data_url(), "detail": "high"},
            },
        ]
        content = self.client.complete(
            [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        return PersonGroundingResult(
            detections=_parse_grounding(content),
            provenance=_provenance(self.config, GROUNDING_PROMPT_VERSION),
        )


class QwenTrackRoleAssigner:
    """Assign a whitelisted role from several images of a temporal track."""

    def __init__(
        self,
        roles: Sequence[str],
        role_descriptions: Optional[Mapping[str, str]] = None,
        evidence_images_per_track: Optional[int] = None,
        config: Optional[QwenBackendConfig] = None,
        transport: Optional[JSONTransport] = None,
    ) -> None:
        self.roles = _role_whitelist(roles)
        if (
            evidence_images_per_track is not None
            and (
                isinstance(evidence_images_per_track, bool)
                or not isinstance(evidence_images_per_track, int)
                or evidence_images_per_track < 1
            )
        ):
            raise VLMValidationError(
                "evidence_images_per_track must be a positive integer or null"
            )
        self.evidence_images_per_track = evidence_images_per_track
        if role_descriptions is None:
            self.role_descriptions: Dict[str, str] = {}
        else:
            if not isinstance(role_descriptions, Mapping):
                raise VLMValidationError("role_descriptions must be a mapping")
            unknown_roles = set(role_descriptions) - set(self.roles)
            if unknown_roles:
                raise VLMValidationError(
                    "role_descriptions contains roles outside the whitelist: "
                    + ", ".join(sorted(unknown_roles))
                )
            normalized_descriptions: Dict[str, str] = {}
            for role, description in role_descriptions.items():
                if not isinstance(description, str) or not description.strip():
                    raise VLMValidationError(
                        "every role description must be non-empty text"
                    )
                normalized_descriptions[role] = description.strip()
            self.role_descriptions = normalized_descriptions
        self.config = config or QwenBackendConfig()
        self.client = OpenAICompatibleChatClient(self.config, transport)

    def assign_role(
        self,
        track_id: str,
        evidence_images: Sequence[EvidenceImage],
    ) -> TrackRoleAssignment:
        if not isinstance(track_id, str) or not track_id.strip():
            raise VLMValidationError("track_id must be a non-empty string")
        if isinstance(evidence_images, (bytes, str)) or not evidence_images:
            raise VLMValidationError(
                "evidence_images must contain at least one image"
            )
        if (
            self.evidence_images_per_track is not None
            and len(evidence_images) > self.evidence_images_per_track
        ):
            raise VLMValidationError(
                "evidence_images contains more than the configured "
                f"evidence_images_per_track limit "
                f"({self.evidence_images_per_track})"
            )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Assign track {json.dumps(track_id)} one of these roles: "
                    f"{json.dumps(self.roles)}. Role descriptions: "
                    f"{json.dumps(self.role_descriptions, sort_keys=True)}. "
                    "Inspect all evidence images "
                    "before deciding, and abstain explicitly when uncertain."
                ),
            }
        ]
        for index, image in enumerate(evidence_images, start=1):
            if not isinstance(image, EvidenceImage):
                raise VLMValidationError(
                    "every evidence_images value must be an EvidenceImage"
                )
            label = image.label or f"evidence-{index}"
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Evidence image {index}: {label}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image.as_data_url(),
                            "detail": "high",
                        },
                    },
                ]
            )

        content = self.client.complete(
            [
                {"role": "system", "content": ROLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        value = parse_json_object(content)
        required_fields = {"track_id", "role", "abstain"}
        allowed_fields = required_fields | {"confidence", "reason"}
        if not required_fields.issubset(value) or not set(value).issubset(
            allowed_fields
        ):
            raise VLMResponseError(
                "role response has missing or unsupported fields"
            )
        if value["track_id"] != track_id:
            raise VLMResponseError(
                "role response track_id does not match the requested track"
            )
        if not isinstance(value["abstain"], bool):
            raise VLMResponseError("role response abstain must be a boolean")

        abstained = value["abstain"]
        role = value["role"]
        if abstained:
            if role is not None:
                raise VLMResponseError(
                    "an abstained role response must set role to null"
                )
        elif not isinstance(role, str) or role not in self.roles:
            raise VLMResponseError(
                "role response must select exactly one whitelisted role"
            )

        try:
            confidence = _optional_confidence(value.get("confidence"))
        except VLMValidationError as exc:
            raise VLMResponseError(f"invalid role confidence: {exc}") from exc
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise VLMResponseError("role response reason must be text or null")

        return TrackRoleAssignment(
            track_id=track_id,
            role=role,
            abstained=abstained,
            confidence=confidence,
            reason=reason,
            provenance=_provenance(
                self.config,
                ROLE_ASSIGNMENT_PROMPT_VERSION,
            ),
        )
