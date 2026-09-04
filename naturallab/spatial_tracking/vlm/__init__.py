"""Swappable VLM backends for person grounding and track-role assignment."""

from .prompts import (
    GROUNDING_PROMPT_VERSION,
    ROLE_ASSIGNMENT_PROMPT_VERSION,
)
from .qwen_backend import (
    DEFAULT_QWEN_MODEL_ID,
    DEFAULT_VLM_BASE_URL,
    VLM_API_KEY_ENV,
    VLM_BASE_URL_ENV,
    OpenAICompatibleChatClient,
    QwenBackendConfig,
    QwenPersonGrounder,
    QwenTrackRoleAssigner,
    VLMResponseError,
    parse_json_object,
)
from .transport import UrllibJSONTransport, VLMTransportError
from .types import (
    EvidenceImage,
    InferenceProvenance,
    JSONTransport,
    NormalizedXYXY,
    PersonGrounder,
    PersonGroundingDetection,
    PersonGroundingResult,
    TrackRoleAssigner,
    TrackRoleAssignment,
    VLMValidationError,
)

__all__ = [
    "DEFAULT_QWEN_MODEL_ID",
    "DEFAULT_VLM_BASE_URL",
    "EvidenceImage",
    "GROUNDING_PROMPT_VERSION",
    "InferenceProvenance",
    "JSONTransport",
    "NormalizedXYXY",
    "OpenAICompatibleChatClient",
    "PersonGrounder",
    "PersonGroundingDetection",
    "PersonGroundingResult",
    "QwenBackendConfig",
    "QwenPersonGrounder",
    "QwenTrackRoleAssigner",
    "ROLE_ASSIGNMENT_PROMPT_VERSION",
    "TrackRoleAssigner",
    "TrackRoleAssignment",
    "UrllibJSONTransport",
    "VLM_API_KEY_ENV",
    "VLM_BASE_URL_ENV",
    "VLMResponseError",
    "VLMTransportError",
    "VLMValidationError",
    "parse_json_object",
]
