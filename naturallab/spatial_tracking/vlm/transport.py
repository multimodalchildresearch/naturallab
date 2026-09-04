"""OpenAI-compatible JSON-over-HTTP transport with an injectable boundary."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping


class VLMTransportError(RuntimeError):
    """Raised when the local VLM service cannot return a JSON response."""


class UrllibJSONTransport:
    """Minimal standard-library transport for local OpenAI-compatible servers."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise VLMTransportError(
                f"VLM endpoint returned HTTP status {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise VLMTransportError("could not reach the configured VLM endpoint") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VLMTransportError("VLM endpoint did not return valid JSON") from exc

        if not isinstance(result, Mapping):
            raise VLMTransportError("VLM endpoint response must be a JSON object")
        return result
