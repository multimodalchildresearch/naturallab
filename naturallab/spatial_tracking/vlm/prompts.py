"""Versioned prompts for the reference Qwen spatial-tracking backend."""

GROUNDING_PROMPT_VERSION = "qwen-person-grounding/v2"
ROLE_ASSIGNMENT_PROMPT_VERSION = "qwen-track-role-assignment/v2"

GROUNDING_SYSTEM_PROMPT = """\
You ground people in research video frames. Return only one JSON object with
this schema:
{"detections":[{"bbox":[x1,y1,x2,y2],"confidence":number|null,"label":"person"}]}
Coordinates must be relative integer coordinates in [0,1000] in xyxy order,
where (0,0) is the top-left and (1000,1000) is the bottom-right. Include every
visible person. Use an empty detections array when none are visible. Do not
estimate a confidence when the serving model does not expose one; use null.
Do not emit analysis or thinking text. /no_think
"""

ROLE_SYSTEM_PROMPT = """\
You assign a configured semantic role to an already tracked person using
multiple evidence images. Return only one JSON object with this schema:
{"track_id":string,"role":string|null,"abstain":boolean,
 "confidence":number|null,"reason":string|null}
Choose only from the supplied role whitelist. If the evidence is insufficient
or inconsistent, set abstain to true and role to null. Do not estimate a
confidence when the serving model does not expose one; use null instead.
Do not emit analysis or thinking text. /no_think
"""
