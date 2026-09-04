"""Validated study/session manifests for reproducible NaturalLab workflows.

The manifest is deliberately independent of any detector, tracker, or model.
It describes the researcher-owned inputs, the selected processing steps, their
dependencies, and the outputs that must exist before a step can be considered
complete.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import yaml  # type: ignore[import-untyped]


MANIFEST_SCHEMA_VERSION = "1.0"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "study_id",
    "session_id",
    "views",
    "steps",
    "metadata",
}
_VIEW_KEYS = {
    "media",
    "calibration",
    "role_input",
    "object_input",
    "gaze_input",
    "metadata",
}
_CALIBRATION_KEYS = {"intrinsics", "floor_plane", "registration"}
_STEP_KEYS = {"selected", "depends_on", "inputs", "outputs", "config"}


class ManifestError(ValueError):
    """Raised when a study manifest violates the workflow contract."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            "manifest metadata and step config must contain JSON-compatible "
            "values"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{field_name} keys must be strings")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(
            f"{field_name} contains unknown field(s): {', '.join(unknown)}"
        )


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ManifestError(
            f"{field_name} must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return value


def _path_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} must be a non-empty path string")
    if "\x00" in value:
        raise ManifestError(f"{field_name} must not contain a null byte")
    return value.strip()


def _paths_overlap(left: Path, right: Path) -> bool:
    """Whether two resolved declarations are equal or contain one another."""

    return left == right or left in right.parents or right in left.parents


def _path_sequence(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{field_name} must be a list of path strings")
    paths = tuple(
        _path_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(paths)) != len(paths):
        raise ManifestError(f"{field_name} must not contain duplicate paths")
    return paths


def _string_sequence(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{field_name} must be a list of step names")
    names = tuple(
        _identifier(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(names)) != len(names):
        raise ManifestError(f"{field_name} must not contain duplicate names")
    return names


def _json_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    data = dict(_mapping(value, field_name))
    # Round-tripping both validates and detaches nested mutable objects supplied
    # by callers.  YAML-specific values such as dates are rejected explicitly.
    return json.loads(_canonical_json(data))


@dataclass(frozen=True)
class CalibrationPaths:
    """Paths to the versioned calibration artifacts for one view."""

    intrinsics: Optional[str] = None
    floor_plane: Optional[str] = None
    registration: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        field_name: str,
    ) -> "CalibrationPaths":
        values = _mapping(data, field_name)
        _reject_unknown_keys(values, _CALIBRATION_KEYS, field_name)
        calibration = cls(
            intrinsics=(
                _path_string(values["intrinsics"], f"{field_name}.intrinsics")
                if "intrinsics" in values
                else None
            ),
            floor_plane=(
                _path_string(
                    values["floor_plane"],
                    f"{field_name}.floor_plane",
                )
                if "floor_plane" in values
                else None
            ),
            registration=(
                _path_string(
                    values["registration"],
                    f"{field_name}.registration",
                )
                if "registration" in values
                else None
            ),
        )
        if not any(
            (
                calibration.intrinsics,
                calibration.floor_plane,
                calibration.registration,
            )
        ):
            raise ManifestError(f"{field_name} must name at least one artifact")
        return calibration

    def to_dict(self) -> Dict[str, str]:
        return {
            key: value
            for key, value in (
                ("intrinsics", self.intrinsics),
                ("floor_plane", self.floor_plane),
                ("registration", self.registration),
            )
            if value is not None
        }

    def iter_paths(self) -> Iterator[Tuple[str, str]]:
        for name, path in (
            ("calibration.intrinsics", self.intrinsics),
            ("calibration.floor_plane", self.floor_plane),
            ("calibration.registration", self.registration),
        ):
            if path is not None:
                yield name, path


@dataclass(frozen=True)
class ViewSpec:
    """Research inputs associated with one named camera or video view."""

    name: str
    media: str
    calibration: Optional[CalibrationPaths] = None
    role_input: Optional[str] = None
    object_input: Optional[str] = None
    gaze_input: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Mapping[str, Any],
    ) -> "ViewSpec":
        name = _identifier(name, "view name")
        values = _mapping(data, f"views.{name}")
        _reject_unknown_keys(values, _VIEW_KEYS, f"views.{name}")
        if "media" not in values:
            raise ManifestError(f"views.{name}.media is required")

        calibration = None
        if "calibration" in values and values["calibration"] is not None:
            calibration = CalibrationPaths.from_dict(
                _mapping(values["calibration"], f"views.{name}.calibration"),
                field_name=f"views.{name}.calibration",
            )

        return cls(
            name=name,
            media=_path_string(values["media"], f"views.{name}.media"),
            calibration=calibration,
            role_input=(
                _path_string(
                    values["role_input"],
                    f"views.{name}.role_input",
                )
                if values.get("role_input") is not None
                else None
            ),
            object_input=(
                _path_string(
                    values["object_input"],
                    f"views.{name}.object_input",
                )
                if values.get("object_input") is not None
                else None
            ),
            gaze_input=(
                _path_string(
                    values["gaze_input"],
                    f"views.{name}.gaze_input",
                )
                if values.get("gaze_input") is not None
                else None
            ),
            metadata=_json_mapping(
                values.get("metadata", {}),
                f"views.{name}.metadata",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"media": self.media}
        if self.calibration is not None:
            data["calibration"] = self.calibration.to_dict()
        for key, value in (
            ("role_input", self.role_input),
            ("object_input", self.object_input),
            ("gaze_input", self.gaze_input),
        ):
            if value is not None:
                data[key] = value
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def iter_content_paths(self) -> Iterator[Tuple[str, str]]:
        yield f"view.{self.name}.media", self.media
        if self.calibration is not None:
            for calibration_field, calibration_path in (
                self.calibration.iter_paths()
            ):
                yield (
                    f"view.{self.name}.{calibration_field}",
                    calibration_path,
                )
        for optional_field, optional_path in (
            ("role_input", self.role_input),
            ("object_input", self.object_input),
            ("gaze_input", self.gaze_input),
        ):
            if optional_path is not None:
                yield f"view.{self.name}.{optional_field}", optional_path


@dataclass(frozen=True)
class StepSpec:
    """One explicitly selected or skipped workflow step."""

    name: str
    selected: bool
    depends_on: Tuple[str, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Mapping[str, Any],
    ) -> "StepSpec":
        name = _identifier(name, "step name")
        values = _mapping(data, f"steps.{name}")
        _reject_unknown_keys(values, _STEP_KEYS, f"steps.{name}")
        if "selected" not in values or not isinstance(values["selected"], bool):
            raise ManifestError(
                f"steps.{name}.selected must explicitly be true or false"
            )
        if "depends_on" not in values:
            raise ManifestError(
                f"steps.{name}.depends_on must explicitly list dependencies"
            )
        if "outputs" not in values:
            raise ManifestError(
                f"steps.{name}.outputs must explicitly list declared outputs"
            )

        step = cls(
            name=name,
            selected=values["selected"],
            depends_on=_string_sequence(
                values["depends_on"],
                f"steps.{name}.depends_on",
            ),
            inputs=_path_sequence(
                values.get("inputs", []),
                f"steps.{name}.inputs",
            ),
            outputs=_path_sequence(
                values["outputs"],
                f"steps.{name}.outputs",
            ),
            config=_json_mapping(
                values.get("config", {}),
                f"steps.{name}.config",
            ),
        )
        if step.selected and not step.outputs:
            raise ManifestError(
                f"steps.{name} is selected but declares no outputs; a step "
                "without verifiable outputs cannot be resumed safely"
            )
        return step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected,
            "depends_on": list(self.depends_on),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class StudyManifest:
    """A validated, schema-versioned study/session workflow definition."""

    study_id: str
    session_id: str
    views: Mapping[str, ViewSpec]
    steps: Mapping[str, StepSpec]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path.cwd, compare=False, repr=False)
    source_path: Optional[Path] = field(
        default=None,
        compare=False,
        repr=False,
    )
    schema_version: str = field(
        default=MANIFEST_SCHEMA_VERSION,
        init=False,
    )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | str = ".",
        source_path: Optional[Path | str] = None,
    ) -> "StudyManifest":
        values = _mapping(data, "manifest")
        _reject_unknown_keys(values, _TOP_LEVEL_KEYS, "manifest")
        version = values.get("schema_version")
        if str(version) != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                "Unsupported manifest schema_version "
                f"{version!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
            )
        for required in ("study_id", "session_id", "views", "steps"):
            if required not in values:
                raise ManifestError(f"manifest.{required} is required")

        view_values = _mapping(values["views"], "views")
        if not view_values:
            raise ManifestError("views must define at least one named view")
        views = {
            name: ViewSpec.from_dict(name, _mapping(value, f"views.{name}"))
            for name, value in view_values.items()
        }

        step_values = _mapping(values["steps"], "steps")
        if not step_values:
            raise ManifestError("steps must define at least one workflow step")
        steps = {
            name: StepSpec.from_dict(name, _mapping(value, f"steps.{name}"))
            for name, value in step_values.items()
        }

        manifest = cls(
            study_id=_identifier(values["study_id"], "study_id"),
            session_id=_identifier(values["session_id"], "session_id"),
            views=views,
            steps=steps,
            metadata=_json_mapping(values.get("metadata", {}), "metadata"),
            base_dir=Path(base_dir).expanduser().resolve(),
            source_path=(
                Path(source_path).expanduser().resolve()
                if source_path is not None
                else None
            ),
        )
        manifest._validate_step_graph()
        return manifest

    @classmethod
    def from_file(cls, path: Path | str) -> "StudyManifest":
        source_path = Path(path).expanduser().resolve()
        suffix = source_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ManifestError(
                "manifest file must use a .json, .yaml, or .yml extension"
            )
        try:
            with source_path.open("r", encoding="utf-8") as handle:
                if suffix == ".json":
                    data = json.load(handle)
                else:
                    data = yaml.safe_load(handle)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ManifestError(
                f"could not read manifest {source_path}: {exc}"
            ) from exc
        return cls.from_dict(
            _mapping(data, "manifest"),
            base_dir=source_path.parent,
            source_path=source_path,
        )

    def _validate_step_graph(self) -> None:
        for name, step in self.steps.items():
            for dependency in step.depends_on:
                if dependency == name:
                    raise ManifestError(
                        f"steps.{name} cannot depend on itself"
                    )
                if dependency not in self.steps:
                    raise ManifestError(
                        f"steps.{name} depends on unknown step {dependency!r}"
                    )
                if step.selected and not self.steps[dependency].selected:
                    raise ManifestError(
                        f"selected step {name!r} depends on unselected step "
                        f"{dependency!r}; select the dependency explicitly"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str, chain: Tuple[str, ...]) -> None:
            if name in visiting:
                cycle = " -> ".join((*chain, name))
                raise ManifestError(
                    f"workflow step dependencies contain a cycle: {cycle}"
                )
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.steps[name].depends_on:
                visit(dependency, (*chain, name))
            visiting.remove(name)
            visited.add(name)

        for step_name in self.steps:
            visit(step_name, ())

        self._validate_selected_output_paths()

    def _validate_selected_output_paths(self) -> None:
        """Reject declarations that cannot retain independent fingerprints."""

        protected_inputs = []
        if self.source_path is not None:
            protected_inputs.append(
                ("manifest", str(self.source_path), self.source_path)
            )
        for logical_name, path in self.iter_content_paths():
            try:
                resolved = self.resolve_path(path)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ManifestError(
                    f"{logical_name} cannot be resolved safely: {exc}"
                ) from exc
            protected_inputs.append((logical_name, path, resolved))

        claimed_outputs: list[tuple[str, str, Path]] = []
        for step in self.selected_steps():
            step_protected_inputs = list(protected_inputs)
            for index, input_path in enumerate(step.inputs):
                try:
                    resolved_input = self.resolve_path(input_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ManifestError(
                        f"steps.{step.name}.inputs path {input_path!r} cannot "
                        f"be resolved safely: {exc}"
                    ) from exc
                step_protected_inputs.append(
                    (
                        f"steps.{step.name}.inputs[{index}]",
                        input_path,
                        resolved_input,
                    )
                )
            for output in step.outputs:
                try:
                    resolved_output = self.resolve_path(output)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ManifestError(
                        f"steps.{step.name}.outputs path {output!r} cannot be "
                        f"resolved safely: {exc}"
                    ) from exc

                for logical_name, input_path, resolved_input in step_protected_inputs:
                    if _paths_overlap(resolved_output, resolved_input):
                        raise ManifestError(
                            f"selected step {step.name!r} output {output!r} "
                            f"overlaps protected input {logical_name!r} "
                            f"({input_path!r}); workflow outputs must not "
                            "modify the manifest or declared inputs"
                        )

                for prior_step, prior_output, resolved_prior in claimed_outputs:
                    same_resolved_path = resolved_output == resolved_prior
                    cross_step_overlap = (
                        prior_step != step.name
                        and _paths_overlap(resolved_output, resolved_prior)
                    )
                    if same_resolved_path or cross_step_overlap:
                        raise ManifestError(
                            f"selected steps {prior_step!r} and {step.name!r} "
                            "declare overlapping outputs "
                            f"{prior_output!r} and {output!r}; independent "
                            "output fingerprints would become stale"
                        )
                claimed_outputs.append((step.name, output, resolved_output))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "session_id": self.session_id,
            "views": {
                name: self.views[name].to_dict()
                for name in sorted(self.views)
            },
            "steps": {
                name: self.steps[name].to_dict()
                for name in sorted(self.steps)
            },
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete canonical manifest configuration."""

        return _sha256_json(self.to_dict())

    def step_config_fingerprint(self, step_name: str) -> str:
        """Fingerprint configuration that determines one step's behavior."""

        try:
            step = self.steps[step_name]
        except KeyError as exc:
            raise ManifestError(f"unknown workflow step {step_name!r}") from exc
        value = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "session_id": self.session_id,
            "metadata": dict(self.metadata),
            "views": {
                name: self.views[name].to_dict()
                for name in sorted(self.views)
            },
            "step": step.to_dict(),
        }
        return _sha256_json(value)

    def selected_steps(self) -> Tuple[StepSpec, ...]:
        """Return selected steps in deterministic dependency order."""

        ordered: list[StepSpec] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            step = self.steps[name]
            for dependency in step.depends_on:
                visit(dependency)
            visited.add(name)
            if step.selected:
                ordered.append(step)

        for name in self.steps:
            if self.steps[name].selected:
                visit(name)
        return tuple(ordered)

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return candidate.resolve()

    def iter_content_paths(self) -> Iterator[Tuple[str, str]]:
        """Yield all declared view inputs under stable logical names."""

        for name in sorted(self.views):
            yield from self.views[name].iter_content_paths()


def load_manifest(path: Path | str) -> StudyManifest:
    """Load and validate a JSON or YAML study/session manifest."""

    return StudyManifest.from_file(path)
