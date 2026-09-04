"""Injectable, fingerprint-checked execution of study manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

from .manifest import StepSpec, StudyManifest
from .state import (
    RunState,
    RunStateError,
    StepRunState,
    StepStatus,
    fingerprint_path,
    utc_now,
)


class WorkflowExecutionError(RuntimeError):
    """Raised when a selected workflow step cannot complete safely."""


@dataclass(frozen=True)
class StepExecutionContext:
    """All model-independent information passed to an injected step."""

    manifest: StudyManifest
    step: StepSpec
    state_path: Path
    attempt: int
    input_paths: Tuple[Path, ...]
    output_paths: Tuple[Path, ...]

    def resolve_path(self, path: str) -> Path:
        return self.manifest.resolve_path(path)


StepExecutor = Callable[[StepExecutionContext], object]
ExecutorCollection = Union[StepExecutor, Mapping[str, StepExecutor]]


@dataclass(frozen=True)
class WorkflowRunResult:
    """Summary of work performed or safely reused in one invocation."""

    state: RunState
    executed_steps: Tuple[str, ...]
    reused_steps: Tuple[str, ...]
    skipped_steps: Tuple[str, ...]


class WorkflowRunner:
    """Execute selected manifest steps and persist conservative resume state."""

    def __init__(
        self,
        manifest: StudyManifest,
        state_path: Optional[Path | str] = None,
    ) -> None:
        self.manifest = manifest
        if state_path is None:
            if manifest.source_path is None:
                raise ValueError(
                    "state_path is required for manifests created in memory"
                )
            state_path = manifest.source_path.with_suffix(".run-state.json")
        self.state_path = Path(state_path).expanduser().resolve()
        self._validate_state_path()

    @staticmethod
    def _paths_overlap(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    def _validate_state_path(self) -> None:
        """Keep mutable run state outside every declared research artifact."""

        declarations: list[tuple[str, Path]] = []
        if self.manifest.source_path is not None:
            declarations.append(("manifest", self.manifest.source_path))
        declarations.extend(
            (
                logical_name,
                self.manifest.resolve_path(path),
            )
            for logical_name, path in self.manifest.iter_content_paths()
        )
        for step_name, step in self.manifest.steps.items():
            declarations.extend(
                (
                    f"steps.{step_name}.inputs[{index}]",
                    self.manifest.resolve_path(path),
                )
                for index, path in enumerate(step.inputs)
            )
            declarations.extend(
                (
                    f"steps.{step_name}.outputs[{index}]",
                    self.manifest.resolve_path(path),
                )
                for index, path in enumerate(step.outputs)
            )

        for logical_name, declared_path in declarations:
            if self._paths_overlap(self.state_path, declared_path):
                raise ValueError(
                    f"state_path {str(self.state_path)!r} overlaps declared "
                    f"path {logical_name!r} ({str(declared_path)!r}); run "
                    "state must be stored outside research inputs and outputs"
                )

    def _initial_state(self) -> RunState:
        return RunState(
            study_id=self.manifest.study_id,
            session_id=self.manifest.session_id,
            manifest_fingerprint=self.manifest.fingerprint,
            steps={
                name: StepRunState(
                    status=(
                        StepStatus.PENDING
                        if step.selected
                        else StepStatus.SKIPPED
                    )
                )
                for name, step in self.manifest.steps.items()
            },
        )

    def _load_state(self) -> RunState:
        if not self.state_path.exists():
            return self._initial_state()
        state = RunState.load(self.state_path)
        if (
            state.study_id != self.manifest.study_id
            or state.session_id != self.manifest.session_id
        ):
            raise RunStateError(
                "run state belongs to "
                f"{state.study_id}/{state.session_id}, not "
                f"{self.manifest.study_id}/{self.manifest.session_id}"
            )

        # Synchronize the current manifest without treating removed or newly
        # selected steps as previously completed.
        state.steps = {
            name: (
                state.steps[name]
                if name in state.steps
                else StepRunState(
                    status=(
                        StepStatus.PENDING
                        if step.selected
                        else StepStatus.SKIPPED
                    )
                )
            )
            for name, step in self.manifest.steps.items()
        }
        for name, step in self.manifest.steps.items():
            if not step.selected:
                state.steps[name] = StepRunState(status=StepStatus.SKIPPED)
        state.manifest_fingerprint = self.manifest.fingerprint
        return state

    def _input_fingerprints(
        self,
        step: StepSpec,
        state: RunState,
    ) -> Dict[str, str]:
        fingerprints = {
            logical_name: fingerprint_path(
                self.manifest.resolve_path(path)
            )
            for logical_name, path in self.manifest.iter_content_paths()
        }
        for index, path in enumerate(step.inputs):
            fingerprints[f"step.{step.name}.input.{index}"] = fingerprint_path(
                self.manifest.resolve_path(path)
            )
        for dependency_name in step.depends_on:
            dependency = self.manifest.steps[dependency_name]
            dependency_state = state.steps[dependency_name]
            if dependency_state.status is not StepStatus.COMPLETED:
                raise WorkflowExecutionError(
                    f"step {step.name!r} cannot run because dependency "
                    f"{dependency_name!r} is not completed"
                )
            current_outputs = self._output_fingerprints(dependency)
            if current_outputs != dependency_state.output_fingerprints:
                raise WorkflowExecutionError(
                    f"completed dependency {dependency_name!r} has changed "
                    "outputs"
                )
            for path, fingerprint in current_outputs.items():
                fingerprints[
                    f"dependency.{dependency_name}.output.{path}"
                ] = fingerprint
        return fingerprints

    def _output_fingerprints(self, step: StepSpec) -> Dict[str, str]:
        return {
            path: fingerprint_path(self.manifest.resolve_path(path))
            for path in step.outputs
        }

    def _can_resume(
        self,
        step: StepSpec,
        persisted: StepRunState,
        config_fingerprint: str,
        input_fingerprints: Mapping[str, str],
    ) -> bool:
        if persisted.status is not StepStatus.COMPLETED:
            return False
        if persisted.config_fingerprint != config_fingerprint:
            return False
        if persisted.input_fingerprints != dict(input_fingerprints):
            return False
        try:
            outputs = self._output_fingerprints(step)
        except (FileNotFoundError, OSError):
            return False
        return (
            bool(outputs)
            and set(outputs) == set(step.outputs)
            and outputs == persisted.output_fingerprints
        )

    @staticmethod
    def _executor_for(
        executors: ExecutorCollection,
        step_name: str,
    ) -> StepExecutor:
        if callable(executors):
            return executors
        try:
            executor = executors[step_name]
        except KeyError as exc:
            raise WorkflowExecutionError(
                f"no executor was supplied for selected step {step_name!r}"
            ) from exc
        if not callable(executor):
            raise WorkflowExecutionError(
                f"executor for step {step_name!r} is not callable"
            )
        return executor

    def run(self, executors: ExecutorCollection) -> WorkflowRunResult:
        """Run, resume, or safely skip every manifest step.

        A step is reused only when its status is ``completed``, its own config
        fingerprint and every declared input fingerprint match, and every
        declared output still exists with exactly the recorded content.  An
        executor returning normally is not sufficient for completion.
        """

        state = self._load_state()
        executed: list[str] = []
        reused: list[str] = []
        skipped = [
            name
            for name, step in self.manifest.steps.items()
            if not step.selected
        ]

        for step in self.manifest.selected_steps():
            persisted = state.steps[step.name]
            config_fingerprint = self.manifest.step_config_fingerprint(
                step.name
            )
            try:
                input_fingerprints = self._input_fingerprints(step, state)
            except Exception as exc:
                persisted.status = StepStatus.FAILED
                persisted.error = str(exc)
                persisted.completed_at = None
                persisted.output_fingerprints = {}
                state.write_atomic(self.state_path)
                if isinstance(exc, WorkflowExecutionError):
                    raise
                raise WorkflowExecutionError(
                    f"could not fingerprint inputs for step {step.name!r}: "
                    f"{exc}"
                ) from exc

            if self._can_resume(
                step,
                persisted,
                config_fingerprint,
                input_fingerprints,
            ):
                reused.append(step.name)
                continue

            # Invalidate an old completion claim before looking up or invoking
            # an executor.  A missing executor must never leave a stale
            # ``completed`` status behind after fingerprints stopped matching.
            persisted.status = StepStatus.PENDING
            persisted.config_fingerprint = config_fingerprint
            persisted.input_fingerprints = dict(input_fingerprints)
            persisted.output_fingerprints = {}
            persisted.started_at = None
            persisted.completed_at = None
            persisted.error = None
            state.write_atomic(self.state_path)

            try:
                executor = self._executor_for(executors, step.name)
            except WorkflowExecutionError as exc:
                persisted.status = StepStatus.FAILED
                persisted.error = str(exc)
                state.write_atomic(self.state_path)
                raise

            persisted.status = StepStatus.RUNNING
            persisted.attempts += 1
            persisted.started_at = utc_now()
            state.write_atomic(self.state_path)

            context = StepExecutionContext(
                manifest=self.manifest,
                step=step,
                state_path=self.state_path,
                attempt=persisted.attempts,
                input_paths=tuple(
                    self.manifest.resolve_path(path) for path in step.inputs
                ),
                output_paths=tuple(
                    self.manifest.resolve_path(path) for path in step.outputs
                ),
            )
            try:
                executor(context)
                output_fingerprints = self._output_fingerprints(step)
                if set(output_fingerprints) != set(step.outputs):
                    raise WorkflowExecutionError(
                        f"step {step.name!r} did not produce exactly its "
                        "declared outputs"
                    )
            except Exception as exc:
                persisted.status = StepStatus.FAILED
                persisted.error = str(exc)
                persisted.completed_at = None
                persisted.output_fingerprints = {}
                state.write_atomic(self.state_path)
                if isinstance(exc, WorkflowExecutionError):
                    raise
                raise WorkflowExecutionError(
                    f"step {step.name!r} failed: {exc}"
                ) from exc

            persisted.output_fingerprints = output_fingerprints
            persisted.status = StepStatus.COMPLETED
            persisted.completed_at = utc_now()
            persisted.error = None
            state.write_atomic(self.state_path)
            executed.append(step.name)

        # Persist synchronized skipped steps and a new manifest fingerprint even
        # when every selected step was safely reused.
        state.write_atomic(self.state_path)
        return WorkflowRunResult(
            state=state,
            executed_steps=tuple(executed),
            reused_steps=tuple(reused),
            skipped_steps=tuple(skipped),
        )
