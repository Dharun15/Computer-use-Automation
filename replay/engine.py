"""
ReplayEngine -- the production execution path.

    artifact + inputs -> deterministic actions -> checkpoint -> outputs

No LLM call happens anywhere in this file. Every action was already decided
at discovery time; replay's only job is to substitute real input values
into the recorded template, execute the exact same steps in the exact same
order through the exact same `ToolExecutor`/`ComputerSurface` stack
discovery used, verify the checkpoint, and return a structured result.

Phase 7 adds the full error taxonomy on top of Phase 6's plain success/fail:
  - a bounded number of automatic retries for transient (`timeout`) failures
    -- the "recoverable condition" bucket -- logged in `recovery_attempts`
    whether or not they end up mattering;
  - failure classification (see `replay/classifier.py`) into
    BUSINESS_OUTCOME / HUMAN_INTERVENTION / HARD_FAILURE, based on the
    actual page reached, not just the raw action status;
  - pause/resume support for HUMAN_INTERVENTION: `replay()` can return a
    paused result mid-artifact, and `resume()` continues the SAME replay
    (same surface, same session, same outputs collected so far) from
    exactly where it stopped once a human has intervened.

Reusing `ToolExecutor` here (not talking to the surface directly) is
deliberate: a step is validated identically whether it came from an LLM's
tool call during discovery or from a saved artifact during replay.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from artifacts.schema import (
    Artifact,
    ArtifactClickStep,
    ArtifactExtractStep,
    ArtifactNavigateStep,
    ArtifactTypeStep,
)
from replay.checkpoints import verify_checkpoint
from replay.classifier import classify_failure
from replay.errors import OutcomeCode, RecoveryAttempt, ReplayResult, ReplayStatus
from handoff.manager import HandoffManager
from replay.locator import substitute_string, substitute_target
from surface.observation import Observation
from tools.executor import ToolExecutor

_RETRIABLE_STATUSES = {"timeout"}


class ReplayEngine:
    def __init__(
        self,
        executor: ToolExecutor,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        handoff: Optional[HandoffManager] = None,
    ):
        self._executor = executor
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._handoff = handoff

    def replay(self, artifact: Artifact, inputs: dict[str, Any]) -> ReplayResult:
        input_error = self._validate_inputs(artifact, inputs)
        if input_error:
            return ReplayResult(
                status=ReplayStatus.HARD_FAILURE,
                outcome_code=OutcomeCode.UNKNOWN,
                total_steps=len(artifact.steps),
                message=input_error,
            )
        return self._run_from(artifact, inputs, start_index=0, outputs={})

    def resume(self, artifact: Artifact, inputs: dict[str, Any], paused: ReplayResult) -> ReplayResult:
        """Continue a replay that previously returned HUMAN_INTERVENTION,
        on the assumption a human has just acted on the SAME live session
        (see handoff/manager.py) to clear whatever was blocking progress."""
        if paused.status != ReplayStatus.HUMAN_INTERVENTION:
            raise ValueError(
                "resume() is only valid for a result with status=HUMAN_INTERVENTION "
                f"(got {paused.status.value})."
            )
        return self._run_from(
            artifact, inputs, start_index=paused.steps_completed, outputs=dict(paused.outputs)
        )

    # ------------------------------------------------------------- core loop

    def _run_from(
        self,
        artifact: Artifact,
        inputs: dict[str, Any],
        start_index: int,
        outputs: dict[str, Any],
    ) -> ReplayResult:
        recovery_attempts: list[RecoveryAttempt] = []

        for i in range(start_index, len(artifact.steps)):
            step = artifact.steps[i]
            try:
                tool_input = self._build_tool_input(step, inputs)
            except ValueError as e:
                return ReplayResult(
                    status=ReplayStatus.HARD_FAILURE,
                    outcome_code=OutcomeCode.UNKNOWN,
                    outputs=outputs,
                    steps_completed=i,
                    total_steps=len(artifact.steps),
                    failed_step_id=step.id,
                    failed_step_action=step.action,
                    message=f"Failed to prepare step '{step.id}': {e}",
                )

            result = self._execute_with_retries(step, tool_input, recovery_attempts)

            if (not result.valid) or result.status != "ok":
                return self._fail(
                    artifact, outputs, i, step, result.validation_error or result.message,
                    recovery_attempts,
                )

            if isinstance(step, ArtifactExtractStep):
                outputs[step.output] = self._coerce_output(
                    result.data, artifact.outputs.get(step.output)
                )

        checkpoint_result_obs = self._executor.execute("observe_page", {})
        if checkpoint_result_obs.observation is None:
            return ReplayResult(
                status=ReplayStatus.HARD_FAILURE,
                outcome_code=OutcomeCode.UNKNOWN,
                outputs=outputs,
                steps_completed=len(artifact.steps),
                total_steps=len(artifact.steps),
                message="Could not observe final page state to verify checkpoint.",
                recovery_attempts=recovery_attempts,
            )

        final_observation = Observation(**checkpoint_result_obs.observation)
        checkpoint = verify_checkpoint(artifact.checkpoint, final_observation)

        if not checkpoint.passed:
            # Even though every step reported "ok", the workflow may have
            # ended up somewhere recognizable (a business outcome / hard
            # failure page) rather than the expected end state -- classify
            # it the same way a mid-step failure would be, before falling
            # back to a generic checkpoint-mismatch failure.
            classification = classify_failure(final_observation, "")
            if classification.outcome_code != OutcomeCode.UNKNOWN:
                return ReplayResult(
                    status=classification.status,
                    outcome_code=classification.outcome_code,
                    outputs=outputs,
                    steps_completed=len(artifact.steps),
                    total_steps=len(artifact.steps),
                    message=classification.message,
                    expected=checkpoint.expected,
                    observed=checkpoint.observed,
                    recovery_attempts=recovery_attempts,
                )
            return ReplayResult(
                status=ReplayStatus.HARD_FAILURE,
                outcome_code=OutcomeCode.CHECKPOINT_NOT_SATISFIED,
                outputs=outputs,
                steps_completed=len(artifact.steps),
                total_steps=len(artifact.steps),
                message="Checkpoint was not satisfied after all steps completed.",
                expected=checkpoint.expected,
                observed=checkpoint.observed,
                recovery_attempts=recovery_attempts,
            )

        return ReplayResult(
            status=ReplayStatus.SUCCESS,
            outputs=outputs,
            steps_completed=len(artifact.steps),
            total_steps=len(artifact.steps),
            message="Replay completed successfully.",
            recovery_attempts=recovery_attempts,
        )

    # ------------------------------------------------------------- helpers

    def _execute_with_retries(self, step, tool_input: dict, recovery_attempts: list[RecoveryAttempt]):
        result = self._executor.execute(step.action, tool_input)
        attempt = 0
        while result.valid and result.status in _RETRIABLE_STATUSES and attempt < self.max_retries:
            attempt += 1
            time.sleep(self.retry_backoff_seconds)
            retried = self._executor.execute(step.action, tool_input)
            outcome = "retried_ok" if (retried.valid and retried.status == "ok") else "retried_failed"
            recovery_attempts.append(
                RecoveryAttempt(step_id=step.id, attempt=attempt, reason="timeout", outcome=outcome)
            )
            result = retried
        if result.valid and result.status in _RETRIABLE_STATUSES and attempt >= self.max_retries:
            if recovery_attempts and recovery_attempts[-1].step_id == step.id:
                recovery_attempts[-1].outcome = "exhausted"
        return result

    def _fail(self, artifact, outputs, step_index, step, raw_message, recovery_attempts):
        fresh = self._executor.execute("observe_page", {})
        observation = Observation(**fresh.observation) if fresh.observation else None
        classification = classify_failure(observation, raw_message)

        screenshot_path = None
        shot = self._executor.execute("screenshot", {})
        if shot.valid and shot.screenshot_path:
            screenshot_path = shot.screenshot_path

        intervention_dict = None
        if classification.status == ReplayStatus.HUMAN_INTERVENTION and self._handoff is not None:
            request = self._handoff.request_intervention(
                reason=classification.outcome_code.value,
                message=classification.message,
                artifact_id=artifact.artifact_id,
                step_id=step.id,
                screenshot_path=screenshot_path,
                context={
                    "url": observation.url if observation else None,
                    "title": observation.title if observation else None,
                },
            )
            intervention_dict = request.model_dump()

        return ReplayResult(
            status=classification.status,
            outcome_code=classification.outcome_code,
            outputs=outputs,
            steps_completed=step_index,
            total_steps=len(artifact.steps),
            failed_step_id=step.id,
            failed_step_action=step.action,
            message=classification.message,
            expected="action to complete with status 'ok'",
            observed=f"{raw_message}" + (f" | page={observation.title!r}" if observation else ""),
            recovery_attempts=recovery_attempts,
            screenshot_path=screenshot_path,
            intervention=intervention_dict,
        )

    @staticmethod
    def _validate_inputs(artifact: Artifact, inputs: dict[str, Any]) -> Optional[str]:
        for name, spec in artifact.inputs.items():
            if spec.required and name not in inputs:
                return f"Missing required input '{name}'."
            if name in inputs and spec.type == "number":
                try:
                    float(str(inputs[name]).replace("$", "").replace(",", ""))
                except ValueError:
                    return f"Input '{name}' must be a number, got {inputs[name]!r}."
            if name in inputs and spec.type == "boolean" and not isinstance(inputs[name], bool):
                return f"Input '{name}' must be a boolean, got {inputs[name]!r}."
        return None

    @staticmethod
    def _build_tool_input(step, inputs: dict[str, Any]) -> dict:
        if isinstance(step, ArtifactNavigateStep):
            return {"url": substitute_string(step.url, inputs)}
        if isinstance(step, ArtifactClickStep):
            return {"target": substitute_target(step.target, inputs).model_dump(exclude_none=True)}
        if isinstance(step, ArtifactTypeStep):
            return {
                "target": substitute_target(step.target, inputs).model_dump(exclude_none=True),
                "value": substitute_string(step.value, inputs),
            }
        if isinstance(step, ArtifactExtractStep):
            return {"target": substitute_target(step.target, inputs).model_dump(exclude_none=True)}
        raise ValueError(f"No replay handler for step type: {type(step).__name__}")

    @staticmethod
    def _coerce_output(value: Any, spec) -> Any:
        if spec is None or value is None:
            return value
        if spec.type == "number":
            try:
                return float(str(value).replace("$", "").replace(",", ""))
            except ValueError:
                return value
        if spec.type == "boolean":
            return str(value).strip().lower() in ("true", "1", "yes")
        return value
