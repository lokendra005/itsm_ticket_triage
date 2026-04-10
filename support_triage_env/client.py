"""WebSocket client for remote or containerized support triage environments."""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from support_triage_env.models import SupportAction, SupportObservation, SupportState


class SupportTriageEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    def _step_payload(self, action: SupportAction) -> dict:
        return {"message": action.message}

    def _parse_result(self, payload: dict) -> StepResult[SupportObservation]:
        obs = SupportObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> SupportState:
        return SupportState.model_validate(payload)
