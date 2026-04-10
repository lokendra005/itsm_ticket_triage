"""Typed Action, Observation, State, and Reward models for support ticket triage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class SupportAction(Action):
    """Natural-language message; agent should embed a JSON triage decision in the text."""

    message: str = Field(..., min_length=1, description="Utterance containing structured triage JSON")


class TriageReward(BaseModel):
    """Structured reward decomposition (0.0–1.0 components; scalar is weighted total)."""

    model_config = {"extra": "forbid"}

    scalar: float = Field(..., ge=0.0, le=1.0, description="Overall score for grading / logging")
    priority_match: float = Field(default=0.0, ge=0.0, le=1.0)
    department_match: float = Field(default=0.0, ge=0.0, le=1.0)
    macro_match: float = Field(default=0.0, ge=0.0, le=1.0)
    action_match: float = Field(default=0.0, ge=0.0, le=1.0)
    notify_match: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_penalty: float = Field(default=0.0, le=0.0, description="Non-positive penalty term")


class SupportObservation(Observation):
    """What the agent sees after reset or step."""

    echoed_message: str = Field(
        ...,
        description="Compact natural-language summary for logging and baseline agents",
    )
    instruction: str = Field(..., description="Task instructions and output format")
    ticket_id: str = Field(..., description="Synthetic ticket identifier")
    ticket_body: str = Field(..., description="Customer message to triage")
    task_id: str = Field(..., description="Task / scenario identifier")
    difficulty: str = Field(..., description="easy | medium | hard")
    step_number: int = Field(default=0, ge=0)
    max_steps: int = Field(..., ge=1)
    feedback: str = Field(default="", description="Shaping feedback from the last transition")
    macro_menu: Optional[str] = Field(
        default=None, description="Optional macro cheat-sheet for medium/hard scenarios"
    )
    reward_hint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured hint from last step (e.g. partial credit breakdown)",
    )


class SupportState(State):
    """Serializable server-side session state."""

    task_id: str = ""
    last_parse_ok: bool = False
    loop_warning: bool = False
    terminal_grader_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cumulative_reward: float = 0.0

