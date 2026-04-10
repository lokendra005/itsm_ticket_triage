"""Support desk triage simulation — routes tickets using structured agent messages."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from support_triage_env.graders import (
    _norm_bool,
    grade_submission,
    partial_step_credit,
    triage_reward_breakdown,
)
from support_triage_env.models import SupportAction, SupportObservation, SupportState
from support_triage_env.tasks import TASK_ORDER, TASK_SPECS, TriageTaskSpec


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object from free text or a fenced block."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _finalize_fields_ok(task: TriageTaskSpec, parsed: Dict[str, Any]) -> bool:
    for key in task.required_fields:
        if key not in parsed:
            return False
        val = parsed[key]
        if key == "macro_id" and task.gold.get("macro_id") is None:
            continue
        if key == "notify_manager":
            if _norm_bool(val) is None:
                return False
            continue
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
    return True


class SupportTriageEnvironment(Environment):
    """Simulates helpdesk triage with partial credit and terminal graders."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__(transform=None, rubric=None)
        self._state = SupportState()
        self._task: Optional[TriageTaskSpec] = None
        self._step_num = 0
        self._last_message_hash: Optional[str] = None
        self._last_submission: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        self._reset_rubric()
        tid = task if task in TASK_SPECS else TASK_ORDER[0]
        self._task = TASK_SPECS[tid]
        self._step_num = 0
        self._last_message_hash = None
        self._last_submission = None
        self._state = SupportState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=tid,
            last_parse_ok=False,
            loop_warning=False,
            terminal_grader_score=None,
            cumulative_reward=0.0,
        )
        spec = self._task
        body = spec.ticket_body
        shortened = body if len(body) <= 160 else body[:157] + "..."
        echoed = f"{spec.ticket_id} ({spec.difficulty}): {shortened}"
        return SupportObservation(
            echoed_message=echoed,
            instruction=spec.instruction,
            ticket_id=spec.ticket_id,
            ticket_body=spec.ticket_body,
            task_id=spec.task_id,
            difficulty=spec.difficulty,
            step_number=0,
            max_steps=spec.max_steps,
            feedback="Reply with triage JSON. Use finalize=true only when all required fields are set.",
            macro_menu=spec.macro_menu,
            reward_hint=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: SupportAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SupportObservation:
        if not isinstance(action, SupportAction):
            raise ValueError(f"Expected SupportAction, got {type(action)}")
        assert self._task is not None

        self._step_num += 1
        self._state.step_count = self._step_num

        raw = action.message.strip()
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        loop_penalty = 0.0
        if self._last_message_hash == digest:
            loop_penalty = -0.12
            self._state.loop_warning = True
        else:
            self._state.loop_warning = False
        self._last_message_hash = digest

        parsed = extract_json_object(raw)
        feedback_bits = []
        step_reward = loop_penalty

        if parsed is None:
            self._state.last_parse_ok = False
            feedback_bits.append("Parse error: embed a JSON object with your triage fields.")
            step_reward -= 0.05
            breakdown_dict = None
        else:
            self._state.last_parse_ok = True
            self._last_submission = parsed
            step_reward += partial_step_credit(self._task, parsed)
            bd = triage_reward_breakdown(self._task, parsed)
            breakdown_dict = bd.model_dump()
            feedback_bits.append(
                f"JSON ok — field overlap score ~{bd.scalar:.2f} (not final until you set finalize=true)."
            )

        done = False
        terminal_score: Optional[float] = None

        if parsed and bool(parsed.get("finalize") is True):
            if _finalize_fields_ok(self._task, parsed):
                terminal_score = grade_submission(self._task, parsed)
                self._state.terminal_grader_score = terminal_score
                step_reward += terminal_score * 0.85
                done = True
                feedback_bits.append(
                    f"Final submission graded: {terminal_score:.2f} / 1.00 per task rubric."
                )
            else:
                step_reward -= 0.06
                feedback_bits.append(
                    "finalize=true but missing/invalid required fields — fix keys before finalizing."
                )

        if self._step_num >= self._task.max_steps and not done:
            done = True
            terminal_score = grade_submission(self._task, self._last_submission)
            self._state.terminal_grader_score = terminal_score
            step_reward += terminal_score * 0.55
            feedback_bits.append(
                f"Step budget exhausted — graded last structured attempt: {terminal_score:.2f} / 1.00."
            )

        step_reward = float(max(-0.35, min(0.95, step_reward)))
        self._state.cumulative_reward += step_reward

        spec = self._task
        body = spec.ticket_body
        shortened = body if len(body) <= 120 else body[:117] + "..."
        echoed = f"step {self._step_num}/{spec.max_steps} | {spec.ticket_id} | {shortened}"

        obs = SupportObservation(
            echoed_message=echoed,
            instruction=spec.instruction,
            ticket_id=spec.ticket_id,
            ticket_body=spec.ticket_body,
            task_id=spec.task_id,
            difficulty=spec.difficulty,
            step_number=self._step_num,
            max_steps=spec.max_steps,
            feedback=" ".join(feedback_bits),
            macro_menu=spec.macro_menu,
            reward_hint=breakdown_dict,
            done=done,
            reward=step_reward,
            metadata=(
                {
                    "terminal_grader_score": terminal_score,
                    "loop_penalty": loop_penalty,
                }
                if terminal_score is not None
                else {"loop_penalty": loop_penalty}
            ),
        )
        return obs

    @property
    def state(self) -> SupportState:
        self._state.step_count = self._step_num
        if self._task is not None:
            self._state.task_id = self._task.task_id
        return self._state
