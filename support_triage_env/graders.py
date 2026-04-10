"""Deterministic agent graders: map submitted triage payloads to scores in [0.0, 1.0]."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .tasks import TriageTaskSpec


def _norm_macro(value: Any) -> Optional[str]:
    if value is None or value == "null":
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        if v in ("", "NONE", "NULL"):
            return None
        return v
    return None


def _norm_action(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip()


def _norm_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes"):
        return True
    if value in (0, "0", "false", "False", "no"):
        return False
    return None


def grade_submission(task: TriageTaskSpec, submission: Optional[Dict[str, Any]]) -> float:
    """Return aggregate score in [0, 1] for the final submission."""
    if not submission:
        return 0.0

    gold = task.gold
    parts: List[float] = []

    pr = str(submission.get("priority", "")).strip().upper()
    parts.append(1.0 if pr == gold["priority"] else 0.0)

    dep = str(submission.get("department", "")).strip().lower()
    gdep = str(gold["department"]).strip().lower()
    parts.append(1.0 if dep == gdep else 0.0)

    if "macro_id" in gold:
        sm = _norm_macro(submission.get("macro_id"))
        gm = gold["macro_id"]
        if gm is None:
            parts.append(1.0 if sm is None else 0.0)
        else:
            parts.append(1.0 if sm == str(gm).strip().upper() else 0.0)

    if "action" in gold:
        sa = _norm_action(submission.get("action"))
        ga = _norm_action(gold.get("action"))
        parts.append(1.0 if sa == ga else 0.0)

    if "notify_manager" in gold:
        sb = _norm_bool(submission.get("notify_manager"))
        gb = gold["notify_manager"]
        if isinstance(gb, bool) and sb is not None:
            parts.append(1.0 if sb is gb else 0.0)
        else:
            parts.append(0.0)

    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def triage_reward_breakdown(task: TriageTaskSpec, submission: Optional[Dict[str, Any]]) -> "TriageReward":
    """Expose per-field breakdown as TriageReward for logging / metadata."""
    from .models import TriageReward

    if not submission:
        return TriageReward(scalar=0.0)

    gold = task.gold
    priority_match = (
        1.0 if str(submission.get("priority", "")).strip().upper() == gold["priority"] else 0.0
    )
    department_match = (
        1.0
        if str(submission.get("department", "")).strip().lower()
        == str(gold["department"]).strip().lower()
        else 0.0
    )
    macro_match = 0.0
    if "macro_id" in gold:
        sm = _norm_macro(submission.get("macro_id"))
        gm = gold["macro_id"]
        if gm is None:
            macro_match = 1.0 if sm is None else 0.0
        else:
            macro_match = 1.0 if sm == str(gm).strip().upper() else 0.0

    action_match = 0.0
    if "action" in gold:
        sa = _norm_action(submission.get("action"))
        ga = _norm_action(gold.get("action"))
        action_match = 1.0 if sa == ga else 0.0

    notify_match = 0.0
    if "notify_manager" in gold:
        sb = _norm_bool(submission.get("notify_manager"))
        gb = gold["notify_manager"]
        notify_match = 1.0 if isinstance(gb, bool) and sb is not None and sb is gb else 0.0

    safety_penalty = 0.0
    if task.task_id == "sla_vip_security":
        sm = _norm_macro(submission.get("macro_id"))
        act = _norm_action(submission.get("action"))
        if act == "reply_with_macro" and sm in ("M1", "M4"):
            safety_penalty -= 0.35

    weights = [priority_match, department_match]
    if "macro_id" in gold:
        weights.append(macro_match)
    if "action" in gold:
        weights.append(action_match)
    if "notify_manager" in gold:
        weights.append(notify_match)
    scalar = max(0.0, min(1.0, sum(weights) / len(weights) + safety_penalty))

    return TriageReward(
        scalar=scalar,
        priority_match=priority_match,
        department_match=department_match,
        macro_match=macro_match,
        action_match=action_match,
        notify_match=notify_match,
        safety_penalty=safety_penalty,
    )


def partial_step_credit(task: TriageTaskSpec, submission: Optional[Dict[str, Any]]) -> float:
    """Smaller per-step signal in [0, ~0.25] for correct fields before finalize."""
    if not submission:
        return 0.0
    breakdown = triage_reward_breakdown(task, submission)
    # Only positive components for shaping; omit safety penalty until finalize path
    pos = (
        breakdown.priority_match * 0.08
        + breakdown.department_match * 0.08
        + (breakdown.macro_match * 0.07 if "macro_id" in task.gold else 0.0)
        + (breakdown.action_match * 0.06 if "action" in task.gold else 0.0)
        + (breakdown.notify_match * 0.05 if "notify_manager" in task.gold else 0.0)
    )
    return min(0.22, pos)
