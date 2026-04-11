"""Unit tests for deterministic graders."""

from support_triage_env.graders import grade_submission, triage_reward_breakdown
from support_triage_env.tasks import TASK_SPECS


def test_easy_perfect():
    task = TASK_SPECS["ticket_routing_basic"]
    sub = {
        "priority": "P3",
        "department": "billing",
        "reason": "invoice mismatch",
        "finalize": True,
    }
    assert grade_submission(task, sub) == 0.99
    assert triage_reward_breakdown(task, sub).scalar == 0.99


def test_medium_perfect():
    task = TASK_SPECS["macro_selection"]
    sub = {
        "priority": "P2",
        "department": "support",
        "macro_id": "M2",
        "reason": "rbac",
        "finalize": True,
    }
    assert grade_submission(task, sub) == 0.99


def test_hard_wrong_macro_safe():
    task = TASK_SPECS["sla_vip_security"]
    sub_bad = {
        "priority": "P1",
        "department": "security",
        "macro_id": "M1",
        "action": "reply_with_macro",
        "notify_manager": True,
        "finalize": True,
    }
    bd = triage_reward_breakdown(task, sub_bad)
    assert bd.safety_penalty < 0
    assert bd.scalar < 0.75


def test_hard_perfect():
    task = TASK_SPECS["sla_vip_security"]
    sub = {
        "priority": "P1",
        "department": "security",
        "macro_id": None,
        "action": "escalate_to_soc",
        "notify_manager": True,
        "finalize": True,
    }
    assert grade_submission(task, sub) == 0.99
