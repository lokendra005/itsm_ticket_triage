"""Scenario definitions for synthetic support-desk triage tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TriageTaskSpec:
    task_id: str
    difficulty: str  # easy | medium | hard
    ticket_id: str
    ticket_body: str
    instruction: str
    macro_menu: Optional[str]
    max_steps: int
    gold: Dict[str, Any]
    required_fields: List[str]


TASK_SPECS: Dict[str, TriageTaskSpec] = {
    "ticket_routing_basic": TriageTaskSpec(
        task_id="ticket_routing_basic",
        difficulty="easy",
        ticket_id="TKT-10492",
        ticket_body=(
            "Hi,\n\n"
            "Invoice #INV-4421 for our Starter plan doesn't match what we agreed in the "
            "last renewal email. Can someone on the finance side correct it before we "
            "close the quarter?\n\n"
            "Thanks,\nAlex (Ops Coordinator)"
        ),
        instruction=(
            "You are a support queue manager. Read the ticket and respond with a single JSON "
            "object (optionally wrapped in a markdown code block) containing:\n"
            '- "priority": one of P1, P2, P3, P4\n'
            '- "department": one of billing, support, security, sales\n'
            '- "reason": one short sentence\n'
            '- "finalize": false on intermediate thoughts, true when submitting your final routing\n'
            "Use P1 for emergencies/outages/security, P2 for revenue-impacting issues, "
            "P3 for standard customer issues, P4 for informational requests."
        ),
        macro_menu=None,
        max_steps=6,
        gold={
            "priority": "P3",
            "department": "billing",
        },
        required_fields=["priority", "department"],
    ),
    "macro_selection": TriageTaskSpec(
        task_id="macro_selection",
        difficulty="medium",
        ticket_id="TKT-8891",
        ticket_body=(
            "Subject: Can't access shared dashboard after role change\n\n"
            "Hello,\n"
            "I was promoted yesterday but I still see the old permissions on the analytics "
            "dashboard. I need this fixed today because I'm presenting to a customer at 4pm. "
            "I've already cleared cache and tried another browser.\n\n"
            "— Jordan"
        ),
        instruction=(
            "Reply planning must pick a support macro. Output JSON with:\n"
            '- "priority": P1–P4\n'
            '- "department": billing | support | security | sales\n'
            '- "macro_id": M1 | M2 | M3 | M4\n'
            '- "reason": short string\n'
            '- "finalize": boolean\n'
            "Choose the macro that best matches escalation + permission work; finalize when done."
        ),
        macro_menu=(
            "Macros:\n"
            "- M1: password_reset_self_serve — automated password / MFA reset links\n"
            "- M2: permissions_escalation — hand-off to identity team for RBAC changes (time-sensitive)\n"
            "- M3: billing_credit_template — invoices, refunds, plan mismatch\n"
            "- M4: generic_thanks — informational closing with no engineering work"
        ),
        max_steps=10,
        gold={
            "priority": "P2",
            "department": "support",
            "macro_id": "M2",
        },
        required_fields=["priority", "department", "macro_id"],
    ),
    "sla_vip_security": TriageTaskSpec(
        task_id="sla_vip_security",
        difficulty="hard",
        ticket_id="TKT-7710",
        ticket_body=(
            "Subject: URGENT — CEO locked out + strange alerts\n\n"
            "This is Maria (EA to CEO). He needs his SSO password reset immediately for the "
            "board deck, but I also see an alert on his phone about an "
            '"unauthorized access attempt" from an unknown location 20 minutes ago while he '
            "was in a closed-door meeting. SLA breach timer already shows red in the exec "
            "console — please advise simultaneously.\n\n"
            "Do NOT send the standard customer macro with self-serve links for this thread."
        ),
        instruction=(
            "Security-sensitive VIP escalation. Output JSON with:\n"
            '- "priority": P1–P4\n'
            '- "department": billing | support | security | sales\n'
            '- "macro_id": M1–M4 or null\n'
            '- "action": reply_with_macro | escalate_to_soc | schedule_callback\n'
            '- "notify_manager": true | false\n'
            "- short \"reason\"\n"
            '- "finalize": boolean\n'
            "You must avoid unsafe self-serve flows when indicators suggest account compromise. "
            "When finalize is true, all fields must be present."
        ),
        macro_menu=(
            "Macros (use carefully):\n"
            "- M1: password_reset_self_serve\n"
            "- M2: permissions_escalation\n"
            "- M3: billing_credit_template\n"
            "- M4: generic_thanks"
        ),
        max_steps=12,
        gold={
            "priority": "P1",
            "department": "security",
            "macro_id": None,
            "action": "escalate_to_soc",
            "notify_manager": True,
        },
        required_fields=["priority", "department", "macro_id", "action", "notify_manager"],
    ),
}


TASK_ORDER: List[str] = ["ticket_routing_basic", "macro_selection", "sla_vip_security"]
