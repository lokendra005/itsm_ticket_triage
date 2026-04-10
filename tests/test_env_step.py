"""Smoke tests for environment transitions (no HTTP)."""

import pytest

from support_triage_env.models import SupportAction
from support_triage_env.server.support_environment import SupportTriageEnvironment


@pytest.fixture
def env():
    return SupportTriageEnvironment()


def test_reset_and_finalize_easy(env):
    o0 = env.reset(task="ticket_routing_basic")
    assert o0.done is False
    assert o0.task_id == "ticket_routing_basic"
    o1 = env.step(
        SupportAction(
            message='{"priority":"P3","department":"billing","reason":"x","finalize":true}'
        )
    )
    assert o1.done is True
    assert o1.reward is not None
    assert env.state.terminal_grader_score == 1.0


def test_loop_penalty(env):
    env.reset(task="ticket_routing_basic")
    msg = '{"priority":"P2","department":"sales","finalize":false}'
    _ = env.step(SupportAction(message=msg))
    o2 = env.step(SupportAction(message=msg))
    assert o2.reward is not None
    assert env.state.loop_warning is True
