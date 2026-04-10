"""Grading-style checks for inference.py stdout format (no live LLM / env)."""

import io
import re
import sys
from contextlib import redirect_stdout

import inference as inf


def test_log_start_no_extra_quotes():
    buf = io.StringIO()
    with redirect_stdout(buf):
        inf.log_start(task="ticket_routing_basic", env="support_triage_openenv", model="gpt-4o-mini")
    line = buf.getvalue().strip()
    assert line == "[START] task=ticket_routing_basic env=support_triage_openenv model=gpt-4o-mini"


def test_log_step_format():
    buf = io.StringIO()
    with redirect_stdout(buf):
        inf.log_step(step=2, action='{"a":1}', reward=0.5, done=True, error=None)
        inf.log_step(step=3, action="x", reward=-0.1, done=False, error="boom")
    lines = buf.getvalue().strip().splitlines()
    assert lines[0] == '[STEP] step=2 action={"a":1} reward=0.50 done=true error=null'
    assert lines[1] == "[STEP] step=3 action=x reward=-0.10 done=false error=boom"


def test_log_end_format():
    buf = io.StringIO()
    with redirect_stdout(buf):
        inf.log_end(success=True, steps=3, score=1.0, rewards=[0.0, 0.5, 0.5])
    line = buf.getvalue().strip()
    assert line == "[END] success=true steps=3 score=1.00 rewards=0.00,0.50,0.50"


def test_one_line_action_newlines_collapsed():
    buf = io.StringIO()
    with redirect_stdout(buf):
        inf.log_step(step=1, action="a\nb", reward=0.0, done=False, error=None)
    assert "action=a b" in buf.getvalue()


def test_api_key_prefers_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    import importlib

    importlib.reload(inf)
    assert inf.API_KEY == "hf-test"
