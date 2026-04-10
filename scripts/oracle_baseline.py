#!/usr/bin/env python3
"""
Deterministic upper-bound baseline: submits gold labels in one step per task.

No LLM, no Docker, no HF. Same score normalization as inference.py.
Prints [START]/[STEP]/[END] lines for parity with the competition log format.

Usage:
  python scripts/oracle_baseline.py
  .venv/bin/python scripts/oracle_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_triage_env import SupportAction, TASK_ORDER  # noqa: E402
from support_triage_env.server.support_environment import (  # noqa: E402
    SupportTriageEnvironment,
)
from support_triage_env.tasks import TASK_SPECS  # noqa: E402

BENCHMARK = "support_triage_openenv_oracle"
MODEL_NAME = "oracle_gold_policy"
SUCCESS_SCORE_THRESHOLD = 0.65


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task!r} env={env!r} model={model!r}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err_lit = "null" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step} action={action!r} reward={reward:+.4f} done={done} error={err_lit}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def gold_message(task_id: str) -> str:
    spec = TASK_SPECS[task_id]
    payload: Dict[str, Any] = {
        **{k: v for k, v in spec.gold.items()},
        "reason": "oracle gold submission",
        "finalize": True,
    }
    return json.dumps(payload)


def run_task_sync(task_name: str) -> float:
    env = SupportTriageEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env.reset(task=task_name)

    message = gold_message(task_name)
    step = 1
    obs = env.step(SupportAction(message=message))
    reward = float(obs.reward or 0.0)
    rewards.append(reward)
    steps_taken = step
    log_step(step=step, action=message, reward=reward, done=bool(obs.done), error=None)

    score = float(sum(rewards))
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    scores: Dict[str, float] = {}
    for task in TASK_ORDER:
        scores[task] = run_task_sync(task)
    print("\n### Oracle summary (copy into README if useful)", flush=True)
    easy = scores["ticket_routing_basic"]
    med = scores["macro_selection"]
    hard = scores["sla_vip_security"]
    print(
        f"| Oracle (gold policy, no LLM) | {easy:.4f} | {med:.4f} | {hard:.4f} "
        "| Upper bound; run `inference.py` for LLM baseline |"
    )


if __name__ == "__main__":
    main()
