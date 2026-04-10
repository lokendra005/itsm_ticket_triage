"""
Baseline inference for Support Triage OpenEnv.

MANDATORY env (per competition sample):
  API_BASE_URL     LLM endpoint (default: https://api.openai.com/v1); use validator-injected LiteLLM URL when present
  MODEL_NAME       Model id (default: gpt-4o-mini)
  API_KEY          Preferred LLM key (competition validators inject this for the LiteLLM proxy)
  HF_TOKEN         Fallback API key for local / HF (Hub token must NOT win over API_KEY for LLM calls)
  LOCAL_IMAGE_NAME Optional; docker image for from_docker_image() (alias: IMAGE_NAME, OPENENV_IMAGE)
  OPENENV_BASE_URL Space / server URL when not using docker (no trailing slash)

STDOUT (exact line types; reward/score 2 decimals; done/success lowercase true/false;
 rewards comma-separated; error=null or raw one-line string):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

from openai import OpenAI
from openenv.core.utils import run_async_safely

from support_triage_env import SupportAction, SupportTriageEnv, TASK_ORDER


def _competition_llm_env_ready() -> bool:
    """True when the harness injected both LiteLLM vars with non-empty values."""
    return bool(os.environ.get("API_BASE_URL", "").strip()) and bool(os.environ.get("API_KEY", "").strip())


def _sync_openai_sdk_env_from_competition_vars() -> None:
    """Normalize API_* and mirror to OPENAI_* so the SDK and LiteLLM harness see the same endpoint/key."""
    os.environ["API_BASE_URL"] = os.environ["API_BASE_URL"].strip()
    os.environ["API_KEY"] = os.environ["API_KEY"].strip()
    os.environ["OPENAI_BASE_URL"] = os.environ["API_BASE_URL"]
    os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


def build_openai_client() -> OpenAI:
    """
    LLM criteria check: use os.environ['API_BASE_URL'] and os.environ['API_KEY'].

    Local / HF dev: fall back when those are unset (see README).
    """
    if _competition_llm_env_ready():
        _sync_openai_sdk_env_from_competition_vars()
        # Literal subscripts match competition instructions; values normalized above.
        return OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )

    base = os.environ.get("API_BASE_URL", "").strip() or "https://api.openai.com/v1"
    key = (
        os.environ.get("API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )
    if not key:
        raise RuntimeError("Set API_KEY and API_BASE_URL (grading) or API_KEY/HF_TOKEN for local runs.")
    return OpenAI(base_url=base, api_key=key)


# Snapshot for tests / introspection (import-time; re-run importlib.reload after changing env).
if _competition_llm_env_ready():
    API_BASE_URL = os.environ["API_BASE_URL"].strip()
    API_KEY = os.environ["API_KEY"].strip()
else:
    API_BASE_URL = os.environ.get("API_BASE_URL", "").strip() or "https://api.openai.com/v1"
    API_KEY = (
        os.environ.get("API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = os.environ.get("OPENENV_BENCHMARK_NAME", "support_triage_openenv")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.65"))
OPENENV_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_NAME = (
    os.environ.get("LOCAL_IMAGE_NAME", "").strip()
    or os.environ.get("IMAGE_NAME", "").strip()
    or os.environ.get("OPENENV_IMAGE", "").strip()
)
USE_DOCKER = os.environ.get("OPENENV_USE_DOCKER", "").lower() in ("1", "true", "yes")


def _one_line(s: str) -> str:
    return re.sub(r"[\r\n]+", " ", s).strip()


def log_start(*, task: str, env: str, model: str) -> None:
    # No Python repr quotes — match sample: task=name env=name model=name
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    a = _one_line(action)
    r = f"{float(reward):.2f}"
    d = "true" if done else "false"
    if error is None:
        e = "null"
    else:
        e = _one_line(error)
    print(f"[STEP] step={step} action={a} reward={r} done={d} error={e}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    s = "true" if success else "false"
    parts: List[str] = []
    for x in rewards:
        try:
            parts.append(f"{float(x):.2f}")
        except (TypeError, ValueError):
            parts.append("0.00")
    rs = ",".join(parts)
    try:
        sc = f"{float(score):.2f}"
    except (TypeError, ValueError):
        sc = "0.00"
    print(f"[END] success={s} steps={int(steps)} score={sc} rewards={rs}", flush=True)


def _user_prompt_for_obs(obs, last_reward: float, history: List[str]) -> str:
    chunks = [
        f"ticket_id={obs.ticket_id}",
        f"task_id={obs.task_id} difficulty={obs.difficulty}",
        f"step={obs.step_number}/{obs.max_steps}",
        "--- TICKET ---",
        obs.ticket_body,
        "--- POLICY ---",
        obs.instruction,
    ]
    if obs.macro_menu:
        chunks.append("--- MACROS ---\n" + obs.macro_menu)
    chunks.append(f"environment_feedback={obs.feedback!r}")
    chunks.append(f"last_step_reward={last_reward:.2f}")
    if history:
        chunks.append("--- HISTORY ---\n" + "\n".join(history[-8:]))
    chunks.append(
        "Respond with a JSON object only (markdown fence allowed). "
        "Include finalize=true when submitting for grading."
    )
    return "\n\n".join(chunks)


def get_model_message(
    client: OpenAI,
    step: int,
    last_echoed: str,
    last_reward: float,
    history: List[str],
    obs,
) -> str:
    system = (
        "You are an expert support desk triage agent. "
        "Always ground decisions in the ticket text. "
        "Output valid JSON with the keys requested in POLICY; never invent SLA details beyond the ticket."
    )
    if obs is not None:
        user = _user_prompt_for_obs(obs, last_reward, history)
    else:
        user = f"Observation summary: {last_echoed}\nLast reward: {last_reward:.2f}\nStep: {step}"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        err = _one_line(str(exc))[:500]
        return json.dumps({"reason": "llm_error", "detail": err, "finalize": False})

    if not resp.choices:
        return '{"reason":"no_choices","finalize":false}'
    choice = resp.choices[0].message.content
    if not choice:
        return '{"reason":"empty_model","finalize":false}'
    return choice.strip()


async def run_one_task(task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[SupportTriageEnv] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        client = build_openai_client()

        if IMAGE_NAME:
            env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
        elif USE_DOCKER:
            raise RuntimeError("OPENENV_USE_DOCKER=1 requires LOCAL_IMAGE_NAME or IMAGE_NAME or OPENENV_IMAGE.")
        else:
            env = SupportTriageEnv(base_url=OPENENV_BASE_URL)
            await env.connect()

        result = await env.reset(task=task_name)
        last_echoed = result.observation.echoed_message
        last_reward = 0.0
        obs = result.observation
        max_steps = max(4, obs.max_steps)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            message = get_model_message(
                client, step, last_echoed, last_reward, history, obs
            )
            try:
                result = await env.step(SupportAction(message=message))
                err: Optional[str] = None
            except Exception as exc:
                err = str(exc)
                log_step(step=step, action=message, reward=0.0, done=False, error=err)
                history.append(f"Step {step}: error {err}")
                break

            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=None)

            history.append(f"Step {step}: reward {reward:.2f}")

            if done:
                break

        score = float(sum(rewards))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_one_task error: {exc}", file=sys.stderr, flush=True)
        success = False
        if not rewards:
            score = 0.0
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error (container cleanup): {exc}", file=sys.stderr, flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    for task in TASK_ORDER:
        await run_one_task(task)


if __name__ == "__main__":
    # Validators may invoke this script under an already-running event loop;
    # asyncio.run() raises RuntimeError in that case. OpenEnv provides a safe runner.
    run_async_safely(main())
