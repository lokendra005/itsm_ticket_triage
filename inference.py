"""
Baseline inference for Support Triage OpenEnv.

Structured logs (exact keys):
  [START] task=... env=... model=...
  [STEP] step=... action=... reward=... done=... error=...
  [END] success=... steps=... score=... rewards=...

Environment variables:
  API_BASE_URL      LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME        Model id (default: gpt-4o-mini)
  OPENAI_API_KEY    Preferred API key for OpenAI-compatible servers
  HF_TOKEN          Fallback API key if OPENAI_API_KEY is unset
  OPENENV_BASE_URL  Running Space / server URL (http(s)://...) when not using Docker
  IMAGE_NAME        Same as OPENENV_IMAGE (sample / eval compatibility)
  OPENENV_IMAGE     Local image tag for from_docker_image()
  OPENENV_USE_DOCKER Set to 1 to force Docker even if IMAGE_NAME unset
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

from openai import OpenAI

from support_triage_env import SupportAction, SupportTriageEnv, TASK_ORDER

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
BENCHMARK = os.environ.get("OPENENV_BENCHMARK_NAME", "support_triage_openenv")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.65"))
OPENENV_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_NAME = (
    os.environ.get("IMAGE_NAME", "").strip()
    or os.environ.get("OPENENV_IMAGE", "").strip()
)
USE_DOCKER = os.environ.get("OPENENV_USE_DOCKER", "").lower() in ("1", "true", "yes")


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task!r} env={env!r} model={model!r}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_lit = "null" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step} action={action!r} reward={reward:+.4f} done={done} error={err_lit}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


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
    chunks.append(f"last_step_reward={last_reward:+.4f}")
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
        user = f"Observation summary: {last_echoed!r}\nLast reward: {last_reward:+.4f}\nStep: {step}"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0].message.content
    if not choice:
        return '{"reason":"empty_model","finalize":false}'
    return choice.strip()


async def run_one_task(task_name: str) -> None:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set for inference.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env: SupportTriageEnv
    if IMAGE_NAME:
        env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
    elif USE_DOCKER:
        raise RuntimeError("OPENENV_USE_DOCKER=1 requires IMAGE_NAME or OPENENV_IMAGE.")
    else:
        env = SupportTriageEnv(base_url=OPENENV_BASE_URL)
        await env.connect()

    try:
        result = await env.reset(task=task_name)
        last_echoed = result.observation.echoed_message
        last_reward = 0.0
        obs = result.observation

        # max steps from observation after reset (task-specific)
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
                history.append(f"Step {step}: error {err!r}")
                break

            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=None)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # Episode score = cumulative shaped reward, clamped (same scale as per-step rewards).
        score = float(sum(rewards))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    for task in TASK_ORDER:
        await run_one_task(task)


if __name__ == "__main__":
    asyncio.run(main())
