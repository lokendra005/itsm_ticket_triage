"""
Baseline inference for Support Triage OpenEnv.

Uses the OpenAI client with:
  base_url = os.environ["API_BASE_URL"]
  api_key  = os.environ["API_KEY"]

Participants must use OpenAI Client for all LLM calls using above variables.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import List, Optional

from openai import OpenAI

from support_triage_env import SupportAction, SupportTriageEnv, TASK_ORDER

# ── LLM config (competition injects API_BASE_URL + API_KEY) ──────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

# ── Env config ────────────────────────────────────────────────────────────────
BENCHMARK = os.getenv("OPENENV_BENCHMARK_NAME", "support_triage_openenv")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.65"))
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or os.getenv("OPENENV_IMAGE")


# ── Stdout helpers (strict format for automated scoring) ──────────────────────

def _one_line(s: str) -> str:
    return re.sub(r"[\r\n]+", " ", s).strip()


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    a = _one_line(action)
    r = f"{float(reward):.2f}"
    d = "true" if done else "false"
    e = "null" if error is None else _one_line(error)
    print(f"[STEP] step={step} action={a} reward={r} done={d} error={e}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    s = "true" if success else "false"
    rs = ",".join(f"{float(x):.2f}" for x in rewards)
    sc = f"{float(score):.2f}"
    print(f"[END] success={s} steps={steps} score={sc} rewards={rs}", flush=True)


# ── LLM call (NO silent error swallowing) ─────────────────────────────────────

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

    # Do NOT catch exceptions here — if the proxy is unreachable or the key
    # is wrong, we MUST fail loudly so the grading harness sees real errors
    # instead of silently running steps with dummy responses.
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


# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_one_task(task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[SupportTriageEnv] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Exactly as the competition instructions require:
        #   OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )

        if IMAGE_NAME:
            env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
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
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        success = False
        if not rewards:
            score = 0.0
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    # Debug: show which LLM endpoint and key prefix we're actually using
    print(
        f"[CONFIG] API_BASE_URL={os.environ.get('API_BASE_URL', '<unset>')!r} "
        f"API_KEY={os.environ.get('API_KEY', '<unset>')[:8] + '...' if os.environ.get('API_KEY') else '<unset>'!r} "
        f"MODEL_NAME={MODEL_NAME!r} "
        f"OPENENV_BASE_URL={OPENENV_BASE_URL!r} "
        f"IMAGE_NAME={IMAGE_NAME!r}",
        file=sys.stderr,
        flush=True,
    )
    for task in TASK_ORDER:
        await run_one_task(task)


if __name__ == "__main__":
    asyncio.run(main())
