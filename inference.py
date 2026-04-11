"""
Baseline inference for Support Triage OpenEnv.

Mandatory environment variables (injected by competition harness):
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
  IMAGE_NAME     Docker image for from_docker_image().
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import List, Optional

from openai import OpenAI

from support_triage_env import SupportAction, SupportTriageEnv, TASK_ORDER

# ── Env vars (match sample: HF_TOKEN first) ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")
BENCHMARK = os.getenv("OPENENV_BENCHMARK_NAME", "support_triage_openenv")
MAX_TOTAL_REWARD = float(os.getenv("MAX_TOTAL_REWARD", "1.0"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.65"))


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
        text = (resp.choices[0].message.content or "").strip()
        return text if text else '{"finalize":false}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"finalize":false}'


async def run_one_task(
    task_name: str, client: OpenAI, env: SupportTriageEnv
) -> None:
    """Run a single task episode. Matches sample structure: try/finally with log_end in finally."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
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

            result = await env.step(SupportAction(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # LocalDockerProvider maps host_port:8000 inside the container.
    # Pass PORT=8000 so uvicorn listens on 8000 (Dockerfile defaults to 7860 for HF Spaces).
    env = await SupportTriageEnv.from_docker_image(
        IMAGE_NAME, env_vars={"PORT": "8000"}
    )

    try:
        for task in TASK_ORDER:
            await run_one_task(task, client, env)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
