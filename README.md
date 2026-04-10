---
title: Support Triage OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

# Support Desk Ticket Triage (OpenEnv)

Realistic **helpdesk / ITSM triage** simulation: the agent reads customer tickets, applies queue policy, chooses macro templates, and (on hard scenarios) handles **VIP + security** constraints. The environment follows the OpenEnv contract (`reset` / `step` / `state`) with Pydantic **`SupportAction`**, **`SupportObservation`**, **`SupportState`**, and a structured **`TriageReward`** model used by deterministic graders.

## Why this task exists

Teams route thousands of tickets a day under SLAs. Automating **priority**, **queue**, **macros**, and **escalation** safely is a production problem—not a game. This env is meant for evaluating agents on structured decisions, partial credit, and penalties for risky shortcuts (for example, self-serve password macros during possible account takeover).

## Action space

- **Type:** `SupportAction`
- **Fields:** `message: str` — natural language utterance that **must contain a JSON object** (plain or fenced markdown) with triage keys described in the observation `instruction`.

## Observation space

- **Type:** `SupportObservation`
- **Primary fields:** `echoed_message` (short text for logging), `instruction`, `ticket_id`, `ticket_body`, `task_id`, `difficulty`, `step_number`, `max_steps`, `feedback`, optional `macro_menu`, optional `reward_hint`.
- **Gym-style signals:** `reward`, `done` (inherited from OpenEnv `Observation`).

## State

- **Type:** `SupportState` — `episode_id`, `step_count`, `task_id`, `last_parse_ok`, `loop_warning`, `terminal_grader_score`, `cumulative_reward`.

## Tasks & difficulty

| Task id | Difficulty | Objective |
| --- | --- | --- |
| `ticket_routing_basic` | Easy | Set `priority` + `department` for an invoice/plan mismatch ticket. |
| `macro_selection` | Medium | Add correct `macro_id` for a permissions/time-sensitive case. |
| `sla_vip_security` | Hard | Full triage under VIP + breach + compromise language; **avoid unsafe self-serve** paths; SOC escalation fields required. |

Each task exposes a **programmatic grader** returning **0.0–1.0** as the mean of required field matches (plus explicit **safety penalties** on the hard task).

## Reward design

- **Shaping:** partial credit each step via field overlap (`partial_step_credit`).
- **Terminal:** submitting `finalize=true` with valid required keys adds roughly `terminal_grader_score * 0.85` to the step reward.
- **Penalties:** repeated identical messages (loops), missing JSON, premature finalize with incomplete fields, and dangerous macro choices on the security task.

## Quickstart (local server)

```bash
cd /path/to/this/repo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

Health: `GET http://127.0.0.1:8000/health`  
Schema: `GET http://127.0.0.1:8000/schema`

### Docker

```bash
docker build -t support-triage-openenv .
docker run --rm -p 7860:7860 -e PORT=7860 support-triage-openenv
```

Pre-submission check (example): `curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:7860/reset`

## Deploy to Hugging Face Spaces

**Option A — Link GitHub (no HF git token in terminal):** In the Space **Settings → Repository**, connect `lokendra005/itsm_ticket_triage` branch `main`. Hugging Face rebuilds when you push to GitHub.

**Option B — Push from this clone with a write token** (keep the token in your shell only; never commit it):

```bash
export HF_TOKEN='hf_...'   # https://huggingface.co/settings/tokens
bash scripts/push_hf_space.sh
```

The Space must use the **Docker** SDK and match repo id `lokiii005/ITSM_ticket_triage` (adjust the script if your Space name differs).

## Baseline inference

Requires an OpenAI-compatible HTTP API:

```bash
export OPENAI_API_KEY=sk-...
# optional overrides
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENENV_BASE_URL=http://127.0.0.1:8000
pip install -e ".[inference]"
python inference.py
```

Docker-backed env (matches sample `from_docker_image(IMAGE_NAME)` — **set a tag**):

```bash
export IMAGE_NAME=support-triage-openenv:latest
# or: export OPENENV_IMAGE=...
python inference.py
```

### Baseline scores

Episode score = **sum of step rewards**, clamped to **[0, 1]** (same in `inference.py` and `scripts/oracle_baseline.py`).

| Model | Easy | Medium | Hard | Notes |
| --- | --- | --- | --- | --- |
| Oracle (gold policy, no LLM) | 0.95 | 0.95 | 0.95 | Run `python scripts/oracle_baseline.py` — upper bound |
| _Your LLM (e.g. gpt-4o-mini)_ | — | — | — | Run `python inference.py` with `OPENAI_API_KEY` and a running env |

One-shot local checks (tests, validate, oracle, optional Docker):

```bash
bash scripts/check_local.sh
```

Logs use `[START]`, `[STEP]`, `[END]` lines for automated scoring.

## OpenEnv validation

```bash
pip install openenv-core
openenv validate .   # when CLI available in your OpenEnv install
```

## Project layout

- `support_triage_env/models.py` — Action / Observation / State / `TriageReward`
- `support_triage_env/tasks.py` — scenario specs + gold labels
- `support_triage_env/graders.py` — deterministic scoring helpers
- `support_triage_env/server/support_environment.py` — Gym-style environment
- `support_triage_env/server/app.py` — FastAPI app for Spaces / Docker
- `inference.py` — baseline agent harness
- `openenv.yaml` — Space manifest (`tags: openenv`)

## License

MIT — see OpenEnv competition / your org policy.
