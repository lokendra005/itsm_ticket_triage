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

Helpdesk / ITSM triage simulation: an agent reads customer tickets, sets priority and department, picks macro templates, and handles VIP security escalations. Three tasks (easy → medium → hard) with deterministic graders scoring in (0, 1).

## Why this task exists

Support teams route thousands of tickets daily under SLAs. Automating priority, queue, macros, and escalation safely is a real production problem. This environment evaluates agents on structured triage decisions with partial credit and penalties for unsafe shortcuts.

## Action space

- **Type:** `SupportAction`
- **Field:** `message: str` — must contain a JSON object with triage keys described in the observation's `instruction`.

## Observation space

- **Type:** `SupportObservation`
- **Fields:** `ticket_id`, `ticket_body`, `instruction`, `task_id`, `difficulty`, `step_number`, `max_steps`, `feedback`, optional `macro_menu`, optional `reward_hint`.
- **Signals:** `reward` (float), `done` (bool) — inherited from OpenEnv `Observation`.

## State

- **Type:** `SupportState` — `episode_id`, `step_count`, `task_id`, `last_parse_ok`, `loop_warning`, `terminal_grader_score`, `cumulative_reward`.

## Tasks

| Task id | Difficulty | Graded fields | Objective |
| --- | --- | --- | --- |
| `ticket_routing_basic` | Easy | priority, department | Route an invoice mismatch ticket. |
| `macro_selection` | Medium | priority, department, macro_id | Pick the right macro for a permissions escalation. |
| `sla_vip_security` | Hard | priority, department, macro_id, action, notify_manager | VIP + breach indicators — avoid unsafe self-serve; escalate to SOC. |

Graders return the mean of per-field matches, clamped to (0.01, 0.99). The hard task adds a safety penalty for dangerous macro choices.

## Reward design

- **Shaping:** partial credit each step for correct fields (`partial_step_credit`).
- **Terminal:** `finalize=true` with valid required keys triggers `grade_submission` (≈ 0.85× terminal score added to step reward).
- **Penalties:** repeated messages (−0.12), missing JSON (−0.05), premature finalize (−0.06), unsafe macro on security task (−0.35).

## Setup

```bash
pip install -e ".[dev]"
```

### Run locally

```bash
uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 8000
# Health: GET http://127.0.0.1:8000/health
```

### Docker

```bash
docker build -t support-triage-openenv .
docker run --rm -p 8000:8000 -e PORT=8000 support-triage-openenv
```

### Baseline inference

```bash
export HF_TOKEN=...           # or API_KEY
export API_BASE_URL=...       # LLM endpoint
export MODEL_NAME=gpt-4o-mini
export IMAGE_NAME=support-triage-openenv:latest
python inference.py
```

Logs emit `[START]`, `[STEP]`, `[END]` lines for automated scoring.

### Baseline scores

| Model | Easy | Medium | Hard |
| --- | --- | --- | --- |
| Oracle (gold answers) | 0.95 | 0.95 | 0.95 |

### Tests & validation

```bash
pytest tests/ -v
openenv validate .
```

## Project layout

| Path | Purpose |
| --- | --- |
| `support_triage_env/models.py` | Action / Observation / State / TriageReward |
| `support_triage_env/tasks.py` | 3 scenario specs + gold labels |
| `support_triage_env/graders.py` | Deterministic scoring |
| `support_triage_env/server/support_environment.py` | Environment (reset/step/state) |
| `support_triage_env/server/app.py` | FastAPI app |
| `inference.py` | Baseline agent |
| `openenv.yaml` | OpenEnv manifest |

## License

MIT
