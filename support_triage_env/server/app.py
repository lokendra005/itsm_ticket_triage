"""FastAPI entrypoint for Hugging Face Spaces and local Docker runs."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from support_triage_env.models import SupportAction, SupportObservation
from support_triage_env.server.support_environment import SupportTriageEnvironment

app = create_app(
    SupportTriageEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_triage_env",
)


@app.get("/")
def _space_root():
    """HF / browsers often probe `/`; OpenEnv lives under /health, /reset, /docs, /ws."""
    return {
        "service": "support_triage_openenv",
        "health": "/health",
        "reset": "POST /reset",
        "docs": "/docs",
        "schema": "/schema",
    }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
