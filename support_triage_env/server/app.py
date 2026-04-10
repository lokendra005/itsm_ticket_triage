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


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
