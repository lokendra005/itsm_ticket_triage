"""
OpenEnv / HF compatibility entry: some validators expect `server.app:app`.

The environment implementation lives under `support_triage_env/`.
"""

from __future__ import annotations

import os

from support_triage_env.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
