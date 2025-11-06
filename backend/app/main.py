"""Compatibility shim exposing the FastAPI application for legacy imports."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from app.models.config import Settings

# Ensure backend root is importable so `server` resolves.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from server import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    settings = Settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
