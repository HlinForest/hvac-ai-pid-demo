from __future__ import annotations

"""Minimal .env loader so API keys never need to be exported or committed.

Supported format: `KEY=VALUE` lines (optional surrounding quotes, `#` comments)
in `.env` at the project root.  No third-party dependency is introduced, and
real environment variables always take precedence over file entries.
"""

import os
from pathlib import Path


def load_project_env(project_root: str | Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from `<root>/.env`; existing env vars win. Returns what was loaded."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
