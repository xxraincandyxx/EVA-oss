"""Filesystem locations used by the backend."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]


def _user_dir(env_name: str, xdg_name: str, fallback: Path) -> Path:
  if configured := os.getenv(env_name):
    return Path(configured).expanduser().resolve()
  if xdg_root := os.getenv(xdg_name):
    return Path(xdg_root).expanduser().resolve() / "eva"
  return fallback


DATA_DIR = _user_dir(
  "EVA_DATA_DIR", "XDG_DATA_HOME", Path.home() / ".local" / "share" / "eva"
)
STATE_DIR = _user_dir(
  "EVA_STATE_DIR", "XDG_STATE_HOME", Path.home() / ".local" / "state" / "eva"
)
LOG_DIR = STATE_DIR / "logs"
LOG_CACHE_DIR = LOG_DIR / "cache"
MODEL_DIR = Path(
  os.getenv("EVA_MODEL_DIR", PROJECT_DIR / "src" / "weights")
).expanduser()
