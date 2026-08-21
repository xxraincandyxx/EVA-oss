"""Shared backend utilities."""

from ..paths import LOG_CACHE_DIR, LOG_DIR
from ._config import (
  is_debug_mode,
  set_debug_mode,
)
from ._logging import _EvaLogger, setup_eva_logger
from .utils import is_macos

__all__ = ["get_logger", "is_debug_mode", "is_macos", "set_debug_mode"]

LOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
setup_eva_logger(log_dir=LOG_DIR, cache_dir=LOG_CACHE_DIR, streaming=False)

__eva_logger_ins = _EvaLogger()
get_logger = __eva_logger_ins.get_logger
