# _config.py
# Caching self-customized configurations

from dataclasses import dataclass, field


@dataclass
class EvaWorldConfig:
  debug: bool = field(default=False)


eva_world_config = EvaWorldConfig(debug=False)


# ==============================================================================
# I/O SYSTEM FOR EVA WORLD CONFIG
# ==============================================================================


def set_debug_mode(debug: bool):
  eva_world_config.debug = debug


def is_debug_mode() -> bool:
  return eva_world_config.debug


# _config.py ends here
