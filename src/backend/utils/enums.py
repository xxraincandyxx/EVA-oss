# enums.py

from dataclasses import dataclass
from typing import Dict

ROT_COMMAND: Dict[str, int] = {"CLAMP": 0, "RELEASE": 1, "ROTATE": 2}
PUMP_COMMAND: Dict[str, int] = {"ATTACH": 0, "DETACH": 1, "SHUTDOWN": 2}

SCHEDULE_ROBOARM_ACTIONS = {"CARTESIAN", "AXES"}
SCHEDULE_ROT_ACTIONS = {"ROT_CLAMP", "ROT_RELEASE", "ROT_ROTATE"}
SCHEDULE_PUMP_ACTIONS = {"PUMP_ATTACH", "PUMP_DETACH", "PUMP_SHUTDOWN"}
SCHEDULE_FUNCTIONAL = {"SUSPEND", "CAPTURE"}
SCHEDULE_ACTIONS_SET = (
  SCHEDULE_ROBOARM_ACTIONS
  | SCHEDULE_ROT_ACTIONS
  | SCHEDULE_PUMP_ACTIONS
  | SCHEDULE_FUNCTIONAL
)


@dataclass
class SpaceTest:
  # (x, y, z) ;; (width, height)
  _SERIES_IDX_00 = ((0.2, 0.2, 0.2), (0.5, 0.5))
  _SERIES_IDX_01 = ((0.3, 0.3, 0.3), (0.25, 0.75))


@dataclass
class SpaceX:
  _SERIES_IDX_00 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_01 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_02 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_03 = ((0.0, 0.0, 0.0), (0.0, 0.0))


@dataclass
class SpaceY:
  _SERIES_IDX_00 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_01 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_02 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_03 = ((0.0, 0.0, 0.0), (0.0, 0.0))


@dataclass
class SpaceZ:
  _SERIES_IDX_00 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_01 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_02 = ((0.0, 0.0, 0.0), (0.0, 0.0))
  _SERIES_IDX_03 = ((0.0, 0.0, 0.0), (0.0, 0.0))


# enums.py ends here
