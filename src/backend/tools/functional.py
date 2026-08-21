# functional.py

import random
from typing import Any, Dict, List, Tuple

# ----------------- #
# --- Functions --- #
# ----------------- #


def dataclass2list(data) -> List[Tuple[Tuple, Tuple]]:
  lst = []

  idx = 0
  while True:
    _pack = getattr(data, f"_SERIES_IDX_0{idx}", None)

    if _pack is None:
      break

    pos, xy_loc = _pack
    lst.append((pos, xy_loc))

    idx += 1

  return lst


def xy_loc2xyxy(xy_loc: Tuple[float, float], img_size: Tuple[float, float], dx, dy):
  dxmin = random.randint(dx[0], dx[1])
  dxmax = random.randint(dx[0], dx[1])
  dymin = random.randint(dy[0], dy[1])
  dymax = random.randint(dy[0], dy[1])
  x_min = max(xy_loc[0] * img_size[0] - dxmin, 0)
  y_min = max(xy_loc[1] * img_size[1] - dxmax, 0)
  x_max = min(xy_loc[0] * img_size[0] + dymin, img_size[0])
  y_max = min(xy_loc[1] * img_size[1] + dymax, img_size[1])
  return x_min, y_min, x_max, y_max


def _convert_string_to_float(value_str: Any):
  if not isinstance(value_str, str):
    return value_str

  if not value_str:
    return None

  value_str = value_str.strip().lower()
  if value_str in ("null", "none", "nan", ""):
    return None

  try:
    return float(value_str)
  except ValueError:
    return None


def purify_action_data(action_data: Dict[str, str]) -> Dict[str, Any]:
  keys_to_convert = [
    "X",
    "Y",
    "Z",
    "A",
    "B",
    "C",
    "Axis1",
    "Axis2",
    "Axis3",
    "Axis4",
    "Axis5",
    "Axis6",
    "duration",
    "rotation",
  ]
  processed_data: Dict[str, Any] = {
    **action_data,
    **{key: _convert_string_to_float(action_data.get(key)) for key in keys_to_convert},
  }
  return processed_data


# functional.py ends here
