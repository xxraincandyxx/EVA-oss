# schedule_manager.py

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flask_socketio import SocketIO

from .config import EvaGlobalConfig
from .console import EvaConsole
from .tools.functional import purify_action_data
from .tools.stopwatch import Stopwatch
from .utils import get_logger
from .utils.enums import SCHEDULE_ACTIONS_SET

if TYPE_CHECKING:
  from .web.socket_events import EvaSocketRelay

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #


logger = get_logger()


# ----------------------- #
# --- ScheduleManager --- #
# ----------------------- #


class EvaScheduleManager:
  """Manages the command schedule, including its state and execution."""

  def __init__(
    self,
    eva_console: "EvaConsole",
    config: "EvaGlobalConfig",
    socketio: "SocketIO",
    eva_socket_relay: Optional["EvaSocketRelay"] = None,
  ):
    self.eva_console = eva_console
    self.config = config
    self.socketio = socketio
    self.actions: List[Dict[str, Any]] = []
    self.eva_socket_relay = eva_socket_relay

  def __len__(self):
    return len(self.get_actions())

  def set_schedule(self, new_schedule: List[Dict[str, str]]):
    """
    Sets a new schedule for the EvaScheduleManager.

    Args:
      new_schedule (List[Dict]): A list of dictionaries, each representing a scheduled action.

    Raises:
      TypeError: If new_schedule is not a list.

    Returns:
      None
    """
    if isinstance(new_schedule, List):
      self.actions = [purify_action_data(action) for action in new_schedule]
      logger.info(f"Schedule set with {len(self.actions)} actions.")
    else:
      logger.warning("Attempted to set schedule with invalid data type.")

  def add(
    self,
    cartesian: Optional[List[float]] = None,
    axes: Optional[List[float]] = None,
    action: Optional[str] = None,
  ):
    logger.debug(f"EvaScheduleManager.add() - getting action ({action}).")

    idx = len(self.actions)
    base_action: Dict[str, str] = {
      "id": f"{idx}",
      "action": f"{action}",
      "name": f"Step {idx + 1}",
      "X": "null",
      "Y": "null",
      "Z": "null",
      "A": "null",
      "B": "null",
      "C": "null",
      "Axis1": "null",
      "Axis2": "null",
      "Axis3": "null",
      "Axis4": "null",
      "Axis5": "null",
      "Axis6": "null",
      "duration": "null",
      "rotation": "null",
    }

    if action == "CARTESIAN" and cartesian:
      base_action.update(
        {
          "name": f"Step {idx + 1} - Cartesian",
          "X": f"{cartesian[0]:.4f}",
          "Y": f"{cartesian[1]:.4f}",
          "Z": f"{cartesian[2]:.4f}",
          "A": f"{cartesian[3]:.4f}",
          "B": f"{cartesian[4]:.4f}",
          "C": f"{cartesian[5]:.4f}",
          "duration": f"{cartesian[6]:.2f}",
          "rotation": f"{cartesian[7]:.2f}",
        }
      )
    elif action == "AXES" and axes and cartesian:
      base_action.update(
        {
          "name": f"Step {idx + 1} - Axes",
          "Axis1": f"{axes[0]:.2f}",
          "Axis2": f"{axes[1]:.2f}",
          "Axis3": f"{axes[2]:.2f}",
          "Axis4": f"{axes[3]:.2f}",
          "Axis5": f"{axes[4]:.2f}",
          "Axis6": f"{axes[5]:.2f}",
          "duration": f"{cartesian[6]:.2f}",
          "rotation": f"{cartesian[7]:.2f}",
        }
      )
    elif action and action in SCHEDULE_ACTIONS_SET:
      action_map = {
        "PUMP_ATTACH": "Pump Attach",
        "PUMP_DETACH": "Pump Detach",
        "PUMP_SHUTDOWN": "Pump Shutdown",
        "ROT_CLAMP": "Rot Clamp",
        "ROT_RELEASE": "Rot Release",
        "ROT_ROTATE": "Rot Rotate",
        "SUSPEND": "Suspend",
      }
      base_action["name"] = f"Step {idx + 1} - {action_map.get(action, action)}"
    else:
      logger.error(f"Invalid action call to _add_to_schedule: {action}")
      return

    self.actions.append(base_action)

  def pop(self):
    if self.actions:
      return self.actions.pop()
    return None

  def clear(self):
    self.actions = []

  def get_actions(self) -> List[Dict[str, Dict[str, str]]]:
    return self.actions

  def emit_all(self):
    """Executes all actions in the schedule. This is a blocking call."""
    if not self.eva_console:
      logger.error("Cannot emit schedule, EvaConsole is not available.")
      return

    stopwatch = Stopwatch()
    actions_to_emit = [purify_action_data(action) for action in self.actions]

    for idx, action in enumerate(actions_to_emit):
      action_name = action.get("action")
      if not isinstance(action_name, str):
        logger.error(f"Invalid action in schedule: {action_name}")
        continue
      stopwatch.start(session_name=action_name)

      if action_name not in SCHEDULE_ACTIONS_SET:
        logger.error(f"Invalid action in schedule: {action_name}")
        continue

      logger.debug(f"Emit Schedule gets action:\n{action}")

      if action_name == "CARTESIAN":
        self.eva_console.ctrl_with_cartesian(
          orientation=[
            action["X"],
            action["Y"],
            action["Z"],
            action["A"],
            action["B"],
            action["C"],
          ],
          duration=action["duration"],
          rotation=action["rotation"],
        )
        time.sleep(action["duration"] + self.config._action_time_interval)
      elif action_name == "AXES":
        self.eva_console.ctrl_with_thetas(
          thetas=[
            action["Axis1"],
            action["Axis2"],
            action["Axis3"],
            action["Axis4"],
            action["Axis5"],
            action["Axis6"],
          ],
          duration=action["duration"],
          rotation=action["rotation"],
        )
        time.sleep(action["duration"] + self.config._action_time_interval)
      elif action_name.startswith("ROT_"):
        self.eva_console.ctrl_rot(action_name.split("_")[1])
        time.sleep(0.8)
      elif action_name.startswith("PUMP_"):
        pump_cmd = action_name.split("_")[1]
        self.eva_console.ctrl_pump(pump_cmd)
        time.sleep({"ATTACH": 1.8, "DETACH": 0.8, "SHUTDOWN": 0.8}.get(pump_cmd, 0.8))
      elif action_name == "SUSPEND":
        time.sleep(0.5)
      elif action_name == "CAPTURE":
        logger.debug(
          f"EvaScheduleManger.emit_all() - 'CAPTURE' triggered with index ({idx})."
        )
        if self.eva_socket_relay is None:
          logger.warning(
            "EvaScheduleManager.emit_all() - EvaSocketRelay is not init, cannot"
            " execute action 'CAPTURE', skipped."
          )
          time.sleep(0.5)
          continue

        self.eva_socket_relay._detect(desc=action.get("name"))
        time.sleep(0.5)
      else:
        logger.warning(
          f"EvaScheduleManager.emit_all() - gets unbuilt action {action_name}."
        )

      stopwatch.stop()
    stopwatch.view_results(view_func=logger.debug)
