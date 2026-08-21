# console.py

import math
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import EvaGlobalConfig
from .dev.flasher import EvaFlasher
from .utils import get_logger
from .utils.enums import PUMP_COMMAND, ROT_COMMAND
from .utils.structs import FrameInfo

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================


if "_dynamo_registered" not in sys.modules:
  from .lib import dynamo_

  sys.modules["_dynamo_registered"] = True


# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================


logger = get_logger()


# ==============================================================================
# EVA CONSOLE CLASS
# ==============================================================================


class EvaConsole:
  def __init__(
    self,
    config: EvaGlobalConfig,
    tar_pos: Optional[Union[List[float], Tuple[float]]] = None,
    time_spent: float = 3.6,
    time_sleep_interval: float = 0.36,
    shared_direct_vec: Optional[Any] = None,
    shared_positi_vec: Optional[Any] = None,
    verbose: bool = False,
    debug: bool = False,
  ):
    self.config = config
    self.RAD_TO_DEG = 180.0 / math.pi
    self.DEG_TO_RAD = math.pi / 180.0
    self.eva_flasher = (
      EvaFlasher(gpio_addr=self.config._flasher_gpio_addr)
      if self.config.enable_port and self.config._flasher_gpio_addr
      else None
    )

    _roboarm_init_thetas: Optional[Union["dynamo_.Thetas", List[float]]] = (
      config._roboarm_init_thetas
    )
    _rotplat_init_theta: Optional[float] = config._rotplat_init_theta

    _C_init_thetas = None
    if not isinstance(_roboarm_init_thetas, List):
      _C_init_thetas = _roboarm_init_thetas
      self.roboarm_init_thetas = []
      for i in range(6):
        self.roboarm_init_thetas.append(float(_roboarm_init_thetas[i]))
    else:
      _C_init_thetas = dynamo_.Thetas(
        _roboarm_init_thetas[0],
        _roboarm_init_thetas[1],
        _roboarm_init_thetas[2],
        _roboarm_init_thetas[3],
        _roboarm_init_thetas[4],
        _roboarm_init_thetas[5],
      )
      self.roboarm_init_thetas = _roboarm_init_thetas

    self.rotplat_init_theta = _rotplat_init_theta

    if tar_pos is None:
      if debug:
        tar_pos = (-0.28, 0.0, 0.28)
      else:
        tar_pos = (0.28, 0.0, 0.28)
    self.tar_pos = tar_pos

    self.time_spent = time_spent
    self.time_sleep_interval = time_sleep_interval
    self.utime_spent = int(time_spent * 1e6)
    self.utime_sleep_interval = int(time_sleep_interval)

    self.verbose = verbose
    self.debug = debug

    dynamo_.init_logging("")  # initialize logger for dynamo_
    self._kinematics = dynamo_.Kinematics(enable_cache=debug)
    self._armctrl = dynamo_.ArmCtrl(
      path=self.config.dev_path, debug=debug, enable_port=config.enable_port
    )
    self._rotctrl = dynamo_.RotCtrl(
      *tar_pos,
      path=self.config.dev_path,
      debug=debug,
      enable_port=config.enable_port,
    )
    self._pumpctrl = dynamo_.PumpCtrl(
      path=self.config.dev_path, debug=debug, enable_port=config.enable_port
    )
    self._instancestreamer = dynamo_.InstanceStreamer(
      kinematics=self._kinematics,
      armctrl=self._armctrl,
      rotctrl=self._rotctrl,
      pumpctrl=self._pumpctrl,
      init_thetas=_C_init_thetas,
      init_theta=_rotplat_init_theta,
      verbose=verbose,
      debug=debug,
    )

    self.current_end_position = [0.0, 0.0, 0.0]
    self.current_end_pose = [0.0, 0.0, 0.0]
    self.current_states: List[List[float]] = []

    self.states_lst = []

    self.online_status = True if not debug else False

    self._shared_direct_vec = shared_direct_vec
    self._shared_positi_vec = shared_positi_vec
    self._init_threads()

  def _init_threads(self):
    threading.Thread(
      target=self.send_dir_pos_updates,
      args=(
        self._shared_direct_vec,
        self._shared_positi_vec,
        self._instancestreamer,
      ),
      daemon=True,
    ).start()
    threading.Thread(target=self._orientation_updater_thread, daemon=True).start()

  # ----------------------------------------------------------------------------
  # STATUS METHODS
  # ----------------------------------------------------------------------------

  def is_online(self) -> bool:
    return self.online_status

  def restore(self):
    self._instancestreamer.restore()

  def emit_workflow(self):
    warnings.warn(
      "EvaConsole.emit_workflow() - The 'emit_workflow' method is deprecated.",
      DeprecationWarning,
      2,
    )

    logger.info("EvaConsole.emit_workflow() - Starting EvaConsole Workflow.")
    if not self.debug:
      logger.info("EvaConsole.emit_workflow() - Entering EvaCore.")

      from backend.allocator import EvaCore

      self.eva_core = EvaCore()
      self.eva_core.emit_workflow()

    else:
      logger.info(
        "EvaConsole.emit_workflow()"
        " - DEBUG Mode is activated, entering EvaCore falls back."
      )

  # ----------------------------------------------------------------------------
  # BACKGROUND THREADING METHODS
  # ----------------------------------------------------------------------------

  def _orientation_updater_thread(self):
    _update_interval = 0.5

    while True:
      _orientation = self._instancestreamer.get_roboarm_orientation()
      self.current_end_position = _orientation[:3]
      self.current_end_pose = _orientation[3:]
      time.sleep(_update_interval)

  def send_dir_pos_updates(
    self,
    shared_direct_vec,
    shared_positi_vec,
    __instancestreamer: "dynamo_.InstanceStreamer",
  ):
    while True:
      _direct_vec = __instancestreamer.get_direction_vector()
      _positi_vec = self.get_current_end_position()

      for i in range(len(_direct_vec)):
        shared_direct_vec[i] = _direct_vec[i]
        shared_positi_vec[i] = _positi_vec[i]

      time.sleep(self.config._cons_update_sleep_time)

  # ----------------------------------------------------------------------------
  # FORWARD & INVERSE KINEMATICS METHODS
  # ----------------------------------------------------------------------------

  def forward_kinematics(self, thetas: List[float]) -> List[float]:
    orientation = self._instancestreamer.forward_kinematics(thetas)
    return orientation

  def inverse_kinematics(
    self, x: float, y: float, z: float, a: float, b: float, c: float
  ) -> List[float]:
    thetas = self._instancestreamer.inverse_kinematics([x, y, z, a, b, c])
    return thetas

  # ----------------------------------------------------------------------------
  # CONTROL METHODS
  # ----------------------------------------------------------------------------

  # == VISION ==
  def _flash_on(self, duration: float):
    if not self.eva_flasher:
      logger.warning("EvaConsole._flash_on() - eva_flasher gets NoneType.")
      return

    self.eva_flasher.flash_on()

    if duration < 0.0:
      return

    time.sleep(duration)
    self._flash_off()

  def _flash_off(self):
    if not self.eva_flasher:
      logger.warning("EvaConsole._flash_off() - eva_flasher gets NoneType.")
      return

    self.eva_flasher.flash_off()

  def flash(self, duration: float = 0.3):
    _flash_thread = threading.Thread(target=self._flash_on, args=(duration,))
    _flash_thread.start()

  # == ROBOARM & TABLE ==
  def ctrl_with_thetas(self, thetas: List[float], duration: float, rotation: float):
    u_duration = int(duration * 1e6)
    self._instancestreamer.dual_derive(
      input_orientation=[],
      input_thetas=thetas,
      u_duration=u_duration,
      return_states=False,
    )

  def ctrl_with_cartesian(
    self, orientation: List[float], duration: float, rotation: float
  ):
    u_duration = int(duration * 1e6)
    self._instancestreamer.dual_derive(
      input_orientation=orientation,
      input_thetas=[],
      u_duration=u_duration,
      return_states=False,
    )

  def ctrl_rot(self, command: str):
    logger.debug(f"EvaConsole.ctrl_rot() called with passed argument {command}.")

    command = command.upper()
    if command not in ROT_COMMAND.keys():
      logger.error(
        "EvaConsole.ctrl_rot() - invalid command"
        f" '{command}'. Must be one of {list(ROT_COMMAND.keys())}."
      )
      return

    command_idx = ROT_COMMAND[command]
    logger.info(
      f"EvaConsole.ctrl_rot() receives rot command ({command} as index {command_idx})."
    )
    if not self.config._use_ctrl_rot_fallback:
      self._instancestreamer.ctrl_rot(command_idx)
    else:
      self._instancestreamer.ctrl_rot_fallback(command_idx)

  def ctrl_pump(self, command: str):
    logger.debug(f"EvaConsole.ctrl_pump() called with passed argument {command}.")

    command = command.upper()
    if command not in PUMP_COMMAND.keys():
      logger.error(
        f"EvaConsole.ctrl_pump() - invalid command '{command}'."
        f" Must be one of {list(PUMP_COMMAND.keys())}."
      )
      return

    command_idx = PUMP_COMMAND[command]
    logger.info(
      "EvaConsole.ctrl_pump() receives"
      f" pump command ({command} as index {command_idx})."
    )
    self._instancestreamer.ctrl_pump(command_idx)

  # ----------------------------------------------------------------------------
  # FPV (First-Person View) CONTROL METHODS
  # ----------------------------------------------------------------------------

  def fpv_mv(
    self,
    fpv_pos: List[float],
    fpv_ori: List[float],
    last_orientation: Optional[List[float]] = None,
    duration: float = 0.1,
  ):
    if last_orientation is None:
      last_orientation = self.get_current_end_pose()  # in Degree

    _delta_thetas = [
      __theta * self.RAD_TO_DEG
      for __theta in self.inverse_kinematics(*fpv_pos, *last_orientation)
    ]
    for i in range(3):
      _delta_thetas[3 + i] += fpv_ori[i]
    logger.info(
      f"EvaConsole.fpv_mv() - _delta_thetas[{len(_delta_thetas)}]: {_delta_thetas}"
    )
    self.ctrl_with_thetas(thetas=_delta_thetas, duration=duration, rotation=0.0)
    return

  def fpv_mv_(
    self,
    pos_delta: List[float],
    ori_delta_deg: List[float],
    duration: float = 0.1,
  ):
    """
    Moves the robot arm based on relative changes in position and orientation.
    This is the core method for FPV control.

    :param pos_delta: A list of 3 floats [dx, dy, dz] for position change in meters.
    :param ori_delta_deg: A list of 3 floats [da, db, dc] for orientation change in degrees.
    :param duration: The time for the movement in seconds.
    """

    # Get the current absolute position and orientation
    current_pos_m, current_orient_deg = self.get_current_orientation(rad=False)

    # Calculate the new absolute target position and orientation
    target_pos_m = [current_pos_m[i] + pos_delta[i] for i in range(3)]
    target_orient_deg = [current_orient_deg[i] + ori_delta_deg[i] for i in range(3)]

    # Use the standard cartesian controller to move to the new absolute target
    # The rotation parameter is not used in FPV mode.
    self.ctrl_with_cartesian(
      orientation=[*target_pos_m, *target_orient_deg],
      duration=duration,
      rotation=0.0,
    )

  # ----------------------------------------------------------------------------
  # STATE & ORIENTATION GETTER METHODS
  # ----------------------------------------------------------------------------

  def get_init_states(self):
    return self.get_states_with_thetas(self.roboarm_init_thetas)

  def get_states_lst(self):
    return self.states_lst

  def set_states_lst(self, states_lst):
    self.states_lst = states_lst

  def get_rotplat_init_theta(self) -> float:
    return self.rotplat_init_theta

  def get_roboarm_init_thetas(self) -> List[float]:
    return self.roboarm_init_thetas

  def get_current_thetas(self, rad: bool = False) -> List[float]:
    _rad_thetas = self._get_roboarm_thetas()

    if not rad:
      return [__theta * self.RAD_TO_DEG for __theta in _rad_thetas]
    return _rad_thetas

  def get_current_states(self) -> List[List[float]]:
    return self.get_states_with_thetas(self.get_current_thetas())

  def get_current_end_position(self) -> List[float]:
    _orientation = self._instancestreamer.get_roboarm_orientation()
    self.current_end_position = _orientation[:3]
    self.current_end_pose = _orientation[3:]  # in Degree
    return self.current_end_position

  def get_current_end_pose(self) -> List[float]:
    _orientation = self._instancestreamer.get_roboarm_orientation()
    self.current_end_position = _orientation[:3]
    self.current_end_pose = _orientation[3:]
    return self.current_end_pose

  def get_current_orientation(
    self, rad: bool = False
  ) -> Tuple[List[float], List[float]]:
    """
    Returns the current end-effector orientation.
    :param rad: If True, returns orientation angles in radians. Otherwise, degrees.
    :return: A tuple of (position_meters, orientation_angles).
    """

    position = self.get_current_end_position()
    # Note: get_current_end_pose() already returns degrees
    pose_deg = self.get_current_end_pose()

    if rad:
      pose_rad = [angle * self.DEG_TO_RAD for angle in pose_deg]
      return position, pose_rad
    return position, pose_deg

  def get_cartesian_with_thetas(
    self,
    thetas: List[float],
    duration: Optional[float] = None,
    rotation: Optional[float] = None,
  ):
    """
    This method is equivalent to forward_kinematics,
    we build this for better illustration.

    Args:
        thetas (List[float]): List of the axes angles, in unit Degree, typically 6 elements for 6DOF robots.
        duration (Optional[float], optional): Nonsense. Defaults to None.
        rotation (Optional[float], optional): Nonsense. Defaults to None.
    """

    return [*self.forward_kinematics(thetas), duration, rotation]

  def get_thetas_with_cartesian(
    self,
    cartesian: List[float],
    duration: Optional[float] = None,
    rotation: Optional[float] = None,
  ):
    """
    This method is equivalent to inverse_kinematics,
    we build this for better illustration.

    Args:
        cartesian (List[float]): List of the cartesian info, in unit Degree, typically 6 elements.
        duration (Optional[float], optional): Nonsense. Defaults to None.
        rotation (Optional[float], optional): Nonsense. Defaults to None.
    """

    return [*self.inverse_kinematics(*cartesian), duration, rotation]

  def get_states_with_cartesian(
    self, x: float, y: float, z: float, a: float, b: float, c: float, **kwargs
  ):
    """
    Considering that the provided APIs support only the followings:
      1. Convert orientation/cartesian to thetas
      2. Convert thetas to states
    We build this method in term of the above rule.

    Args:
        <x, y, z, a, b, c> (float): Input orientations.
        **kwargs: Might be holding args like duration or rotation,
                  which is not supposed to be useful here in this method.
    """

    thetas = self.inverse_kinematics(x, y, z, a, b, c)
    thetas = [_theta * self.RAD_TO_DEG for _theta in thetas]
    for i, _theta in enumerate(self.get_current_thetas()):
      thetas[i] += _theta

    logger.info(f"EvaConsole.get_states_with_cartesian() - returned thetas: {thetas}")
    states = self._instancestreamer.get_states(input_thetas=thetas, verbose=True)
    states.insert(0, [0.0, 0.0, 0.0])
    return states

  def get_states_with_thetas(
    self, thetas: List[float], **kwargs: Dict[str, Any]
  ) -> List[List[float]]:
    """
    Returns the states via the API InstanceStreamer::getStates()

    Args:
        thetas: Input thetas caching the current thetas of the robotic arm.
        **kwargs: Might be holding args like duration or rotation,
                  which is not supposed to be useful here in this method.
    """

    states = self._instancestreamer.get_states(input_thetas=thetas, verbose=True)
    states.insert(0, [0.0, 0.0, 0.0])
    return states

  def get_frame_info(self) -> FrameInfo:
    direct_vec = self._instancestreamer.get_direction_vector()
    positi_vec = self.get_current_end_position()
    return FrameInfo(
      direct_vec=direct_vec,
      positi_vec=positi_vec,
    )

  def _get_roboarm_thetas(self) -> List[float]:
    return self._instancestreamer.get_roboarm_thetas()

  # ----------------------------------------------------------------------------
  # STATE STACK METHODS
  # ----------------------------------------------------------------------------

  def push_states_lst(self, states: List[List[float]]) -> List[List[List[float]]]:
    return self.states_lst.append(states)

  def pop_states_lst(self) -> List[List[float]]:
    return self.states_lst.pop()

  # ----------------------------------------------------------------------------
  # DEPRECATED METHODS
  # ----------------------------------------------------------------------------

  def proc(self, shared_direct_vec: List[float], shared_positi_vec: List[float]):
    warnings.warn("The 'proc' method is deprecated, ", DeprecationWarning, 2)

    logger.info("<| EvaConsole.proc |>")

    while True:
      frame_info = self.get_frame_info()
      _direct_vec = frame_info.direct_vec
      _positi_vec = frame_info.positi_vec

      if _direct_vec is None or _positi_vec is None:
        warnings.warn(
          "EvaConsole.proc() - _direct_vec or _positi_vec is None",
        )
        return

      for i in range(len(_direct_vec)):
        shared_direct_vec[i] = _direct_vec[i]
        shared_positi_vec[i] = _positi_vec[i]

      time.sleep(self.config._cons_update_sleep_time)

  def get_console_proc_fn(self):
    return self.proc

  def move_to(
    self,
    orientation: "dynamo_.Orientation",
    duration: float = 2.0,
    return_states: bool = False,
    return_angles: bool = False,
    verbose: bool = False,
  ) -> Optional[Tuple[Union[None, List], Union[None, List]]]:
    warnings.warn("The 'move_to' method is deprecated, ", DeprecationWarning, 2)

    utime_spent = int(duration * 1e6)

    return self._instancestreamer.move_to(
      orientation=orientation,
      utime_spent=utime_spent,
      return_states=return_states,
      return_angles=return_angles,
      verbose=verbose,
    )


# console.py ends here
