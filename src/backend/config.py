# config.py

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .model_config import BaseConfig
from .tools._sys import get_sys_camera_index, get_system_memory
from .utils import get_logger

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #


logger = get_logger()


# ---------------------------- #
# --- Global Configuration --- #
# ---------------------------- #


class EvaGlobalConfig:
  def __init__(
    self,
    dev_path: Optional[str] = "/dev/ttyHS1",
    camera_index: Optional[int] = None,
    enable_camera: Optional[bool] = True,
    enable_vision: Optional[bool] = True,
    server_host: str = "0.0.0.0",
    server_port: int = 8000,
    server_ssl_cert_path: Optional[Union[str, os.PathLike[str]]] = None,
    server_ssl_key_path: Optional[Union[str, os.PathLike[str]]] = None,
    vision_model_type: Optional[str] = "transformer",
    vision_model_config: Optional[Union[BaseConfig, Any]] = None,
    vision_pretrained_model_path: Optional[Union[str, os.PathLike[str]]] = None,
    vision_example_input_array_shape: Optional[Union[Tuple[int], List[int]]] = [
      1,
      3,
      224,
      224,
    ],
    robo_frame_capture_freq: int = 12,
    enable_robo_eye_preprocessing: Optional[bool] = False,
    config_path: Optional[Union[str, Path]] = None,
    processes: Optional[int] = None,
    secure_mode: Optional[bool] = True,
    debug: bool = False,
    debug_video_stream: bool = False,
    enable_port: Optional[bool] = False,
    _tracemalloc: Optional[bool] = None,
    _mpq_max_size: Optional[int] = 2,
    _flash_interval: float = 0.3,
    _flasher_gpio_addr: Optional[str] = "492",
    _display_img_size: Tuple[int, int] = (640, 480),
    _action_time_interval: Optional[float] = 0.16,
    _roboarm_init_thetas: Optional[List[float]] = [0.0] * 6,
    _rotplat_init_theta: Optional[float] = 0.0,
    _cons_update_sleep_time: float = 0.02,
    _real_frame_sleep_time: float = 0.08,
    _robo_frame_sleep_time: Optional[float] = 0.12,
    _enable_frame_extraction: Optional[bool] = False,
    _use_ctrl_rot_fallback: Optional[bool] = False,
    _shrink: Optional[bool] = False,
    dx__: Optional[Any] = (64, 196),
    dy__: Optional[Any] = (64, 196),
    dist_threshold__: Optional[float] = 0.01,
    # 是否進入示教/腳本模式；默認關閉以走真實 LLM 對話
    game__: Optional[bool] = False,
    god__: Optional[bool] = True,
  ):
    self.dev_path = dev_path
    if not isinstance(self.dev_path, str):
      raise ValueError(
        f"EvaGlobalConfig: provided device path is supposed to be string, but got ({type(self.dev_path)})."
      )

    if camera_index is None and enable_camera:
      # release all using cameras aggressively
      # FIXME: the releasing logic is somehow bugged, and is not scheduled to fix, whatever
      logger.debug("--- Releasing Cameras ---")
      # _sys_release_reports = release_sys_all_using_cameras(aggressive=True)
      _sys_release_reports: Dict[str, Any] = {}

      if _sys_release_reports:
        logger.debug(" -- Summary --")
        logger.debug(_sys_release_reports["summary"])

        if _sys_release_reports["actions"]:
          logger.debug(" -- Actions --")
          for _idx, _action in enumerate(_sys_release_reports["actions"]):
            logger.debug(f"- Actions[{_idx}]")
            logger.debug(f"-- Device: {_action['device']}")
            logger.debug(f"-- PIDs Found: {_action['pids']}")
            logger.debug(f"-- Killed: {_action['killed']}")
            logger.debug("-")

      # automatically search for the available cameras
      _cam_index = get_sys_camera_index()
      if _cam_index is None:
        logger.warning("Cannot find available camera, set to NoneType by default.")
        self.camera_index = None
      else:
        logger.info(
          f"`camera_index` is not provided, automatically assign camera index with index ({_cam_index})."
        )
        self.camera_index = _cam_index
    else:
      self.camera_index = camera_index

    self.enable_camera = enable_camera
    self.enable_vision = enable_vision

    # --- Server Relates --- #
    self.server_host = server_host
    self.server_port = server_port
    self.server_ssl_cert_path = server_ssl_cert_path
    self.server_ssl_key_path = server_ssl_key_path

    # --- Vision Model Relates --- #
    if vision_model_type == "transformer" and vision_model_config is None:
      logger.warning(
        "Eva System cannot process transformer model without model configuration, please re-check your config."
      )
      warnings.warn(
        "Eva System cannot process transformer model without model configuration, please re-check your config."
      )

    self.vision_model_type = vision_model_type
    self.vision_model_config = vision_model_config
    self.vision_pretrained_model_path = vision_pretrained_model_path
    self.vision_example_input_array_shape = vision_example_input_array_shape

    self.robo_frame_capture_freq = robo_frame_capture_freq
    self.enable_robo_eye_preprocessing = enable_robo_eye_preprocessing
    self.config_path = config_path
    self.processes = processes
    self.secure_mode = secure_mode
    self.debug = debug
    self.debug_video_stream = debug_video_stream
    self.enable_port = enable_port

    # ------------------------ #
    # --- Inner Attributes --- #
    # ------------------------ #
    # MUTABLE, BUT MODIFY WITH CAUTION

    self._tracemalloc = _tracemalloc if _tracemalloc is not None else False
    self._mpq_max_size = _mpq_max_size if _mpq_max_size is not None else 2
    if get_system_memory() > 16.0:  # GB
      self._mpq_max_size = 4
    self._flash_interval = _flash_interval
    self._flasher_gpio_addr = _flasher_gpio_addr
    self._display_img_size = _display_img_size

    self._action_time_interval = _action_time_interval
    self._roboarm_init_thetas = _roboarm_init_thetas
    self._rotplat_init_theta = _rotplat_init_theta

    self._cons_update_sleep_time = _cons_update_sleep_time
    self._real_frame_sleep_time = _real_frame_sleep_time
    self._robo_frame_sleep_time = _robo_frame_sleep_time
    self._enable_frame_extraction = _enable_frame_extraction

    self._use_ctrl_rot_fallback = _use_ctrl_rot_fallback

    # ---------------------------- #
    # --- Protected Attributes --- #
    # ---------------------------- #
    # IMMUTABLE, DO NOT MODIFY

    self.dx__ = dx__
    self.dy__ = dy__
    self.dist_threshold__ = dist_threshold__

    self.god__ = god__
    self.game__ = game__

    # --------------------- #
    # --- Post Initiate --- #
    # --------------------- #

    self._shrink = _shrink if _shrink is not None else False
    if self._shrink:
      logger.info("Shrink mode enabled for low-performance devices.")
      self._real_frame_sleep_time = 0.25  # Slower camera frame processing
      self._robo_frame_sleep_time = 0.35  # Slower vision model processing
      self._cons_update_sleep_time = 0.05  # Slower console updates
      self._display_img_size = (480, 360)  # Smaller image resolution
      self.robo_frame_capture_freq = 24  # Process vision less frequently
      self._mpq_max_size = 1  # Smaller queues to save memory


# config.py ends here
