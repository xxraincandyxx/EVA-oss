# eyes.py
# Eva's Sacred Eyes
# Refactored for a hybrid threading/multiprocessing architecture.

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import threading
import time
import warnings
from multiprocessing.synchronize import Event as ProcessEvent
from typing import Any, Callable, Dict, Optional, Union

import cv2  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from ..config import EvaGlobalConfig
from ..tools.functional import xy_loc2xyxy
from ..tools.ultrafuncs import ultra2cv
from ..utils import get_logger, is_debug_mode
from ..utils.structs import FrameInfo
from .tools import draw_boxes_and_resize

# ==============================================================================
# LOCAL VARIABLES
# ==============================================================================


logger = get_logger()
RECONNECT_DELAY_SECONDS = 5.0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def identity(*args: Any, **kwargs: Dict[Any, Any]):
  return args, kwargs


# ==============================================================================
# BASE CLASS
# ==============================================================================


class _BaseWorker:
  """A base class for stoppable workers (threads or processes)."""

  def __init__(self, config: EvaGlobalConfig):
    self.config = config
    self._stop_event: Union[threading.Event, ProcessEvent, None] = None

  def stop(self):
    """Signals the worker to stop."""
    if self._stop_event is not None:
      self._stop_event.set()

  def _is_stopped(self) -> bool:
    """Checks if the stop event has been set."""
    return self._stop_event is not None and self._stop_event.is_set()


# ==============================================================================
# REAL EYE
# ==============================================================================


class RealEye(_BaseWorker):
  """
  Manages camera frame capture in a dedicated, lightweight thread.

  This is an I/O-bound task, making it a perfect candidate for threading.
  It continuously captures frames and places them into queues for consumption
  by other parts of the application.
  """

  def __init__(
    self,
    config: EvaGlobalConfig,
    shared_direct_vec: Any,
    shared_positi_vec: Any,
    real_frame_queue: queue.Queue[Any],
    robo_frame_queue_in: "mp.Queue[Any]",
  ):
    super().__init__(config)
    self._camera_source = config.camera_index
    self._shared_direct_vec = shared_direct_vec
    self._shared_positi_vec = shared_positi_vec
    self._real_frame_queue = real_frame_queue
    self._robo_frame_queue_in = robo_frame_queue_in
    self._stop_event = threading.Event()
    self._cap = None
    logger.info(f"<RealEye Thread> Initialized for source: '{self._camera_source}'.")

  def _connect(self) -> bool:
    """Establishes or re-establishes the camera connection."""
    logger.info(f"Attempting to connect to camera: {self._camera_source}...")
    if self._cap and self._cap.isOpened():
      self._cap.release()

    if isinstance(self._camera_source, str):
      self._camera_source = int(self._camera_source)

    if self._camera_source is None:
      warnings.warn("No camera source specified. Defaulting to camera index 0.")
      self._camera_source = 0

    if self._camera_source < 0 or self._camera_source > 10:
      logger.error(f"Invalid camera index: {self._camera_source}")
      return False

    backend = cv2.CAP_FFMPEG if isinstance(self._camera_source, str) else cv2.CAP_ANY
    self._cap = cv2.VideoCapture(self._camera_source, backend)

    if not self._cap.isOpened():
      logger.error(f"Failed to open camera source: {self._camera_source}")
      return False

    logger.info(f"Successfully connected to camera: {self._camera_source}")
    return True

  def run(self):
    """The main loop for the camera capture thread."""
    logger.info("<RealEye Thread> Capture thread started.")
    self._connect()
    frame_count = 0

    while not self._is_stopped():
      if not self._cap or not self._cap.isOpened():
        logger.warning(
          f"Camera disconnected. Retrying in {RECONNECT_DELAY_SECONDS}s..."
        )
        time.sleep(RECONNECT_DELAY_SECONDS)
        self._connect()
        continue

      ret, frame = self._cap.read()

      if not ret:
        logger.warning("Failed to read frame. Attempting to reconnect.")
        self._connect()
        continue

      # Put raw frame onto the queue for live monitoring (non-blocking).
      self._put_to_queue(self._real_frame_queue, frame)

      # Periodically send a frame copy to the RoboEye process.
      if frame_count % self.config.robo_frame_capture_freq == 0:
        frame_info = FrameInfo(
          direct_vec=list(self._shared_direct_vec),
          positi_vec=list(self._shared_positi_vec),
        )
        self._put_to_queue(self._robo_frame_queue_in, (frame.copy(), frame_info))

      frame_count = (frame_count + 1) % 4096
      time.sleep(self.config._real_frame_sleep_time)

    if self._cap and self._cap.isOpened():
      self._cap.release()
    logger.info("<RealEye Thread> Stopped and camera released.")

  def _put_to_queue(
    self,
    q: Union[queue.Queue[Any], "mp.Queue[Any]"],
    item: Any,
  ):
    """Puts an item into a queue, dropping the oldest if full."""
    if q.full():
      try:
        q.get_nowait()
      except (queue.Empty, EOFError):
        pass
    try:
      q.put_nowait(item)
    except (queue.Full, EOFError):
      logger.warning_once(
        f"Queue for {self.__class__.__name__} is full. Dropping item."
      )


# ==============================================================================
# ROBO EYE
# ==============================================================================


class RoboEye(_BaseWorker):
  """
  Performs vision model inference in an isolated process.

  This is a CPU-bound task. Running it in a separate process prevents the
  Global Interpreter Lock (GIL) from blocking the main application, ensuring
  the UI and network servers remain responsive.
  """

  def __init__(
    self,
    config: EvaGlobalConfig,
    robo_frame_queue_in: mp.Queue[
      Union[
        cv2.typing.MatLike,
        np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
      ]
    ],
    robo_frame_queue_out: mp.Queue[
      Union[
        cv2.typing.MatLike,
        np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
      ]
    ],
  ):
    super().__init__(config)
    self._robo_frame_queue_in = robo_frame_queue_in
    self._robo_frame_queue_out = robo_frame_queue_out
    self._stop_event = mp.Event()
    self.model = None
    self._process_fn: Callable[..., Any] = self._process_frame_default

  def _init_model(self):
    """Initializes the vision model based on the configuration."""
    if is_debug_mode():
      logger.info("Debug mode: RoboEye model is an identity function.")
      self.model = identity
      self._process_fn = self._process_frame_default
      return

    if self.config.game__:
      self._process_fn = self._process_game_mode_frame
    elif self.config.vision_model_type == "yolo":
      self._init_yolo()
      self._process_fn = self._process_yolo_frame
    elif self.config.vision_model_type == "yolo_dpu":
      self._init_yolo_dpu()
      self._process_fn = self._process_yolo_dpu_frame
    # ... other model types like 'transformer'
    else:
      logger.warning(
        "Unknown model type '%s'. Using identity function.",
        self.config.vision_model_type,
      )
      self._process_fn = self._process_frame_default

  def _init_yolo(self):
    """Loads and initializes the YOLO model."""
    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise RuntimeError(
        "YOLO vision requires the optional 'ultralytics' package."
      ) from error

    path = self.config.vision_pretrained_model_path
    if not path or not os.path.exists(path):
      raise FileNotFoundError(f"YOLO model file not found: {path}")
    logger.info(f"Loading YOLO model from: {path}")
    self.model = YOLO(path)

  def _init_yolo_dpu(self):
    """
    初始化 DPU 版 YOLOv3:
    - 使用 DPURunner 载入 .xmodel
    - 依赖 postprocess_yolo 做解码与 NMS
    """
    from .yolo_dpu import DPURunner

    path = self.config.vision_pretrained_model_path
    if path is None or not os.path.exists(path):
      raise FileNotFoundError(
        f"YOLO DPU xmodel file not found: {path}. "
        "请在 EvaGlobalConfig.vision_pretrained_model_path 中配置 .xmodel 路径。"
      )
    logger.info(f"Loading YOLO DPU xmodel from: {path}")
    self._dpu_runner = DPURunner(str(path))

  def run(self):
    """The main loop for the model inference process."""
    logger.info("<RoboEye Process> Inference process started.")
    if self.config.enable_vision:
      self._init_model()
    else:
      logger.warning("Vision is disabled. RoboEye will perform no operations.")
    logger.info("<RoboEye Process> Initialized.")

    while not self._is_stopped():
      try:
        frame, frame_info = self._robo_frame_queue_in.get(timeout=1.0)
        processed_frame = self._process_frame(frame, frame_info)
        self._put_to_queue(self._robo_frame_queue_out, processed_frame)
      except queue.Empty:
        continue  # Timeout occurred, loop again to check stop_event
      except (EOFError, BrokenPipeError):
        logger.error("Input queue pipe broke. Shutting down RoboEye.")
        break
    logger.info("<RoboEye Process> Stopped.")

  def _process_frame(
    self,
    frame: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
    frame_info: FrameInfo,
  ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    """General frame processing pipeline."""
    return self._process_fn(frame, frame_info)

  def _process_yolo_frame(
    self,
    frame: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
    _: FrameInfo,
  ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    """Processes a frame using the YOLO model."""
    results = self.model(frame)
    return ultra2cv(results[0])

  def _process_yolo_dpu_frame(
    self,
    frame: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
    _: FrameInfo,
  ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    """
    使用 DPU 版 YOLOv3 进行目标检测：
    - 通过 DPURunner.run 得到 3 个特征图输出(int8)
    - 使用 postprocess_yolo 解码为 boxes / scores / classes
    - 利用 draw_boxes_and_resize 将检测框画回帧并缩放到显示尺寸
    """
    from .run_yolov3_dpu_imagenet import postprocess_yolo

    h, w = frame.shape[:2]
    outputs = self._dpu_runner.run(frame)

    # Returns boxes, scores, classes
    boxes, _, _ = postprocess_yolo(outputs, self._dpu_runner, (h, w))

    if boxes.shape[0] == 0:
      # Undetected, returns original frame
      return cv2.resize(
        frame, self.config._display_img_size, interpolation=cv2.INTER_LINEAR
      )

    return draw_boxes_and_resize(frame, boxes, self.config._display_img_size)

  def _process_game_mode_frame(
    self,
    frame: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
    _: FrameInfo,
  ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    """Handles special logic for 'game mode'."""
    if random.random() < 0.05:
      box = xy_loc2xyxy(
        (random.random(), random.random()), self.config._display_img_size
      )
      return draw_boxes_and_resize(
        frame, np.array([box]), self.config._display_img_size
      )
    return frame

  def _process_frame_default(
    self,
    frame: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
    _: FrameInfo,
  ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    """Default passthrough processing."""
    return frame

  def _put_to_queue(self, q: "mp.Queue[Any]", item: Any):
    """Identical to RealEye's put method, adapted for multiprocessing."""
    if q.full():
      try:
        q.get_nowait()
      except (queue.Empty, EOFError):
        pass
    try:
      q.put_nowait(item)
    except (queue.Full, EOFError):
      logger.warning_once(
        f"Queue for {self.__class__.__name__} is full. Dropping item."
      )


# ==============================================================================
# EVA EYE
# ==============================================================================


class EvaEye:
  """
  Main coordinator for the vision system.

  Manages the lifecycle of the RealEye (thread) and RoboEye (process),
  providing a clean interface to start, stop, and retrieve frames.
  """

  def __init__(
    self,
    config: EvaGlobalConfig,
    shared_direct_vec: Any,
    shared_positi_vec: Any,
  ):
    self.config = config
    self._real_eye_thread: Optional[threading.Thread] = None
    self._robo_eye_process: Optional[mp.Process] = None

    # --- Queues --- #
    # Live display queue (thread-safe, in-process)
    self._real_queue: queue.Queue[Any] = queue.Queue(maxsize=config._mpq_max_size)
    # Process-safe queues for Thread -> Process and Process -> Main communication
    self._robo_queue_in: queue.Queue[Any] = mp.Queue(maxsize=config._mpq_max_size)
    self._robo_queue_out: queue.Queue[Any] = mp.Queue(maxsize=config._mpq_max_size)

    # --- Caches --- #
    self._last_real_frame = None
    self._last_robo_frame = None

    # --- Instantiate Components --- #
    self.real_eye = RealEye(
      config,
      shared_direct_vec,
      shared_positi_vec,
      self._real_queue,
      self._robo_queue_in,
    )
    self.robo_eye = RoboEye(config, self._robo_queue_in, self._robo_queue_out)

  def start(self):
    """Starts the RealEye thread and RoboEye process."""
    if not self.config.enable_camera and not self.config.enable_vision:
      logger.warning("Both camera and vision are disabled. EvaEye will not start.")
      return

    logger.info("Starting EvaEye services...")
    # Start the camera capture thread
    self._real_eye_thread = threading.Thread(
      target=self.real_eye.run, name="RealEye", daemon=True
    )
    self._real_eye_thread.start()

    # Start the model inference process
    self._robo_eye_process = mp.Process(
      target=self.robo_eye.run, name="RoboEye", daemon=True
    )
    self._robo_eye_process.start()

  def stop(self):
    """Stops the vision thread and process gracefully."""
    logger.info("Stopping EvaEye services...")
    self.real_eye.stop()
    self.robo_eye.stop()

    if self._real_eye_thread and self._real_eye_thread.is_alive():
      self._real_eye_thread.join(timeout=5.0)

    if self._robo_eye_process and self._robo_eye_process.is_alive():
      self._robo_eye_process.join(timeout=5.0)
      if self._robo_eye_process.is_alive():
        logger.warning("RoboEye process did not exit gracefully. Terminating.")
        self._robo_eye_process.terminate()

    # Clean up queues
    for q in [self._robo_queue_in, self._robo_queue_out]:
      q.close()
      q.join_thread()

    logger.info("EvaEye services stopped.")

  def _get_latest_from_queue(
    self, q: Union[queue.Queue[Any], "mp.Queue[Any]"]
  ) -> Optional[Any]:
    """Drains a queue and returns only the most recent item."""
    latest_item = None
    while True:
      try:
        latest_item = q.get_nowait()
      except queue.Empty:
        break
    return latest_item

  def get_latest_real_frame(
    self,
  ) -> Optional[np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]]:
    """Gets the most recent raw camera frame."""
    frame = self._get_latest_from_queue(self._real_queue)
    if frame is not None:
      return frame

  def get_latest_robo_frame(
    self,
  ) -> Optional[np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]]:
    """Gets the most recent processed frame from the model."""
    frame = self._get_latest_from_queue(self._robo_queue_out)
    if frame is not None:
      return frame
