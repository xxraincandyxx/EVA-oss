# web/web_routes.py

import os
import time
import warnings
from os import PathLike
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np
from flask import Blueprint, Response, send_file, send_from_directory

from ..utils import get_logger

if TYPE_CHECKING:
  from ..server import EvaSocketRelay

logger = get_logger()


def create_web_routes_blueprint(relay: "EvaSocketRelay") -> Blueprint:
  """
  Creates and configures the Flask Blueprint for the React SPA and video
  streams.
  """
  # This path assumes the Flask app is run from the project root directory
  # (the one containing the 'server' and 'client' folders).
  # 專案中前端位於 src/frontend，Vite 打包輸出到 frontend/dist
  react_build_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
  )

  if not os.path.isdir(react_build_folder):
    logger.error(
      f"React build folder not found at: {react_build_folder}. "
      "Please run 'npm run build' in the 'client' directory before starting the"
      " server."
    )

  web_routes = Blueprint(
    name="web_routes",
    import_name=__name__,
    static_folder=react_build_folder,
    static_url_path="/",
  )

  @web_routes.route("/", defaults={"path": ""})
  @web_routes.route("/<path:path>")
  def serve(path: Union[PathLike[str], str]) -> Response:  # pyright: ignore
    """
    Serves the React application.
    - If the path is a file in the static build folder (e.g., /assets/index.js),
      it's served directly.
    - For any other path (e.g., /control, /fpv), it serves the main index.html
      file, allowing the client-side React router (or our state manager) to
      handle the page change.
    """
    base_dir = web_routes.static_folder
    if base_dir is None:
      warnings.warn(
        "No static folder is configured for the web_routes blueprint. "
        "Please set the static_folder attribute before using this blueprint."
      )
      return Response(status=404)
    full_path = os.path.join(base_dir, path)
    index_path = os.path.join(base_dir, "index.html")

    if path != "" and os.path.exists(full_path):
      return send_from_directory(base_dir, path)

    if os.path.exists(index_path):
      return send_file(index_path)

    warnings.warn(f"React index.html not found at {index_path}. Please build frontend.")
    return Response(status=404)

  # --- Video Streaming Logic (Kept from original) ---

  # Cache for real and robo frames
  last_real_frame_cache = np.zeros((480, 640, 3), dtype=np.uint8)
  last_robo_frame_cache = np.zeros((480, 640, 3), dtype=np.uint8)

  # Frame generator iteration counters
  real_frame_counter: int = 0
  robo_frame_counter: int = 0
  counter_log_cycle: int = 1024

  def get_placeholder_frame(
    text: str,
  ) -> np.ndarray[tuple[int, int, int], np.dtype[np.unsignedinteger]]:
    """Generates a standard placeholder image."""
    img = np.full(
      shape=(480, 640, 3), fill_value=60, dtype=np.uint8
    )  # Dark grey background
    cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

  def _generate_frames(source: str):
    """Generator function that yields frames for video streaming."""
    nonlocal last_real_frame_cache, last_robo_frame_cache
    nonlocal real_frame_counter, robo_frame_counter

    if not relay.config.enable_camera:
      placeholder = get_placeholder_frame(text="Camera Disabled")
      while True:
        _, buffer = cv2.imencode(".jpg", placeholder)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(1)

    jpeg_quality = 75 if not relay.config._shrink else 50
    while True:
      frame = get_placeholder_frame("Initial Frame")

      if relay.config.debug_video_stream:
        if source == "real":
          if relay.eva_eye:
            frame = relay.eva_eye.get_latest_real_frame()

          if frame is not None:
            last_real_frame_cache = frame
            if real_frame_counter == 0:
              logger.debug("Latest Real Frame: %s", frame.shape)
          else:
            frame = last_real_frame_cache
            logger.debug("Latest Real Frame (cached): %s", frame.shape)
          real_frame_counter = (real_frame_counter + 1) % counter_log_cycle

        elif source == "robo":
          if relay.eva_eye:
            frame = relay.eva_eye.get_latest_robo_frame()

          if frame is not None:
            last_robo_frame_cache = frame
            if robo_frame_counter == 0:
              logger.debug("Latest Robo Frame: %s", frame.shape)
          else:
            frame = last_robo_frame_cache
            logger.debug("Latest Robo Frame (cached): %s", frame.shape)
          robo_frame_counter = (robo_frame_counter + 1) % counter_log_cycle

      else:
        if source == "real":
          if relay.eva_eye:
            frame = relay.eva_eye.get_latest_real_frame()

          if frame is not None:
            last_real_frame_cache = frame
          else:
            frame = last_real_frame_cache

        elif source == "robo":
          if relay.eva_eye:
            frame = relay.eva_eye.get_latest_robo_frame()

          if frame is not None:
            last_robo_frame_cache = frame
          else:
            frame = last_robo_frame_cache

      if frame.size == 0:
        frame = get_placeholder_frame("No Signal")

      ret, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
      )
      if not ret:
        continue

      frame_bytes = buffer.tobytes()
      yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
      time.sleep(relay.config._real_frame_sleep_time or 0.05)

  @web_routes.route("/realtime_monitor")
  def realtime_monitor():  # pyright: ignore
    return Response(
      _generate_frames(source="real"),
      mimetype="multipart/x-mixed-replace; boundary=frame",
    )

  @web_routes.route("/auxiliary_stacked_detection_monitor")
  def auxiliary_stacked_detection_monitor():  # pyright: ignore
    return Response(
      _generate_frames(source="robo"),
      mimetype="multipart/x-mixed-replace; boundary=frame",
    )

  return web_routes
