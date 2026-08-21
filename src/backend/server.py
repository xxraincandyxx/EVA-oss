# server.py

from __future__ import annotations

import os
import secrets
import threading
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, Union

from flask import Flask
from flask_socketio import SocketIO

from .config import EvaGlobalConfig
from .console import EvaConsole
from .database.manager import EvaDBManager
from .schedule_manager import EvaScheduleManager
from .utils import get_logger
from .web.socket_events import register_socket_events

# --- Modular Web Components --- #
from .web.web_routes import create_web_routes_blueprint

if TYPE_CHECKING:
  from .vision.eyes import EvaEye


# ==============================================================================
# logger CONFIGURATION
# ==============================================================================


logger = get_logger()


# ==============================================================================
# MAIN WEB SERVER CLASS
# ==============================================================================
class EvaSocketRelay:
  """
  Bridges Eva's core components (Console, Eye) with the Web UI.

  This class orchestrates the Flask web server and Socket.IO communication,
  running them in a dedicated thread to integrate with the application's
  main lifecycle management.
  """

  def __init__(
    self,
    config: EvaGlobalConfig,
    eva_eye: Optional[EvaEye],
    eva_console: EvaConsole,
    shared_direct_vec: Optional[Any] = None,
    shared_positi_vec: Optional[Any] = None,
  ):
    # --- Core Application Setup ---
    self.app = Flask(__name__, static_folder=None)
    self.app.config["SECRET_KEY"] = os.getenv("EVA_SECRET_KEY", secrets.token_hex(32))
    self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
    logger.info("Flask & SocketIO initialized.")

    # --- Core Component References ---
    self.config = config
    self.eva_eye = eva_eye
    self.eva_console = eva_console
    self.shared_direct_vec = shared_direct_vec
    self.shared_positi_vec = shared_positi_vec
    self.debug = config.debug

    # --- Sub-module Instantiation ---
    self.eva_db: Optional[EvaDBManager] = None
    self.schedule_manager: Optional[EvaScheduleManager] = None
    self._server_thread: Optional[threading.Thread] = None

    self._init_submodules()

    # --- Register Web Routes and Socket.IO Events ---
    web_blueprint = create_web_routes_blueprint(self)
    self.app.register_blueprint(web_blueprint)
    register_socket_events(self)

    logger.info("EvaSocketRelay initialized successfully.")

  def _init_submodules(self):
    """Initializes database and schedule manager."""
    try:
      self.eva_db = EvaDBManager()
      logger.info("Database manager initialized.")
    except Exception as e:
      logger.error(f"Failed to initialize database manager: {e}")

    if self.eva_console:
      self.schedule_manager = EvaScheduleManager(
        eva_console=self.eva_console,
        config=self.config,
        socketio=self.socketio,
        eva_socket_relay=self,
      )
      logger.info("Schedule manager initialized.")
    else:
      logger.warning(
        "EvaConsole is not available. Schedule manager will be non-functional."
      )

  def start(
    self,
    host: str = "0.0.0.0",
    port: int = 5000,
    cert_path: Optional[Union[PathLike[str], str]] = None,
    key_path: Optional[Union[PathLike[str], str]] = None,
  ):
    """
    Starts the Flask-SocketIO server.
    Explicitly handles HTTP vs HTTPS logic.
    """
    if self._server_thread is not None and self._server_thread.is_alive():
      logger.warning("Server is already running.")
      return

    # use_reloader must be False for a programmatically controlled server.
    # It forks a new process, which breaks our threading/lifecycle model.
    _use_reloader = False

    # Ensure ssl_context is strictly None if paths are missing
    if cert_path and key_path:
      # SSL Context enabled
      _ssl_context = (cert_path, key_path)
      protocol = "https"
    else:
      # Pure HTTP
      _ssl_context = None
      protocol = "http"

    def server_task():
      return self.socketio.run(  # type: ignore
        self.app,
        host=host,
        port=port,
        ssl_context=_ssl_context,
        use_reloader=_use_reloader,
        allow_unsafe_werkzeug=True,
        debug=self.debug,
      )

    self._server_thread = threading.Thread(
      target=server_task, name="EvaWebServer", daemon=True
    )
    logger.info(f"Starting Eva web server on {protocol}://{host}:{port}")
    self._server_thread.start()

  def stop(self):
    """Release resources before the process exits."""
    logger.info("Shutting down web server...")
    self._close_resources()
    self._server_thread = None
    logger.info("Web server has been stopped.")

  def _close_resources(self):
    """Gracefully closes resources like database connections."""
    if self.eva_db:
      self.eva_db.close_thread_connection()
    logger.info("EvaSocketRelay resources closed.")

  def _detect(self, desc: Optional[str] = None):
    """
    Triggers a detection cycle by capturing a frame and sending it to RoboEye.
    """
    if not self.eva_eye or not self.eva_console:
      logger.warning("Cannot perform detection: EvaEye or EvaConsole is not available.")
      return

    self.eva_console.flash(duration=self.config._flash_interval)
    frame = self.eva_eye.get_latest_real_frame()
    if frame is None:
      logger.warning("Cannot get a frame for detection.")
      return

    frame_info = self.eva_console.get_frame_info()
    frame_info.desc = desc
    self.eva_eye.robo_eye._put_to_queue(
      self.eva_eye._robo_queue_in, (frame, frame_info)
    )

  # --- Setters and Status Checks (Unchanged) ---
  def set_eva_console(self, eva_console: "EvaConsole"):
    self.eva_console = eva_console
    self._init_submodules()
    logger.info("EvaConsole has been set. Submodules re-initialized.")

  def set_eva_eye(self, eva_eye: "EvaEye"):
    self.eva_eye = eva_eye
    logger.info("EvaEye has been set.")

  def is_online(self) -> bool:
    return self.eva_console.is_online() if self.eva_console else False
