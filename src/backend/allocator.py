# allocator.py
# Cores control center for the EVA
# Refactored to manage a hybrid threading/multiprocessing architecture.

from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, Union

from .config import EvaGlobalConfig
from .console import EvaConsole
from .server import EvaSocketRelay
from .utils import get_logger, is_debug_mode, set_debug_mode

if TYPE_CHECKING:
  from .vision.eyes import EvaEye

# ==============================================================================
# logger CONFIGURATION
# ==============================================================================


logger = get_logger()


# ==============================================================================
# MAIN CLASSES
# ==============================================================================


class EvaAlloc:
  """
  Initializes and allocates all core services for EVA.

  This class acts as the main orchestrator, responsible for starting and
  stopping all subsystems like vision, console, and socket communication
  in a coordinated and graceful manner.
  """

  def __init__(
    self,
    config: EvaGlobalConfig,
    shared_direct_vec: Optional[Any] = None,
    shared_positi_vec: Optional[Any] = None,
  ):
    self.config = config
    logger.info("Initializing EvaCore...")
    set_debug_mode(config.debug)
    logger.info(f"Global debug mode status: {is_debug_mode()}")

    # --- Instantiate Submodules --- #
    self.eva_console = EvaConsole(
      config=config,
      shared_direct_vec=shared_direct_vec,
      shared_positi_vec=shared_positi_vec,
    )

    if config.enable_vision or config.enable_camera:
      from .vision.eyes import EvaEye

      self.eva_eye = EvaEye(
        config=config,
        shared_direct_vec=shared_direct_vec,
        shared_positi_vec=shared_positi_vec,
      )
    else:
      self.eva_eye = None

    self.eva_socket_relay = EvaSocketRelay(
      config=config,
      eva_eye=self.eva_eye,
      eva_console=self.eva_console,
      shared_direct_vec=shared_direct_vec,
      shared_positi_vec=shared_positi_vec,
    )

  def start_services(
    self,
    host: str = "0.0.0.0",
    port: int = 5000,
    cert_path: Optional[Union[PathLike[str], str]] = None,
    key_path: Optional[Union[PathLike[str], str]] = None,
  ):
    """Starts all configured EVA services."""
    logger.info("Starting all EvaCore services...")
    if self.eva_eye:
      self.eva_eye.start()
    if self.eva_socket_relay:
      # Assuming EvaSocketRelay also has a start() method that runs in a thread.
      self.eva_socket_relay.start(
        host=host, port=port, cert_path=cert_path, key_path=key_path
      )
    logger.info("All EvaCore services are running.")

  def stop_services(self):
    """Stops all running EVA services gracefully."""
    logger.info("Stopping all EvaCore services...")
    if self.eva_socket_relay:
      self.eva_socket_relay.stop()
    if self.eva_eye:
      self.eva_eye.stop()
    logger.info("All EvaCore services have been stopped.")

  def get_eva_eye(self) -> Optional[EvaEye]:
    """Provides access to the EvaEye instance."""
    if not self.eva_eye:
      logger.warning_once("Vision/camera is not enabled; EvaEye is not available.")
    return self.eva_eye

  def get_eva_console(self) -> EvaConsole:
    """Provides access to the EvaConsole instance."""
    return self.eva_console

  def get_eva_socket_relay(self) -> EvaSocketRelay:
    """Provides access to the EvaSocketRelay instance."""
    return self.eva_socket_relay


# ==============================================================================
# DEMO: A GUIDE FOR ROBUST EXECUTION
# ==============================================================================
def demo():
  """Demonstrates the proper way to initialize, run, and shut down EvaAlloc."""
  # On Windows/macOS, multiprocessing requires this guard.
  mp.set_start_method("spawn", force=True)

  # Initialize configuration and shared memory
  config = EvaGlobalConfig(debug=True)
  shared_direct_vec = mp.Array("d", [0.0] * 3)
  shared_positi_vec = mp.Array("d", [0.0] * 3)

  # Instantiate the allocator
  allocator = EvaAlloc(config, shared_direct_vec, shared_positi_vec)

  # Set up a graceful shutdown handler
  def shutdown_handler(signum, frame):
    print()  # Newline for cleaner exit
    logger.info("Shutdown signal received. Cleaning up...")
    allocator.stop_services()
    sys.exit(0)

  signal.signal(signal.SIGINT, shutdown_handler)  # Ctrl+C
  signal.signal(signal.SIGTERM, shutdown_handler)  # Kill command

  # Start services and run the main application loop
  try:
    allocator.start_services()
    logger.info("Application is running. Press Ctrl+C to exit.")
    # In a real application, this would be a web server loop or a GUI event
    # loop. Here, we just keep the main thread alive.
    while True:
      # You could fetch and display frames here for debugging
      # if allocator.eva_eye:
      #     frame = allocator.eva_eye.get_latest_robo_frame()
      #     if frame is not None:
      #         cv2.imshow("RoboEye Output", frame)
      #         if cv2.waitKey(1) & 0xFF == 27:
      #             break
      time.sleep(1)

  except Exception as e:
    logger.exception(f"An unexpected error occurred: {e}")
  finally:
    logger.info("Application is shutting down.")
    allocator.stop_services()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
  # This structure is essential for multiprocessing to work correctly
  # across different operating systems.
  demo()
