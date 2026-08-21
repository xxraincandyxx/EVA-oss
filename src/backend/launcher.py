# launch.py
# Main entry point for the EVA application.

import multiprocessing as mp
import signal
import threading
import tracemalloc

from .allocator import EvaAlloc
from .config import EvaGlobalConfig
from .utils import get_logger

# ==============================================================================
# logger CONFIGURATION
# ==============================================================================


logger = get_logger()


# ==============================================================================
# MAIN APPLICATION CLASS
# ==============================================================================
class EvaLauncher:
  """
  Orchestrates the startup, execution, and graceful shutdown of the entire
  EVA application.
  """

  def __init__(self, config: EvaGlobalConfig):
    """
    Initializes the application components via the EvaAlloc service manager.

    Args:
        config: The global configuration object for the application.
    """
    logger.info("Initializing Eva...")
    self.config = config

    # --- Shared Memory Initialization --- #
    self.shared_direct_vec = mp.Array("d", [0.0, 0.0, 0.0])
    self.shared_positi_vec = mp.Array("d", [0.0, 0.0, 0.0])

    # --- Service Allocator --- #
    self.eva_alloc = EvaAlloc(
      config=self.config,
      shared_direct_vec=self.shared_direct_vec,
      shared_positi_vec=self.shared_positi_vec,
    )
    self._shutdown_event = threading.Event()
    self._stopped = False
    logger.info("Eva initialized successfully.")

  def run(self):
    """
    Starts all services and keeps the main process alive until shutdown.
    """
    if self.config._tracemalloc:
      tracemalloc.start()

    signal.signal(signal.SIGINT, self._shutdown_handler)
    signal.signal(signal.SIGTERM, self._shutdown_handler)

    try:
      logger.info("Starting background services...")
      self.eva_alloc.start_services(
        host=self.config.server_host,
        port=self.config.server_port,
        cert_path=self.config.server_ssl_cert_path,
        key_path=self.config.server_ssl_key_path,
      )

      # The main thread now simply waits for a shutdown signal. All services
      # (camera, vision, web server) run in background threads or processes.
      logger.info("All services are running. Press Ctrl+C to exit.")
      self._shutdown_event.wait()

    except Exception as e:
      logger.exception(f"A critical error occurred in the main application: {e}")
    finally:
      self.stop()

  def stop(self):
    """Initiates a graceful shutdown of all application services."""
    if self._stopped:
      return

    logger.info("Commencing graceful shutdown...")
    self._stopped = True
    self._shutdown_event.set()

    self.eva_alloc.stop_services()

    if self.config._tracemalloc and tracemalloc.is_tracing():
      self._log_tracemalloc_stats()
      tracemalloc.stop()

    logger.info("Eva has landed successfully.")
    print("\nEva has landed successfully.")

  def _shutdown_handler(self, signum, frame):
    """
    Signal handler that sets the shutdown event to unblock the main thread.
    """
    if not self._shutdown_event.is_set():
      print("\nShutdown signal received. Cleaning up...")
      self._shutdown_event.set()

  def _log_tracemalloc_stats(self):
    """Takes a snapshot of memory usage and logs the top stats."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    logger.info("---- Tracemalloc: Top 10 Memory Usage ----")
    for stat in top_stats[:10]:
      logger.info(f"  {stat}")
    logger.info("------------------------------------------")
