# _logging.py

import getpass
import inspect
import logging
import os
import socket
import sys
from datetime import datetime
from typing import Optional, Set, Tuple, Union

# ---------------------------- #
# --- Environment Settings --- #
# ---------------------------- #


if "EVA_LOGGER_ACTIVATED" not in os.environ:
  os.environ["EVA_LOGGER_ACTIVATED"] = "False"


# ------------------ #
# --- Functional --- #
# ------------------ #


def _remove_log_file(file_path: Union[os.PathLike, str]):
  if (
    "EVA_LOGGER_ACTIVATED" in os.environ
    and os.environ["EVA_LOGGER_ACTIVATED"] == "True"
  ):
    return

  try:
    os.remove(file_path)
  except FileNotFoundError:
    pass
  except PermissionError:
    print(f"Permission denied: cannot remove '{file_path}'.")
  except Exception as e:
    print(f"An error occurred: {e}")


def setup_eva_logger(
  log_dir: Union[str, os.PathLike],
  cache_dir: Union[str, os.PathLike],
  streaming: bool = False,
) -> logging.Logger:
  # --------------------- #
  # --- Configuration --- #
  # --------------------- #

  # Get the top-level logger for lib
  # make all loggers in the lib the children of this logger
  eva_logger = logging.getLogger("eva")

  # Set the minimum level for the lib's logger
  # this can be controlled by an environmental variable for flexibility
  eva_logger.setLevel(logging.DEBUG)  # Capture all logs, filtering happens in handlers

  # Get dynamic info
  # script_name = os.path.splitext(os.path.basename(__file__))[0]
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  hostname = socket.gethostname()
  pid = os.getpid()
  user = getpass.getuser()

  # -------------------- #
  # --- Cache Logger --- #
  # -------------------- #

  cache_log_filename = f"[{timestamp}]-[HOST:{hostname}]-[PID:{pid}]-[USER:{user}].log"
  cache_log_filepath = os.path.join(cache_dir, cache_log_filename)

  cache_log_file_handler = logging.FileHandler(
    filename=cache_log_filepath, mode="a+", encoding="utf-8"
  )
  cache_log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(threadName)s:%(process)d [%(filename)s:%(funcName)s():%(lineno)d] %(message)s"
  )

  cache_log_file_handler.setFormatter(cache_log_formatter)
  cache_log_file_handler.setLevel(logging.DEBUG)

  # Add handler to lib's logger
  eva_logger.addHandler(cache_log_file_handler)

  # -------------------- #
  # --- Debug Logger --- #
  # -------------------- #

  debug_log_filename = "debug.log"
  debug_log_filepath = os.path.join(log_dir, debug_log_filename)
  _remove_log_file(debug_log_filepath)

  debug_log_file_handler = logging.FileHandler(
    filename=debug_log_filepath, mode="a+", encoding="utf-8"
  )
  debug_log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(threadName)s:%(process)d [%(filename)s:%(funcName)s():%(lineno)d] %(message)s"
  )

  debug_log_file_handler.setFormatter(debug_log_formatter)
  debug_log_file_handler.setLevel(logging.DEBUG)

  # Add handler to lib's logger
  eva_logger.addHandler(debug_log_file_handler)

  # ------------------- #
  # --- Info Logger --- #
  # ------------------- #

  info_log_filename = "info.log"
  info_log_filepath = os.path.join(log_dir, info_log_filename)
  _remove_log_file(info_log_filepath)

  info_log_file_handler = logging.FileHandler(
    filename=info_log_filepath, mode="a+", encoding="utf-8"
  )
  info_log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(threadName)s:%(process)d [%(filename)s:%(funcName)s():%(lineno)d] %(message)s"
  )

  info_log_file_handler.setFormatter(info_log_formatter)
  info_log_file_handler.setLevel(logging.INFO)

  # Add handler to lib's logger
  eva_logger.addHandler(info_log_file_handler)

  # ---------------------- #
  # --- Message Logger --- #
  # ---------------------- #

  msg_log_filename = "message.log"
  msg_log_filepath = os.path.join(log_dir, msg_log_filename)
  _remove_log_file(msg_log_filepath)

  msg_log_file_handler = logging.FileHandler(
    filename=msg_log_filepath, mode="a+", encoding="utf-8"
  )
  msg_log_formatter = logging.Formatter("%(levelname)-8s | %(message)s")

  msg_log_file_handler.setFormatter(msg_log_formatter)
  msg_log_file_handler.setLevel(logging.DEBUG)

  # Add handler to lib's logger
  eva_logger.addHandler(msg_log_file_handler)

  # --------------------- #
  # --- Stream Logger --- #
  # --------------------- #

  if streaming:
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(cache_log_formatter)
    eva_logger.addHandler(stream_handler)

  # (IMPORTANT) Stop log messages from propagating to the root logger
  # otherwise, we might duplicate log messages if the user's script
  # also configures the root logger
  eva_logger.propagate = False
  os.environ["EVA_LOGGER_ACTIVATED"] = "True"

  return eva_logger


class _EvaLogger:
  # This is our 'gatekeeper' set, it will store locations
  # of log calls that have already been executed
  # A location is a tuple of (filename, line_number)
  _logged_once_locations: Set[Tuple[str, int]] = set()

  def __init__(self):
    # We will still use Python's standard logger under the hood.
    # We will get the logger for our library, 'eva'.
    self.logger = logging.getLogger("eva")

    def debug_once(message: str, *args, **kwargs):
      """Logs a DEBUG message only once from a given call site"""
      caller_frame = inspect.stack()[1]
      location = (caller_frame.filename, caller_frame.lineno)

      if location not in self._logged_once_locations:
        self.logger.debug(message, *args, **kwargs)
        self._logged_once_locations.add(location)

    def info_once(message: str, *args, **kwargs):
      """Logs an INFO message only once from a given call site"""
      # inspect.stack()[1] gets the frame of the *caller* of this function
      # [0] would be the frame of the info_once itself
      caller_frame = inspect.stack()[1]
      location = (caller_frame.filename, caller_frame.lineno)

      # check the logging status of the this session
      if location not in self._logged_once_locations:
        self.logger.info(message, *args, **kwargs)
        self._logged_once_locations.add(location)

    def warning_once(message: str, *args, **kwargs):
      """Logs a WARNING message only once from a given call site"""
      caller_frame = inspect.stack()[1]
      location = (caller_frame.filename, caller_frame.lineno)

      if location not in self._logged_once_locations:
        self.logger.warning(message, *args, **kwargs)
        self._logged_once_locations.add(location)

    def critical(message: str, *args, **kwargs):
      """Logs a CRITICAL message every time it's called"""
      self.logger.critical(message, *args, **kwargs)

    self.logger.info_once = info_once  # type: ignore[attr-defined]
    self.logger.warning_once = warning_once  # type: ignore[attr-defined]
    self.logger.debug_once = debug_once  # type: ignore[attr-defined]
    self.logger.critical = critical  # type: ignore[attr-defined]

  def get_logger(self, name: Optional[str] = None) -> logging.Logger:
    """Returns the Eva Logger

    Args:
      name(None | str): sometimes, or actually for most time, the called function may receive a
        `name` argument, however, considering we are absolutely using the Eva Logger, we don't need
        this and simply omit the argument.
    """

    return self.logger


# _logging.py
