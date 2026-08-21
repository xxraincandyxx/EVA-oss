# _sys.py
# system-wide functional tools
# for the usage of some of the functions,
# the user may should grant certain access permissions first

import glob
import inspect
import json
import os
import platform
import re
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

import cv2

from ..utils import get_logger

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #


logger = get_logger()


# ----------------------- #
# --- Local Variables --- #
# ----------------------- #


_g_get_sys_camera_index_called_times = 0


# ----------------------------- #
# --- System-wide Functions --- #
# ----------------------------- #


def get_sys_camera_index(max_index: int = 10) -> Optional[int]:
  """
  Finds the first available system camera index that can be successfully opened.

  This function iterates through possible camera indices (0 to max_index-1) and
  returns the first index that can be opened by OpenCV. Works cross-platform
  (Linux and macOS).

  Args:
      max_index: Maximum index to check (default: 10)

  Returns:
      int: First working camera index, or None if no camera is found

  Example:
      >>> cam_idx = get_sys_camera_index()
      >>> print(f"System camera index: {cam_idx}")

  NOTE: This function is supposed to be run with caution, 'cause if a camera is
  instantiated, it is not allowed to access twice, and it will potentially find no
  accessible cameras and return a NoneType value, leading to unexpected performance.
  """

  global _g_get_sys_camera_index_called_times
  _g_get_sys_camera_index_called_times += 1

  # Get the frame of the caller
  caller_frame = inspect.currentframe().f_back
  # Get info about the caller
  caller_name = caller_frame.f_code.co_name
  caller_file = caller_frame.f_code.co_filename
  caller_line = caller_frame.f_lineno

  logger.info(
    f"get_sys_camera_index() was called by {caller_name}() in {caller_file} at line {caller_line} - Called time: {_g_get_sys_camera_index_called_times}"
  )

  # Clean up the frame reference to avoid reference cycles
  del caller_frame

  # Function Core Logic
  for index in range(max_index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
      cap.release()
      return index
    cap.release()
  return None


def list_sys_all_available_cameras(max_index: int = 10) -> List[Dict[str, Any]]:
  """
  Lists all available cameras with detailed information including index, name,
  and device path (Linux) or unique ID (macOS). Works cross-platform.

  On Linux:
      - Uses /dev/video* devices and sysfs for camera names
      - Returns physical device paths
  On macOS:
      - Uses system_profiler to get camera details
      - Returns unique camera IDs

  Args:
      max_index: Maximum index to check for OpenCV devices (default: 10)

  Returns:
      List of dictionaries with camera details:
      [{
          'index': camera index (int),
          'name': camera name (str),
          'device_path': device path (Linux) or unique ID (macOS) (str),
          'platform_specific': additional platform info (dict),
          'is_working': True if camera can be opened (bool)
      }]

  Example:
      >>> cameras = list_sys_all_available_cameras()
      >>> for cam in cameras:
      >>>     print(f"Index: {cam['index']}, Name: {cam['name']}")
  """
  system = platform.system()
  cameras = []

  # Linux-specific camera detection
  if system == "Linux":
    # Get all video devices
    video_devices = sorted(
      [dev for dev in os.listdir("/dev") if re.match(r"video\d+$", dev)]
    )

    for dev in video_devices:
      index = int(dev[5:])  # Extract number from 'videoX'
      dev_path = f"/dev/{dev}"

      # Get camera name from sysfs if available
      name = f"Camera {index}"
      sysfs_path = f"/sys/class/video4linux/{dev}/name"
      if os.path.exists(sysfs_path):
        try:
          with open(sysfs_path, "r") as f:
            name = f.read().strip()
        except IOError:
          pass

      # Verify with OpenCV
      cap = cv2.VideoCapture(index)
      is_working = cap.isOpened()
      if is_working:
        cap.release()

      cameras.append(
        {
          "index": index,
          "name": name,
          "device_path": dev_path,
          "platform_specific": {
            "sysfs_path": sysfs_path,
          },
          "is_working": is_working,
        }
      )

  # macOS-specific camera detection
  elif system == "Darwin":
    # Try to get camera info using system_profiler
    try:
      result = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True,
        text=True,
        check=True,
      )
      camera_data = json.loads(result.stdout)

      # Extract camera details
      cam_list = camera_data.get("SPCameraDataType", [])
      for i, cam in enumerate(cam_list):
        unique_id = cam.get("_name", f"mac_cam_{i}")
        model = cam.get("model_id", f"Camera {i}")

        # Verify with OpenCV
        cap = cv2.VideoCapture(i)
        is_working = cap.isOpened()
        if is_working:
          cap.release()

        cameras.append(
          {
            "index": i,
            "name": model,
            "device_path": unique_id,
            "platform_specific": {
              "spokect": cam.get("spokect", ""),
              "coremedia_id": cam.get("coremedia_id", ""),
            },
            "is_working": is_working,
          }
        )
    except (subprocess.CalledProcessError, FileNotFoundError, SyntaxError):
      # Fallback to OpenCV-only method
      for index in range(max_index):
        cap = cv2.VideoCapture(index)
        is_working = cap.isOpened()
        if is_working:
          cameras.append(
            {
              "index": index,
              "name": f"Camera {index}",
              "device_path": f"mac_cam_{index}",
              "platform_specific": {},
              "is_working": True,
            }
          )
          cap.release()

  # Windows/other platforms (not requested but included for completeness)
  else:
    for index in range(max_index):
      cap = cv2.VideoCapture(index)
      is_working = cap.isOpened()
      if is_working:
        cameras.append(
          {
            "index": index,
            "name": f"Camera {index}",
            "device_path": f"cam_{index}",
            "platform_specific": {},
            "is_working": True,
          }
        )
        cap.release()

  return cameras


def release_sys_all_using_cameras(
  aggressive: bool = False,
  device_paths: Optional[List[str]] = None,
  timeout: float = 2.0,
) -> Dict[str, Any]:
  """
  Releases camera resources by terminating processes locking camera devices.
  Primarily effective on Linux systems. Use with caution!

  Args:
      aggressive: If True, force-terminate processes (default: False)
      device_paths: Specific camera devices to release (default: all)
      timeout: Wait time before force termination (seconds)

  Returns:
      Report dictionary with operation details

  Note:
      - On macOS, only processes started by current user can be terminated
      - Requires appropriate permissions (might need sudo on Linux)
      - Force termination can cause data loss in other applications

  Example:
      >>> report = release_cameras(aggressive=True)
      >>> print(report['summary'])
  """
  system = platform.system()
  report = {
    "platform": system,
    "aggressive": aggressive,
    "devices": device_paths or [],
    "actions": [],
    "summary": "",
    "error": None,
  }
  current_pid = os.getpid()

  try:
    # Linux device handling
    if system == "Linux":
      if not device_paths:
        device_paths = glob.glob("/dev/video*")

      for device in device_paths:
        action = {"device": device, "pids": [], "killed": []}

        # Find processes using the device
        try:
          lsof = subprocess.run(
            ["lsof", "-t", device],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
          )
          pids = [int(pid) for pid in lsof.stdout.splitlines() if pid.strip()]
        except (ValueError, subprocess.SubprocessError):
          pids = []

        action["pids"] = pids

        # Terminate processes
        for pid in pids:
          if pid == current_pid:
            continue  # Skip self

          try:
            os.kill(pid, signal.SIGTERM)
            action["killed"].append(pid)

            # Force terminate if needed
            if aggressive:
              time.sleep(timeout)
              try:
                os.kill(pid, signal.SIGKILL)
                action["killed"].append(pid)
              except ProcessLookupError as e:
                logger.error(f"Killing camera error: {e}")

          except (ProcessLookupError, PermissionError) as e:
            action.setdefault("errors", []).append(str(e))

        report["actions"].append(action)

    # macOS handling
    elif system == "Darwin":
      # Get all camera processes using Apple's core services
      try:
        processes = subprocess.run(
          ["pgrep", "-f", "VDC|AppleCamera"], stdout=subprocess.PIPE, text=True
        )
        pids = [int(pid) for pid in processes.stdout.splitlines() if pid.strip()]
      except (ValueError, subprocess.SubprocessError):
        pids = []

      action = {"device": "coremedia", "pids": pids, "killed": []}

      # Terminate processes
      for pid in pids:
        if pid == current_pid:
          continue

        try:
          os.kill(pid, signal.SIGTERM)
          action["killed"].append(pid)

          if aggressive:
            time.sleep(timeout)
            try:
              os.kill(pid, signal.SIGKILL)
              action["killed"].append(pid)
            except ProcessLookupError:
              pass

        except (ProcessLookupError, PermissionError) as e:
          action.setdefault("errors", []).append(str(e))

      report["actions"].append(action)

  except Exception as e:
    report["error"] = str(e)

  # Generate summary
  total_killed = sum(len(action["killed"]) for action in report["actions"])
  report["summary"] = (
    f"Released {total_killed} processes across {len(report['actions'])} devices"
  )

  return report


def get_system_memory() -> float:
  """Return installed system memory in GiB."""
  system = platform.system()

  if system == "Windows":
    import ctypes

    ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory.restype = (
      ctypes.c_ulonglong
    )
    memory = ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory()
    return memory / (1024**2)

  if system == "Linux":
    with open("/proc/meminfo", "r") as mem:
      for line in mem:
        if line.startswith("MemTotal:"):
          return int(line.split()[1]) / (1024**2)

  if system == "Darwin":  # macOS
    import subprocess

    mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
    return int(mem.strip()) / (1024**3)

  raise NotImplementedError(f"Platform {system} not supported")


# ------------------------ #
# --- Demo [Use Guide] --- #
# ------------------------ #


def sys_camera_tools_demo():
  # --- Example Usage --- #
  print("System Camera Index:", get_sys_camera_index())

  print("\nAvailable Cameras:")
  cameras = list_sys_all_available_cameras()
  for cam in cameras:
    status = "Working" if cam["is_working"] else "Not Working"
    print(f"Index: {cam['index']} | Name: {cam['name']} | Status: {status}")

  print("\nReleasing cameras...")
  release_report = release_sys_all_using_cameras(aggressive=True)
  print(release_report["summary"])
  if release_report["actions"]:
    print("Detailed actions:")
    for action in release_report["actions"]:
      print(f"- Device: {action['device']}")
      print(f"  PIDs found: {action['pids']}")
      print(f"  Killed: {action['killed']}")


# _sys.py ends here
