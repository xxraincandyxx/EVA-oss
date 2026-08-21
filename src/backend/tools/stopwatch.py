# stopwatch.py

import time
from typing import Callable, Dict, List

# ----------------- #
# --- Stopwatch --- #
# ----------------- #


class Stopwatch:
  def __init__(self):
    self.sessions: Dict[str, List[float]] = {}
    self.current_session = None
    self.start_time: float | None = None

  def start(self, session_name: str):
    if self.current_session is not None:
      self.stop()  # Stop current session if one is running
    self.current_session = session_name
    self.start_time = time.perf_counter()  # High-resolution timer

  def stop(self):
    if self.current_session is None or self.start_time is None:
      return

    end_time: float = time.perf_counter()
    duration: float = end_time - self.start_time

    # Record the session
    if self.current_session in self.sessions:
      self.sessions[self.current_session].append(duration)
    else:
      self.sessions[self.current_session] = [duration]

    self.current_session = None

  def get_results(self) -> Dict[str, Dict[str, float]]:
    return {
      name: {
        "runs": len(times),
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
      }
      for name, times in self.sessions.items()
    }

  def view_results(self, view_func: Callable[[str], None]):
    view_func("")
    view_func("--- Stopwatch ---")
    results = self.get_results()
    for idx, (name, stats) in enumerate(results.items()):
      if idx != 0:
        view_func("-----------------")
      view_func(f"Session: {name}")
      view_func(f"  Runs: {stats['runs']}")
      view_func(f"  Total: {stats['total_time']:.4f}s")
      view_func(f"  Avg: {stats['avg_time']:.4f}s")
      view_func(f"  Min: {stats['min_time']:.4f}s")
      view_func(f"  Max: {stats['max_time']:.4f}s")
    view_func("--- endStopwatch ---")
    view_func("")


# stopwatch.py
