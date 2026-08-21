# manager.py

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..paths import DATA_DIR
from ..utils import get_logger

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #

logger = get_logger()

# -------------------- #
# --- Core Classes --- #
# -------------------- #


class EvaDBManager:
  def __init__(self, db_file: Optional[Union[str, os.PathLike]] = None):
    if db_file is None:
      db_file = DATA_DIR / "schedules.db"
    self.db_file = Path(db_file)
    self.db_file.parent.mkdir(parents=True, exist_ok=True)

    self.thread_local = threading.local()

    with self.get_connection() as conn:
      self._setup_database(conn)

  # ---------------------------- #
  # --- Connection Management --- #
  # ---------------------------- #

  def _get_thread_connection(self) -> sqlite3.Connection:
    """Gets or creates a DB connection for the current thread."""
    if not hasattr(self.thread_local, "connection"):
      try:
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        self.thread_local.connection = conn
        logger.debug(f"Created new DB connection for thread {threading.get_ident()}")
      except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    return self.thread_local.connection

  @contextmanager
  def get_connection(self) -> Iterator[sqlite3.Connection]:
    """Provides a transactional scope around a series of operations."""
    conn = self._get_thread_connection()
    try:
      yield conn
    except sqlite3.Error as e:
      logger.error(f"Database operation failed: {e}. Rolling back transaction.")
      conn.rollback()
      raise
    else:
      conn.commit()

  def close_thread_connection(self):
    """Closes the connection for the current thread, if it exists."""
    if hasattr(self.thread_local, "connection"):
      self.thread_local.connection.close()
      del self.thread_local.connection
      logger.debug(f"Closed DB connection for thread {threading.get_ident()}")

  # ---------------------------- #
  # --- Initialization Setup --- #
  # ---------------------------- #

  def _setup_database(self, conn: sqlite3.Connection):
    """Create tables if they don't exist"""
    try:
      cursor = conn.cursor()
      cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_index (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          desc TEXT NOT NULL
        )
        """
      )

      # --- [BUG FIX] --- schedule_id must be INTEGER to match the foreign key.
      cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cartesian (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          action TEXT NOT NULL,
          name TEXT NOT NULL,
          schedule_id INTEGER NOT NULL,
          cartesian_data TEXT NOT NULL,
          FOREIGN KEY (schedule_id) REFERENCES schedule_index(id) ON DELETE CASCADE
        )
        """
      )

    except sqlite3.Error as e:
      logger.error(f"Database setup error: {e}")
      raise

  # ------------------- #
  # --- Public APIs --- #
  # ------------------- #

  def insert_data(self, schedule_data: List[Dict], description: str = "Saved Schedule"):
    """
    Insert a list of scheduled actions as a single new schedule.
    """
    if not isinstance(schedule_data, list):
      logger.warning(
        f"EvaDBManager insertion expects a list of actions, but got {type(schedule_data)}. Skipping."
      )
      return

    try:
      with self.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("INSERT INTO schedule_index(desc) VALUES (?)", (description,))
        schedule_id = cursor.lastrowid
        logger.info(
          f"Created new schedule with ID {schedule_id} and description '{description}'."
        )

        for cartesian in schedule_data:
          action = cartesian.get("action", "UNKNOWN")
          name = cartesian.get("name", "Unnamed Step")

          keys_to_remove = ["id", "action", "name"]
          filtered_cartesian_data = cartesian.copy()
          for key in keys_to_remove:
            filtered_cartesian_data.pop(key, None)

          cartesian_json_data = json.dumps(filtered_cartesian_data, indent=2)

          cursor.execute(
            "INSERT INTO cartesian (action, name, schedule_id, cartesian_data) VALUES (?, ?, ?, ?)",
            (action, name, schedule_id, cartesian_json_data),
          )
        logger.info(
          f"Inserted {len(schedule_data)} actions for schedule ID {schedule_id}."
        )
    except Exception as e:
      logger.error(f"Failed to insert schedule data: {e}")

  def retrieve_data(self) -> Dict[str, Dict]:
    """
    Retrieve all schedules from the database.
    """
    ret = {}
    try:
      with self.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id, desc FROM schedule_index ORDER BY id DESC;")
        schedule_indices = cursor.fetchall()

        for schedule_id, desc in schedule_indices:
          ret[str(schedule_id)] = {"Desc": desc, "Cartesians": []}

          cursor.execute(
            "SELECT id, action, name, cartesian_data FROM cartesian WHERE schedule_id = ? ORDER BY id ASC;",
            (schedule_id,),
          )
          retrieved_items = cursor.fetchall()

          for _id, _action, _name, _cartesian_json_data in retrieved_items:
            try:
              cartesian_dict = {
                "id": str(_id),
                "action": _action,
                "name": _name,
              }
              cartesian_partial_dict = json.loads(_cartesian_json_data)
              cartesian_dict.update(cartesian_partial_dict)
              ret[str(schedule_id)]["Cartesians"].append(cartesian_dict)
            except json.JSONDecodeError as e:
              logger.error(
                f"Skipping corrupted cartesian data (ID: {_id}) for schedule {schedule_id}: {e}"
              )

    except Exception as e:
      logger.error(f"Failed to retrieve data from database: {e}")

    return ret

  def release(self):
    """Clear all data within the tables"""
    try:
      with self.get_connection() as conn:
        cursor = conn.cursor()
        logger.warning("Clearing all data from schedule tables...")

        cursor.execute("DELETE FROM cartesian")
        cursor.execute("DELETE FROM schedule_index")

        cursor.execute("DELETE FROM sqlite_sequence WHERE name='cartesian'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='schedule_index'")

        logger.info("All schedule data has been eliminated.")
    except Exception as e:
      logger.error(f"Failed to release database tables: {e}")


# manager.py ends here
