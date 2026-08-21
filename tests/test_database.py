from backend.database.manager import EvaDBManager


def test_schedule_database_round_trip(tmp_path):
  database = EvaDBManager(tmp_path / "schedules.db")
  schedule = [
    {
      "id": "temporary-client-id",
      "action": "move",
      "name": "inspection pose",
      "x": 0.25,
      "duration": 1.5,
    }
  ]

  database.insert_data(schedule, description="Inspection")
  stored = database.retrieve_data()

  assert len(stored) == 1
  record = next(iter(stored.values()))
  assert record["Desc"] == "Inspection"
  assert record["Cartesians"][0] == {
    "id": "1",
    "action": "move",
    "name": "inspection pose",
    "x": 0.25,
    "duration": 1.5,
  }

  database.close_thread_connection()
