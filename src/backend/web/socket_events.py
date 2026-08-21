# web/socket_events.py

import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Union

import cv2
import numpy as np
from flask import url_for
from flask_socketio import emit  # type: ignore

from ..agent import EvaAgent
from ..tools.functional import purify_action_data
from ..utils import get_logger
from ..utils.structs import FrameInfo

if TYPE_CHECKING:
  from ..server import EvaSocketRelay

# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================

logger = get_logger()

# Global flags to ensure background tasks are started only once.
status_updater_started = False
cache_updater_started = False

# ==============================================================================
# EVENTS REGISTRATION
# ==============================================================================


def register_socket_events(relay: "EvaSocketRelay"):
  """
  Registers all Socket.IO event handlers for the application.
  """
  global status_updater_started, cache_updater_started
  socketio = relay.socketio

  eva_agent = EvaAgent(
    eva_console=relay.eva_console, config=relay.config, socketio=socketio
  )

  repeating_schedule_thread = None
  stop_repeating_flag = threading.Event()

  # ----------------------------------------------------------------------------
  # Connection Handlers
  # ----------------------------------------------------------------------------

  @socketio.on("connect")
  def handle_connect(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    logger.info("Client connected")
    emit("update_system_online_status", {"online": relay.is_online()})
    emit(
      "llm_backend_changed",
      {
        "backend": eva_agent.backend,
        "local_url": getattr(eva_agent, "local_llm_url", ""),
      },
    )

    if not status_updater_started:
      socketio.start_background_task(target=send_status_updates)
      globals()["status_updater_started"] = True
      logger.info("Started status updater background task.")

    if not cache_updater_started:
      socketio.start_background_task(target=send_caches_updates)
      globals()["cache_updater_started"] = True
      logger.info("Started cache updater background task.")

  @socketio.on("disconnect")
  def handle_disconnect(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    logger.info("Client disconnected")
    stop_repeating_flag.set()

  # ----------------------------------------------------------------------------
  # Background Tasks
  # ----------------------------------------------------------------------------

  def send_status_updates():
    update_interval = 0.2
    while True:
      if relay.eva_console:
        status_update = {
          "position": relay.eva_console.get_current_end_position(),
          "orientation": relay.eva_console.get_current_end_pose(),
          "thetas": relay.eva_console.get_current_thetas(),
        }
        # SpikeYOLO-lite: 暴露门控状态（若启用）
        try:
          if relay.eva_eye and hasattr(relay.eva_eye, "get_gating_state"):
            gs = relay.eva_eye.get_gating_state()
            status_update["gating"] = gs
        except Exception:
          pass
        socketio.emit("status_update", status_update)
      socketio.sleep(update_interval)

  def send_caches_updates():
    update_interval = 0.5
    while True:
      if relay.schedule_manager is not None:
        caches_update = {"scheduled_cartesians": relay.schedule_manager.get_actions()}
        socketio.emit("caches_update", caches_update)
      socketio.sleep(update_interval)

  # ----------------------------------------------------------------------------
  # 3D Simulation
  # ----------------------------------------------------------------------------

  simulation_active = threading.Event()

  @socketio.on("toggle_live_simulation")
  def handle_toggle_live_simulation(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, bool],
  ):
    is_active = data.get("active", False)
    if is_active:
      logger.info("Live simulation ACTIVATED by client.")
      simulation_active.set()
    else:
      logger.info("Live simulation DEACTIVATED by client.")
      simulation_active.clear()

  def _emit_arm_sim_update(
    states_list: List[Union[np.ndarray[Any, np.dtype[Any]], List[List[float]]]],
  ):
    serializable_states = [
      states.tolist() if not isinstance(states, List) else states
      for states in states_list
    ]
    socketio.emit("update_arm_sim_data", {"states_list": serializable_states})

  @socketio.on("refresh_arm_sim")
  def handle_refresh_arm_sim(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.eva_console:
      _emit_arm_sim_update([relay.eva_console.get_current_states()])

  @socketio.on("restore_arm_sim")
  def handle_restore_arm_sim(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.eva_console:
      relay.eva_console.restore()
      _emit_arm_sim_update([relay.eva_console.get_init_states()])

  def _live_simulation_updater():
    while True:
      simulation_active.wait()
      if relay.eva_console:
        _emit_arm_sim_update([relay.eva_console.get_current_states()])
      socketio.sleep(0.1 if not relay.config._shrink else 0.25)

  socketio.start_background_task(_live_simulation_updater)

  @socketio.on("simulate_with_cartesian")
  def handle_simulate_with_cartesian(args: Dict[str, Any]):
    if not relay.eva_console:
      return
    sim_states = relay.eva_console.get_states_with_cartesian(
      x=float(args["x"]),
      y=float(args["y"]),
      z=float(args["z"]),
      a=float(args["a"]),
      b=float(args["b"]),
      c=float(args["c"]),
    )
    current_states = relay.eva_console.get_current_states()
    _emit_arm_sim_update([current_states, sim_states])

  @socketio.on("simulate_with_axes")
  def handle_simulate_with_axes(  # pyright: ignore[reportUnusedFunction]
    args: Dict[str, Any],
  ):
    if not relay.eva_console:
      return
    thetas = [float(args.get(f"axis{i + 1}", 0.0)) for i in range(6)]
    current_thetas = relay.eva_console.get_current_thetas()
    target_thetas = [current_thetas[i] + thetas[i] for i in range(6)]
    sim_states = relay.eva_console.get_states_with_thetas(thetas=target_thetas)
    current_states = relay.eva_console.get_current_states()
    _emit_arm_sim_update([current_states, sim_states])

  last_state_before_move = None

  @socketio.on("emit_with_cartesian")
  def handle_emit_with_cartesian(  # pyright: ignore[reportUnusedFunction]
    args: Dict[str, Any],
  ):
    if not relay.eva_console:
      return
    nonlocal last_state_before_move
    cartesian = [
      float(args[key]) for key in ["x", "y", "z", "a", "b", "c", "duration", "rotation"]
    ]
    last_state_before_move = [
      *relay.eva_console.get_current_thetas(),
      cartesian[6],
      cartesian[7],
    ]
    relay.eva_console.ctrl_with_cartesian(cartesian[:6], cartesian[6], cartesian[7])
    socketio.sleep(0.05)
    handle_refresh_arm_sim()

  @socketio.on("emit_with_axes")
  def handle_emit_with_axes(  # pyright: ignore[reportUnusedFunction]
    args: Dict[str, Any],
  ):
    if not relay.eva_console:
      return
    nonlocal last_state_before_move
    thetas = [float(args[f"axis{i + 1}"]) for i in range(6)]
    duration, rotation = float(args["duration"]), float(args["rotation"])
    last_state_before_move = [
      *relay.eva_console.get_current_thetas(),
      duration,
      rotation,
    ]
    relay.eva_console.ctrl_with_thetas(thetas, duration=duration, rotation=rotation)
    socketio.sleep(0.05)
    handle_refresh_arm_sim()

  @socketio.on("undo_last_move")
  def handle_undo_last_move(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if not relay.eva_console:
      return
    nonlocal last_state_before_move
    if last_state_before_move:
      logger.info(f"UNDO triggered. Reverting to state: {last_state_before_move}")
      target_thetas, duration, rotation = (
        last_state_before_move[:6],
        last_state_before_move[6],
        last_state_before_move[7],
      )
      current_thetas = relay.eva_console.get_current_thetas()
      delta_thetas = [target_thetas[i] - current_thetas[i] for i in range(6)]
      relay.eva_console.ctrl_with_thetas(
        delta_thetas, duration=duration, rotation=rotation
      )
      last_state_before_move = None
      socketio.sleep(0.05)
      handle_refresh_arm_sim()
    else:
      logger.warning("UNDO triggered, but no previous state was found.")

  is_tracking = False

  @socketio.on("fpv_move")
  def handle_fpv_move(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Dict[str, float]],
  ):
    if not relay.eva_console:
      return
    move_state, config = data.get("moveState", {}), data.get("config", {})
    pos_increment, orient_increment, duration = (
      config.get("pos_increment", 0.005),
      config.get("orient_increment", 0.5),
      config.get("duration", 0.1),
    )
    pos_delta = [move_state.get(axis, 0.0) * pos_increment for axis in ["x", "y", "z"]]
    ori_delta_deg = [
      move_state.get("a", 0.0) * orient_increment * -1,
      move_state.get("b", 0.0) * orient_increment,
      move_state.get("c", 0.0) * orient_increment,
    ]
    relay.eva_console.fpv_mv_(
      pos_delta=pos_delta, ori_delta_deg=ori_delta_deg, duration=duration
    )
    if not simulation_active.is_set():
      handle_refresh_arm_sim()

  @socketio.on("start_tracking")
  def handle_start_tracking(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    nonlocal is_tracking
    if not is_tracking:
      logger.info("Starting FPV track recording...")
      is_tracking = True
      if relay.schedule_manager is not None:
        relay.schedule_manager.clear()
      socketio.start_background_task(target=_record_track_thread)
      emit("tracking_status_update", {"is_tracking": True})

  @socketio.on("end_tracking")
  def handle_end_tracking(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    nonlocal is_tracking
    if is_tracking:
      logger.info("Stopping FPV track recording.")
      is_tracking = False
      emit("tracking_status_update", {"is_tracking": False})

  @socketio.on("request_tracking_status")
  def handle_request_tracking_status(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    emit("tracking_status_update", {"is_tracking": is_tracking})

  def _record_track_thread():
    nonlocal is_tracking
    tracking_interval = 0.2
    while is_tracking:
      if relay.eva_console and (relay.schedule_manager is not None):
        pos, orient_deg = relay.eva_console.get_current_orientation(rad=False)
        cartesian_state = pos + orient_deg + [tracking_interval, 0.0]
        relay.schedule_manager.add(cartesian=cartesian_state, action="CARTESIAN")
      socketio.sleep(tracking_interval)
    logger.info(
      f"Recording stopped. {len(relay.schedule_manager.get_actions())} points recorded."
    )

  @socketio.on("schedule_with_cartesian")
  def handle_schedule_with_cartesian(  # pyright: ignore[reportUnusedFunction]
    args: Dict[str, Any],
  ):
    if (relay.schedule_manager is None) or relay.eva_console:
      return
    cartesian_data = [
      float(args.get(k, 0.0))
      for k in ["x", "y", "z", "a", "b", "c", "duration", "rotation"]
    ]
    relay.schedule_manager.add(cartesian=cartesian_data, action="CARTESIAN")

  @socketio.on("schedule_with_axes")
  def handle_schedule_with_axes(  # pyright: ignore[reportUnusedFunction]
    args: Dict[str, Any],
  ):
    if (relay.schedule_manager is None) or relay.eva_console:
      return
    thetas = [float(args.get(f"axis{i + 1}", 0.0)) for i in range(6)]
    duration, rotation = (
      float(args.get("duration", 0.0)),
      float(args.get("rotation", 0.0)),
    )
    cartesian_equiv = relay.eva_console.get_cartesian_with_thetas(
      thetas, duration, rotation
    )
    relay.schedule_manager.add(axes=thetas, cartesian=cartesian_equiv, action="AXES")

  simple_schedule_actions = [
    "ROT_CLAMP",
    "ROT_RELEASE",
    "ROT_ROTATE",
    "PUMP_ATTACH",
    "PUMP_DETACH",
    "PUMP_SHUTDOWN",
    "SUSPEND",
    "CAPTURE",
  ]

  def create_schedule_handler(action_name: str):
    @socketio.on(f"schedule_{action_name.lower()}")
    def schedule_handler(  # pyright: ignore[reportUnusedFunction]
      *args,  # pyright: ignore
      **kwargs,  # pyright: ignore
    ):
      logger.debug(f"SocketEvents: Schedule for ({action_name}) is triggered.")
      if relay.schedule_manager is not None:
        # NOTE: action_name is the variable captured by the factory
        relay.schedule_manager.add(action=action_name)

    # We don't need to return anything, as the decorator registered the
    # function. The return value of the factory itself doesn't matter here.

  # Call the factory inside the loop
  for action in simple_schedule_actions:
    create_schedule_handler(action)

  @socketio.on("clear_schedule")
  def handle_clear_schedule(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.schedule_manager is not None:
      relay.schedule_manager.clear()

  @socketio.on("pop_schedule")
  def handle_pop_schedule(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.schedule_manager is not None:
      relay.schedule_manager.pop()

  @socketio.on("emit_schedule")
  def handle_emit_schedule(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.schedule_manager is not None:
      socketio.start_background_task(relay.schedule_manager.emit_all)

  @socketio.on("save_schedule")
  def handle_save_schedule(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Any],
  ):
    if (relay.schedule_manager is not None) and (relay.eva_db is not None):
      actions, description = (
        relay.schedule_manager.get_actions(),
        data.get("description", "Untitled Schedule"),
      )
      relay.eva_db.insert_data(actions, description)
      emit(
        "processing_result",
        {
          "status": "SUCCESS",
          "message": f"Schedule '{description}' saved to database.",
        },
      )

  @socketio.on("request_details")
  def handle_request_details(  # pyright: ignore[reportUnusedFunction]
    prompt: str,
  ):
    if relay.eva_db is not None:
      emit("details_response", relay.eva_db.retrieve_data())

  @socketio.on("item_selected")
  def handle_item_selection(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Any],
  ):
    selected_id = data.get("id")
    if (relay.eva_db is not None) and (relay.schedule_manager is not None):
      db_data = relay.eva_db.retrieve_data()
      actions = db_data.get(selected_id, {}).get("Cartesians", [])
      loaded_actions = [purify_action_data(a) for a in actions]
      relay.schedule_manager.set_schedule(loaded_actions)
      emit(
        "processing_result",
        {
          "status": "SUCCESS",
          "message": f"Successfully loaded schedule {selected_id}.",
        },
      )

  @socketio.on("eliminate_schedule")
  def handle_eliminate_schedule(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.eva_db:
      relay.eva_db.release()
      emit(
        "processing_result",
        {
          "status": "SUCCESS",
          "message": "All schedules deleted from database.",
        },
      )

  @socketio.on("request_schedule_for_save")
  def handle_request_schedule_for_save(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    if relay.schedule_manager is not None:
      emit(
        "schedule_to_save",
        {"schedule": relay.schedule_manager.get_actions()},
      )

  @socketio.on("load_schedule_from_file")
  def handle_load_schedule_from_file(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Any],
  ):
    new_schedule = data.get("schedule")
    if (new_schedule is not None) and (relay.schedule_manager is not None):
      relay.schedule_manager.set_schedule(new_schedule)
    else:
      logger.warning("Invalid schedule data received from file.")

  peripheral_handlers = {
    "rot_clamp": lambda: relay.eva_console.ctrl_rot("CLAMP"),
    "rot_release": lambda: relay.eva_console.ctrl_rot("RELEASE"),
    "rot_rotate": lambda: relay.eva_console.ctrl_rot("ROTATE"),
    "pump_attach": lambda: relay.eva_console.ctrl_pump("ATTACH"),
    "pump_detach": lambda: relay.eva_console.ctrl_pump("DETACH"),
    "pump_shutdown": lambda: relay.eva_console.ctrl_pump("SHUTDOWN"),
  }
  for event, handler in peripheral_handlers.items():
    # Pass a lambda that accepts *args and **kwargs (which SocketIO will pass)
    # but still only calls the stored handler h (which takes no arguments here).
    # This prevents the TypeError by respecting SocketIO's required function
    # signature.
    socketio.on_event(event, lambda *args, h=handler, **kwargs: h())

  @socketio.on("start_repeating_schedule")
  def handle_start_repeating_schedule(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Any],
  ):
    nonlocal repeating_schedule_thread
    if repeating_schedule_thread and not repeating_schedule_thread._thread.dead:
      logger.warning("Repeat schedule already running.")
      return
    interval = data.get("interval", 10)
    stop_repeating_flag.clear()
    repeating_schedule_thread = socketio.start_background_task(
      _schedule_repeater, interval=interval
    )
    emit("repeat_status_update", {"is_repeating": True})
    logger.info(f"Started schedule repeater with interval {interval}s.")

  @socketio.on("stop_repeating_schedule")
  def handle_stop_repeating_schedule(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    nonlocal repeating_schedule_thread
    if not repeating_schedule_thread:
      logger.warning("Stop repeating was called, but no repeating schedule is running.")
      emit("repeat_status_update", {"is_repeating": False})
      return
    stop_repeating_flag.set()
    emit("repeat_status_update", {"is_repeating": False})
    logger.info("Stopping schedule repeater.")

  def _schedule_repeater(interval: float):
    while not stop_repeating_flag.is_set():
      if relay.schedule_manager is not None:
        relay.schedule_manager.emit_all()
      wait_start_time = time.time()
      while time.time() - wait_start_time < interval:
        if stop_repeating_flag.is_set():
          break
        socketio.sleep(0.1)
    logger.info("Schedule repeater thread has stopped.")

  vision_detection_trigger_counter: int = 0

  @socketio.on("start_detection")
  def handle_start_detection(  # pyright: ignore[reportUnusedFunction]
    *args,  # pyright: ignore
    **kwargs,  # pyright: ignore
  ):
    nonlocal vision_detection_trigger_counter
    if relay.eva_eye is None:
      emit(
        "detection_complete",
        {
          "imageUrl": url_for("static", filename="figures/logo.png"),
          "error": "Vision system not ready.",
        },
      )
      return
    relay.eva_console.flash(duration=relay.config._flash_interval)
    frame = relay.eva_eye.get_latest_real_frame()
    if frame is None:
      logger.warning("Cannot get a frame for detection.")
      emit(
        "detection_complete",
        {
          "imageUrl": url_for("static", filename="figures/logo.png"),
          "error": "No camera frame available.",
        },
      )
      return
    frame_info: FrameInfo = relay.eva_console.get_frame_info()
    if relay.config.game__ and vision_detection_trigger_counter < 2:
      if vision_detection_trigger_counter == 0:
        frame_info.desc = "True 0.25 0.25"
      elif vision_detection_trigger_counter == 1:
        frame_info.desc = "True 0.5 0.5"
    vision_detection_trigger_counter += 1
    relay.eva_eye.put_robo_queue_in((frame, frame_info))
    start_time, processed_frame = time.time(), None
    while time.time() - start_time < 1.6:
      processed_frame = relay.eva_eye.get_robo_queue_out()
      if processed_frame is not None:
        socketio.sleep(2.4)
        break
      socketio.sleep(0.05)
    if processed_frame is not None:
      result_fn = f"detect_result_{int(time.time())}.jpg"
      result_fp = os.path.join(relay.app.static_folder, "captures", result_fn)
      cv2.imwrite(result_fp, processed_frame)
      image_url = url_for("static", filename=f"captures/{result_fn}")
      emit("detection_complete", {"imageUrl": image_url})
    else:
      logger.warning("Detection timeout: No processed frame received from RoboEye.")
      emit(
        "detection_complete",
        {
          "imageUrl": url_for("static", filename="figures/logo.png"),
          "error": "Detection timed out.",
        },
      )

  @socketio.on("send_agent_message")
  def handle_agent_message(  # pyright: ignore[reportUnusedFunction]
    data: Dict[str, Any],
  ):
    user_message = data.get("message", "")
    if user_message:
      socketio.start_background_task(
        target=eva_agent.handle_message, user_message=user_message
      )

  # ----------------------------------------------------------------------------
  # LLM backend switch
  # ----------------------------------------------------------------------------

  @socketio.on("set_llm_backend")
  def handle_set_llm_backend(data: Dict[str, Any]):
    """
    data: {
      "backend": "deepseek" | "local_llama",
      "local_url": "http://127.0.0.1:8080/v1/chat/completions",
    }
    """

    backend = data.get("backend", "").strip().lower()
    local_url = data.get("local_url")
    try:
      eva_agent.set_backend(backend, local_url=local_url)
      emit(
        "llm_backend_changed",
        {
          "backend": eva_agent.backend,
          "local_url": getattr(eva_agent, "local_llm_url", ""),
        },
        broadcast=True,
      )
      logger.info(f"LLM backend switched to {eva_agent.backend}")
    except Exception as e:
      logger.error(f"Failed to switch LLM backend: {e}")
      emit(
        "agent_response",
        {"response": f"切換 LLM 後端失敗: {e}"},
      )
