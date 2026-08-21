# robo_plot.py

from functools import partial
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ..lib import dynamo_
from ..utils import get_logger

logger = get_logger(__name__)


def matplot(points):
  points = np.array(points)

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")

  ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c="black", label="Joints")

  num_segments = len(points) - 1
  colors = plt.cm.rainbow(np.linspace(0, 1, num_segments))  # Rainbow color map

  for i in range(num_segments):
    ax.plot(
      points[i : i + 2, 0],
      points[i : i + 2, 1],
      points[i : i + 2, 2],
      color=colors[i],
      linewidth=3,
      label=f"Segment {i + 1}",
    )

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.set_title("Robotic Arm Simulation")
  ax.legend()

  plt.show()


def plotly(points):
  # Scale up to cm
  for i in range(len(points)):
    for j in range(len(points[i])):
      points[i][j] = points[i][j] * 100
  points = np.array(points)

  fig = go.Figure()
  colors = px.colors.qualitative.Plotly

  # Add segments with improved styling
  for i in range(len(points) - 1):
    fig.add_trace(
      go.Scatter3d(
        x=points[i : i + 2, 0],
        y=points[i : i + 2, 1],
        z=points[i : i + 2, 2],
        mode="lines+markers+text",
        line=dict(width=8, color=colors[i]),
        marker=dict(
          size=6,
          color=colors[i],
          symbol="diamond" if i == 0 else "circle",
          line=dict(width=2, color="white"),
        ),
        name=f"Segment {i + 1}",
        text=[f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in points[i : i + 2]],
        textposition="top center",
        hoverinfo="text+name",
        hovertemplate="<b>%{name}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
      )
    )

  # Add end-effector annotation
  fig.add_trace(
    go.Scatter3d(
      x=[points[-1, 0]],
      y=[points[-1, 1]],
      z=[points[-1, 2]],
      mode="markers+text",
      marker=dict(size=7, color="red", symbol="x", line=dict(width=2)),
      name="End Effector",
      text=["End Effector"],
      textposition="bottom center",
    )
  )

  # Equal axis scaling and professional styling
  fig.update_layout(
    scene=dict(
      aspectmode="cube",
      xaxis=dict(
        title="X Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      yaxis=dict(
        title="Y Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      zaxis=dict(
        title="Z Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    ),
    title=dict(
      text="3D Robotic Arm Visualization",
      font=dict(size=24, family="Arial, sans-serif"),
      x=0.5,
    ),
    paper_bgcolor="rgba(25, 25, 35, 1)",
    font=dict(color="white"),
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
      font=dict(size=12),
    ),
  )

  # Correct coordinate system indicator (3 separate axes)
  for axis, color in zip([[2, 0, 0], [0, 2, 0], [0, 0, 2]], ["red", "green", "blue"]):
    fig.add_trace(
      go.Scatter3d(
        x=[0, axis[0]],
        y=[0, axis[1]],
        z=[0, axis[2]],
        mode="lines",
        line=dict(width=4, color=color),
        showlegend=False,
      )
    )

  fig.show()


def animate_arm(start_points, end_points, num_frames=50, fps=30):
  # convert to np.ndarray, then upscale from m to cm
  if isinstance(start_points, List):
    start_points = np.array(start_points) * 100
  if isinstance(end_points, List):
    end_points = np.array(end_points) * 100

  t = np.linspace(0, 1, num_frames)
  interpolated_frames = [start_points + (end_points - start_points) * ti for ti in t]

  # Create figure
  fig = go.Figure()

  # Create initial trace
  initial_points = interpolated_frames[0]
  colors = px.colors.qualitative.Plotly

  # Add segments
  for i in range(len(initial_points) - 1):
    fig.add_trace(
      go.Scatter3d(
        x=initial_points[i : i + 2, 0],
        y=initial_points[i : i + 2, 1],
        z=initial_points[i : i + 2, 2],
        mode="lines+markers",
        line=dict(width=6, color=colors[i]),
        marker=dict(size=8, color=colors[i]),
        name=f"Segment {i + 1}",
      )
    )

  # Create animation frames
  frames = [
    go.Frame(
      data=[
        go.Scatter3d(
          x=frame[i : i + 2, 0], y=frame[i : i + 2, 1], z=frame[i : i + 2, 2]
        )
        for i in range(len(frame) - 1)
      ],
      name=str(k),
    )
    for k, frame in enumerate(interpolated_frames)
  ]
  # logger.debug(f"animate_arm() - frames:\n{frames[:2]}")

  # Configure animation
  fig.frames = frames
  fig.update_layout(
    title="Robotic Arm Animation",
    scene=dict(aspectmode="cube"),
    updatemenus=[
      dict(
        type="buttons",
        buttons=[
          dict(
            label="Play",
            method="animate",
            args=[
              None,
              {"frame": {"duration": 1000 / fps}, "fromcurrent": True},
            ],
          )
        ],
      )
    ],
  )

  return fig


def trace_tracker(
  states_lst: List[List[List[float]]],
  init_thetas: Union[List, np.ndarray] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  _get_states: Optional[Union["dynamo_.InstanceStreamer.get_states", Callable]] = None,
) -> go.Figure:
  # TODO: complete the following docstring
  """Arm Movements States Checkpoints Tracker

  Args:
      states_lst (List[List[float]]): ...

  Returns:
      fig (_type_): ...
  """

  # input check
  logger.debug(f"trace_tracker() - states_lst: {states_lst}")

  if isinstance(init_thetas, np.ndarray):
    init_thetas = init_thetas.tolist()

  # upscale m to cm
  for i in range(len(states_lst)):
    if isinstance(states_lst[i], List):
      states_lst[i] = np.array(states_lst[i]) * 100

  _partial_get_states = partial(
    _get_states,
    input_thetas=dynamo_.Thetas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # placeholder
    using_pylist=True,
    verbose=False,
  )

  # calculating init state
  states_lst.insert(
    0,
    np.array([[0.0, 0.0, 0.0]] + _partial_get_states(input_thetas_pylist=init_thetas))
    * 100,
  )
  logger.debug(f"trace_tracker() - states_lst: {len(states_lst)}\n{states_lst[:2]}")

  fig = go.Figure()
  colors = px.colors.qualitative.Plotly

  for i in range(len(states_lst)):
    # Add segments with improved styling
    for j in range(len(states_lst[i]) - 1):
      fig.add_trace(
        go.Scatter3d(
          x=states_lst[i][j : j + 2, 0],
          y=states_lst[i][j : j + 2, 1],
          z=states_lst[i][j : j + 2, 2],
          mode="lines+markers+text",
          line=dict(width=8, color=colors[j]),
          marker=dict(
            size=6,
            color=colors[j],
            symbol="diamond" if j == 0 else "circle",
            line=dict(width=2, color="white"),
          ),
          name=f"Arm {j + 1}",
          text=[f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in states_lst[i][j : j + 2]],
          textposition="top center",
          hoverinfo="text+name",
          hovertemplate="<b>%{name}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
          showlegend=False if i != 0 else True,
        )
      )

    # Add end-effector annotation
    fig.add_trace(
      go.Scatter3d(
        x=states_lst[i][-1:, 0],
        y=states_lst[i][-1:, 1],
        z=states_lst[i][-1:, 2],
        mode="markers+text",
        marker=dict(size=7, color="red", symbol="x", line=dict(width=2)),
        name="End Effector",
        text=["End Effector"],
        textposition="bottom center",
        showlegend=False if i != 0 else True,
      )
    )

  # Equal axis scaling and professional styling
  fig.update_layout(
    scene=dict(
      aspectmode="cube",
      xaxis=dict(
        title="X Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      yaxis=dict(
        title="Y Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      zaxis=dict(
        title="Z Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    ),
    title=dict(
      text="3D Robotic Arm Visualization",
      font=dict(size=24, family="Arial, sans-serif"),
      x=0.5,
    ),
    paper_bgcolor="rgba(25, 25, 35, 1)",
    font=dict(color="white"),
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
      font=dict(size=12),
    ),
  )

  return fig


def trace_capture_old(
  angles_lst: List[List[float]],
  duration_lst: Optional[List[float]] = [],
  _get_states: Optional[Union["dynamo_.InstanceStreamer.get_states", Callable]] = None,
  fps: int = 30,
) -> go.Figure:
  # TODO: complete the following docstring
  """Arm Movement Trance Capture

  Args:
      angles_lst (List[List[float]]): ...
      duration_lst (Optional[List[float]], optional): ... Defaults to [].
      _get_states (Optional[&quot;dynamo_.InstanceStreamer.get_states&quot;], optional): ... Unit seconds. Defaults to None.
      fps (int, optional): _description_. Defaults to 30.

  Returns:
      fig (plotly.graph_objects.Figure()): ...
  """

  # input check
  logger.debug(f"trace_capture() - angles_lst  : {angles_lst}")
  logger.debug(f"trace_capture() - duration_lst: {duration_lst}")

  if duration_lst is None or len(duration_lst) == 0:
    duration_lst = [3.6] * len(angles_lst)

  assert len(angles_lst) == len(duration_lst), (
    f"Size of angles_lst ({len(angles_lst)}) is supposed to be the same with duration_lst ({len(duration_lst)})."
  )

  for i in range(len(angles_lst)):
    if isinstance(angles_lst[i], List):
      angles_lst[i] = np.array(angles_lst[i])

  _partial_get_states = partial(
    _get_states,
    input_thetas=dynamo_.Thetas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # placeholder
    using_pylist=True,
    verbose=False,
  )

  init_state = (
    np.array(
      [[0.0, 0.0, 0.0]]
      + _partial_get_states(input_thetas_pylist=angles_lst[0].tolist())
    )
    * 100
  )

  fig = go.Figure()
  colors = px.colors.qualitative.Plotly

  # === Plot the Robotic Arm === #
  for j in range(len(init_state) - 1):
    fig.add_trace(
      go.Scatter3d(
        x=init_state[j : j + 2, 0],
        y=init_state[j : j + 2, 1],
        z=init_state[j : j + 2, 2],
        mode="lines+markers+text",
        line=dict(width=8, color=colors[j]),
        marker=dict(
          size=6,
          color=colors[j],
          symbol="diamond" if j == 0 else "circle",
          line=dict(width=2, color="white"),
        ),
        name=f"Arm {j + 1}",
        text=[f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in init_state[j : j + 2]],
        textposition="top center",
        hoverinfo="text+name",
        hovertemplate="<b>%{name}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
      )
    )

  # Add end-effector annotation
  fig.add_trace(
    go.Scatter3d(
      x=init_state[-1:, 0],
      y=init_state[-1:, 1],
      z=init_state[-1:, 2],
      mode="markers+text",
      marker=dict(size=7, color="red", symbol="x", line=dict(width=2)),
      name="End Effector",
      text=["End Effector"],
      textposition="bottom center",
    )
  )

  # === Create Animation Frames === #
  frames = []  # declare frames
  for i in range(len(angles_lst) - 1):
    t = np.linspace(0, 1, round(duration_lst[i] * fps))
    _inter_thetas = [
      angles_lst[i] + (angles_lst[i + 1] - angles_lst[i]) * ti for ti in t
    ]
    logger.debug(
      f"trace_capture() - _inter_thetas: {len(_inter_thetas)}\n{_inter_thetas[-2:]}"
    )

    _inter_frames = [
      np.array(
        [[0.0, 0.0, 0.0]] + _partial_get_states(input_thetas_pylist=__thetas.tolist())
      )
      * 100
      for __thetas in _inter_thetas
    ]
    logger.debug(
      f"trace_capture() - _inter_frames: {len(_inter_thetas)}\n{_inter_frames[-2:]}"
    )

    _frames = [
      go.Frame(
        data=[
          go.Scatter3d(  # for segments
            x=_frame[i : i + 2, 0],
            y=_frame[i : i + 2, 1],
            z=_frame[i : i + 2, 2],
          )
          for i in range(len(_frame) - 1)
        ]
        + [
          go.Scatter3d(  # for end-effector
            x=_frame[-1:, 0],
            y=_frame[-1:, 1],
            z=_frame[-1:, 2],
          )
        ],
        name=f"frame_{i}_{k}",
      )
      for k, _frame in enumerate(_inter_frames)
    ]
    frames.extend(_frames)
  fig.frames = frames
  logger.debug(f"trace_capture() - frames:\n{frames[:2]}")

  # Equal axis scaling and professional styling
  fig.update_layout(
    scene=dict(
      aspectmode="cube",
      xaxis=dict(
        title="X Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      yaxis=dict(
        title="Y Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      zaxis=dict(
        title="Z Axis",
        gridcolor="rgba(150, 150, 150, 0.5)",
        backgroundcolor="rgba(20, 24, 35, 0.1)",
        title_font=dict(size=16),
      ),
      camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    ),
    updatemenus=[
      dict(
        type="buttons",
        buttons=[
          dict(
            label="Play",
            method="animate",
            args=[
              None,
              {"frame": {"duration": 1000 / fps}, "fromcurrent": True},
            ],
          )
        ],
      )
    ],
    title=dict(
      text="3D Robotic Arm Visualization",
      font=dict(size=24, family="Arial, sans-serif"),
      x=0.5,
    ),
    paper_bgcolor="rgba(25, 25, 35, 1)",
    font=dict(color="white"),
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
      font=dict(size=12),
    ),
  )

  return fig


def trace_capture(
  angles_lst: List[List[float]],
  duration_lst: Optional[List[float]] = [],
  _get_states: Optional[Union["dynamo_.InstanceStreamer.get_states", Callable]] = None,
  fps: int = 30,
  ckpt_sleep_duration: float = 2.0,  # s
) -> go.Figure:
  # TODO: complete the following docstring
  """Arm Movement Trance Capture

  Args:
      angles_lst (List[List[float]]): ...
      duration_lst (Optional[List[float]], optional): ... Unit seconds. Defaults to [].
      _get_states (Optional[&quot;dynamo_.InstanceStreamer.get_states&quot;], optional): ... Unit seconds. Defaults to None.
      fps (int, optional): _description_. Defaults to 30.
      ckpt_sleep_duration: Sleep time duration for checkpoints. Unit seconds. Defaults to 2.0s.

  Returns:
      fig (go.Figure): plotly go.Figure object.
  """

  # input check
  logger.debug(f"trace_capture() - angles_lst  : {angles_lst}")
  logger.debug(f"trace_capture() - duration_lst: {duration_lst}")

  if duration_lst is None or len(duration_lst) == 0:
    duration_lst = [2.0] * len(angles_lst)

  assert len(angles_lst) == len(duration_lst), (
    f"Size of angles_lst ({len(angles_lst)}) is supposed to be the same with duration_lst ({len(duration_lst)})."
  )

  for i in range(len(angles_lst)):
    if isinstance(angles_lst[i], List):
      angles_lst[i] = np.array(angles_lst[i])

  _partial_get_states = partial(
    _get_states,
    input_thetas=dynamo_.Thetas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # placeholder
    using_pylist=True,
    verbose=False,
  )

  init_state = (
    np.array(
      [[0.0, 0.0, 0.0]]
      + _partial_get_states(input_thetas_pylist=angles_lst[0].tolist())
    )
    * 100
  )

  fig = go.Figure()
  colors = px.colors.qualitative.Plotly

  # === Plot the Robotic Arm === #
  for j in range(len(init_state) - 1):
    fig.add_trace(
      go.Scatter3d(
        x=init_state[j : j + 2, 0],
        y=init_state[j : j + 2, 1],
        z=init_state[j : j + 2, 2],
        mode="lines+markers",
        line=dict(width=8, color=colors[j]),
        marker=dict(
          size=6,
          color=colors[j],
          symbol="diamond" if j == 0 else "circle",
          line=dict(width=2, color="white"),
        ),
        name=f"Arm {j + 1}",
      )
    )

  # Add end-effector annotation
  fig.add_trace(
    go.Scatter3d(
      x=init_state[-1:, 0],
      y=init_state[-1:, 1],
      z=init_state[-1:, 2],
      mode="markers",
      marker=dict(size=7, color="red", symbol="x", line=dict(width=2)),
      name="End Effector",
    )
  )

  # === Create Animation Frames === #
  frames = []  # declare frames
  for i in range(len(angles_lst) - 1):
    t = np.linspace(0, 1, round(duration_lst[i] * fps))
    _inter_thetas = [
      angles_lst[i] + (angles_lst[i + 1] - angles_lst[i]) * ti for ti in t
    ]
    logger.debug(
      f"trace_capture() - _inter_thetas: {len(_inter_thetas)}\n{_inter_thetas[-2:]}"
    )

    _inter_frames = [
      np.array(
        [[0.0, 0.0, 0.0]] + _partial_get_states(input_thetas_pylist=__thetas.tolist())
      )
      * 100
      for __thetas in _inter_thetas
    ]
    logger.debug(
      f"trace_capture() - _inter_frames: {len(_inter_thetas)}\n{_inter_frames[-2:]}"
    )

    _frames = [
      go.Frame(
        data=[
          go.Scatter3d(  # for segments
            x=_frame[i : i + 2, 0],
            y=_frame[i : i + 2, 1],
            z=_frame[i : i + 2, 2],
            mode="lines+markers",
            line=dict(width=8, color=colors[i]),
            marker=dict(
              size=6,
              color=colors[j],
              symbol="diamond" if i == 0 else "circle",
              line=dict(width=2, color="white"),
            ),
            name=f"Arm {i + 1}",
          )
          for i in range(len(_frame) - 1)
        ]
        + [
          go.Scatter3d(  # for end-effector
            x=_frame[-1:, 0],
            y=_frame[-1:, 1],
            z=_frame[-1:, 2],
            mode="markers",
            marker=dict(size=7, color="red", symbol="x", line=dict(width=2)),
            name="End Effector",
          )
        ],
        name=f"frame_{i}_{k}",
      )
      for k, _frame in enumerate(_inter_frames)
    ]
    frames.extend(_frames)
  fig.frames = frames
  # logger.debug(f"trace_capture() - frames:\n{frames[:2]}")

  # Calculate axis ranges with 10% padding
  # x = init_state[:, 0]
  # y = init_state[:, 1]
  # z = init_state[:, 2]

  def get_axis_range(axis_data):
    min_val = np.min(axis_data)
    max_val = np.max(axis_data)
    span = max_val - min_val
    pad = span * 0.1 if span != 0 else 10  # Fallback padding if no span
    return [min_val - pad, max_val + pad]

  # x_range = get_axis_range(x)
  # y_range = get_axis_range(y)
  # z_range = get_axis_range(z)

  # Using preset range
  x_range = [-84, 84]
  y_range = [-84, 84]
  z_range = [0, 64]

  # === Update Layout with Professional Styling === #
  fig.update_layout(
    scene=dict(
      aspectmode="cube",
      xaxis=dict(
        title="X Axis (cm)",
        range=x_range,
        gridcolor="rgba(100, 100, 100, 0.2)",
        backgroundcolor="rgba(15, 20, 30, 0.05)",
        title_font=dict(size=14, color="#e0e0e0"),
        tickfont=dict(color="#b0b0b0"),
        zerolinecolor="rgba(100, 100, 100, 0.5)",
      ),
      yaxis=dict(
        title="Y Axis (cm)",
        range=y_range,
        gridcolor="rgba(100, 100, 100, 0.2)",
        backgroundcolor="rgba(15, 20, 30, 0.05)",
        title_font=dict(size=14, color="#e0e0e0"),
        tickfont=dict(color="#b0b0b0"),
        zerolinecolor="rgba(100, 100, 100, 0.5)",
      ),
      zaxis=dict(
        title="Z Axis (cm)",
        range=z_range,
        gridcolor="rgba(100, 100, 100, 0.2)",
        backgroundcolor="rgba(15, 20, 30, 0.05)",
        title_font=dict(size=14, color="#e0e0e0"),
        tickfont=dict(color="#b0b0b0"),
        zerolinecolor="rgba(100, 100, 100, 0.5)",
      ),
      camera=dict(
        eye=dict(x=1.6, y=-1.6, z=1.2),
        up=dict(x=0, y=0, z=1),
        projection=dict(type="orthographic"),
      ),
    ),
    updatemenus=[
      dict(
        type="buttons",
        showactive=False,
        x=0.05,
        y=0.05,
        xanchor="left",
        yanchor="bottom",
        bgcolor="#4CAF50",  # Main button color
        borderwidth=2,
        bordercolor="#333333",
        font=dict(color="white", size=12),
        buttons=[
          dict(
            label="▶ Play",
            method="animate",
            args=[
              None,
              {
                "frame": {"duration": 1000 / fps, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 50},
              },
            ],
          )
        ],
      )
    ],
    title=dict(
      text="Robotic Arm Motion Visualization",
      font=dict(size=24, family="Arial", color="white"),
      x=0.5,
      y=0.95,
    ),
    paper_bgcolor="rgba(10, 10, 20, 1)",
    plot_bgcolor="rgba(15, 20, 30, 0.8)",
    font=dict(color="white"),
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=0.98,
      bgcolor="rgba(0, 0, 0, 0.4)",
      font=dict(size=12),
    ),
    margin=dict(t=100, b=50, l=50, r=50),
  )

  return fig


# robo_plot.py ends here
