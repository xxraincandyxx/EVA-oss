import React, { useMemo, useState } from 'react';

import LiveMonitor from './VideoUtils';
import useRobotStore from '../store/useRobotStore';
import ArmSimulation from '../components/three/ArmSimulation';

// --- Main Component ---

const initialState = {
  x: '0.0',
  y: '0.0',
  z: '0.0',
  a: '0.0',
  b: '0.0',
  c: '0.0',
  axis1: '0.0',
  axis2: '0.0',
  axis3: '0.0',
  axis4: '0.0',
  axis5: '0.0',
  axis6: '0.0',
  cartDuration: '2.0',
  cartRotation: '0.0',
  axisDuration: '2.0',
  axisRotation: '0.0',
};

const Control = () => {
  const status = useRobotStore((state) => state.status);
  const schedule = useRobotStore((state) => state.schedule);
  const emit = useRobotStore((state) => state.emit);
  const isLiveSimActive = useRobotStore((state) => state.isLiveSimActive);

  const [inputs, setInputs] = useState(initialState);

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setInputs((prev) => ({ ...prev, [id]: value }));
  };

  const getParsedInputs = (type) => ({
    x: parseFloat(inputs.x),
    y: parseFloat(inputs.y),
    z: parseFloat(inputs.z),
    a: parseFloat(inputs.a),
    b: parseFloat(inputs.b),
    c: parseFloat(inputs.c),
    duration: parseFloat(
      type === 'cart' ? inputs.cartDuration : inputs.axisDuration
    ),
    rotation: parseFloat(
      type === 'cart' ? inputs.cartRotation : inputs.axisRotation
    ),
    axis1: parseFloat(inputs.axis1),
    axis2: parseFloat(inputs.axis2),
    axis3: parseFloat(inputs.axis3),
    axis4: parseFloat(inputs.axis4),
    axis5: parseFloat(inputs.axis5),
    axis6: parseFloat(inputs.axis6),
  });

  const handleCartesianAction = (action) =>
    emit(action, getParsedInputs('cart'));
  const handleAxesAction = (action) => emit(action, getParsedInputs('axis'));

  const handleRestore = (type) => {
    const fieldsToReset =
      type === 'cartesian'
        ? {
            x: '0.0',
            y: '0.0',
            z: '0.0',
            a: '0.0',
            b: '0.0',
            c: '0.0',
            cartDuration: '2.0',
            cartRotation: '0.0',
          }
        : {
            axis1: '0.0',
            axis2: '0.0',
            axis3: '0.0',
            axis4: '0.0',
            axis5: '0.0',
            axis6: '0.0',
            axisDuration: '2.0',
            axisRotation: '0.0',
          };
    setInputs((prev) => ({ ...prev, ...fieldsToReset }));
  };

  const schedulePreview = useMemo(() => {
    if (!schedule || schedule.length === 0) {
      return <p className="empty-schedule">No commands scheduled yet.</p>;
    }
    return schedule.map((item, index) => {
      let details = `Action: ${item.action}`;
      if (item.action === 'CARTESIAN')
        details = `X:${item.X}, Y:${item.Y}, Z:${item.Z}`;
      else if (item.action === 'AXES')
        details = `A1:${item.Axis1}, A2:${item.Axis2}, ...`;

      return (
        <div key={index} className="scheduled-item">
          <span className="index">{index + 1}</span>
          <span className="name">{item.name}</span>
          <span className="details">{details}</span>
        </div>
      );
    });
  }, [schedule]);

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Control</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                Control
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="bottom-data">
        <div className="orders">
          <div className="header">
            <i className="bx bx-shape-polygon"></i>
            <h3>3D Visualization</h3>
            <div className="header-controls">
              <label
                htmlFor="live-sim-toggle-control"
                className="live-sim-label"
              >
                Live Sim
              </label>
              <input
                type="checkbox"
                id="live-sim-toggle-control"
                className="lcs_check"
                checked={!!isLiveSimActive}
                onChange={(e) =>
                  emit('toggle_live_simulation', { active: e.target.checked })
                }
              />
              <label htmlFor="live-sim-toggle-control"></label>
            </div>
          </div>
          <ArmSimulation />
          <div className="mini-panel-container">
            <button
              className="refresh-arm-sim-btn"
              onClick={() => emit('refresh_arm_sim')}
            >
              ↺ REFRESH
            </button>
            <button
              className="emergent-stop-btn"
              onClick={() => emit('emergent_stop')}
            >
              ⚠ EMERGENT STOP
            </button>
          </div>
          <div className="mini-panel-container">
            <button
              className="restore-arm-sim-btn"
              onClick={() => emit('restore_arm_sim')}
            >
              RESTORE
            </button>
          </div>
        </div>

        <div className="reminders">
          <LiveMonitor
            title="Real-time Monitor"
            endpoint="/realtime_monitor"
            icon="bx bx-video"
          />

          <LiveMonitor
            title="Auxiliary Stacked Detection"
            endpoint="/auxiliary_stacked_detection_monitor"
            icon="bx bx-camera-movie"
            style={{ marginTop: '1rem' }}
          />
        </div>
      </div>

      <div className="bottom-data">
        <div className="orders">
          <div className="header">
            <i className="bx bx-stats"></i>
            <h3>System Status</h3>
          </div>
          <div className="status-grid">
            <div className="status-item">
              <span className="status-label">Position</span>
              <span className="status-value">{`X:${status.position[0].toFixed(
                4
              )} Y:${status.position[1].toFixed(
                4
              )} Z:${status.position[2].toFixed(4)}`}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Orientation</span>
              <span className="status-value">{`A:${status.orientation[0].toFixed(
                2
              )} B:${status.orientation[1].toFixed(
                2
              )} C:${status.orientation[2].toFixed(2)}`}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Axis Angles</span>
              <span className="status-value">
                {status.thetas.map((a) => a.toFixed(2)).join(', ')}
              </span>
            </div>
          </div>
        </div>
        <div className="reminders">
          <div className="header">
            <i className="bx bx-list-check"></i>
            <h3>Scheduled Commands</h3>
          </div>
          <div className="cache-container" id="controlPageSchedulePreview">
            {schedulePreview}
          </div>
        </div>
      </div>

      <div className="bottom-data">
        <div className="orders">
          <div className="header">
            <i className="bx bx-cog"></i>
            <h3>Control Interface</h3>
          </div>
          <div className="control-group">
            <h4>Cartesian Control</h4>
            <div className="input-group">
              {['x', 'y', 'z', 'a', 'b', 'c'].map((id) => (
                <div className="input-container" key={id}>
                  <input
                    type="number"
                    id={id}
                    step="0.1"
                    value={inputs[id]}
                    onChange={handleInputChange}
                    required
                  />
                  <label htmlFor={id}>{id.toUpperCase()}</label>
                </div>
              ))}
              <div className="duration-container">
                <input
                  type="number"
                  id="cartDuration"
                  step="0.1"
                  value={inputs.cartDuration}
                  onChange={handleInputChange}
                  required
                />
                <label htmlFor="cartDuration">Duration</label>
              </div>
              <div className="rotation-container">
                <input
                  type="number"
                  id="cartRotation"
                  step="0.1"
                  value={inputs.cartRotation}
                  onChange={handleInputChange}
                  required
                />
                <label htmlFor="cartRotation">Rotation</label>
              </div>
            </div>
            <div className="button-container">
              <button
                className="control-btn"
                onClick={() => handleCartesianAction('simulate_with_cartesian')}
              >
                SIMULATE
              </button>
              <button
                className="control-btn"
                onClick={() => handleCartesianAction('emit_with_cartesian')}
              >
                EMIT
              </button>
              <button
                className="caching-btn"
                onClick={() => handleCartesianAction('schedule_with_cartesian')}
              >
                ADD TO SCHEDULE
              </button>
              <button
                className="undo-btn"
                onClick={() => emit('undo_last_move')}
              >
                UNDO
              </button>
              <button
                className="restore-btn"
                onClick={() => handleRestore('cartesian')}
              >
                <i className="bx bx-reset"></i>
              </button>
            </div>
          </div>

          <div className="control-group">
            <h4>Axis Control</h4>
            <div className="input-group">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div className="input-container" key={`axis${i}`}>
                  <input
                    type="number"
                    id={`axis${i}`}
                    step="0.1"
                    value={inputs[`axis${i}`]}
                    onChange={handleInputChange}
                    required
                  />
                  <label htmlFor={`axis${i}`}>{`Axis ${i}`}</label>
                </div>
              ))}
              <div className="duration-container">
                <input
                  type="number"
                  id="axisDuration"
                  step="0.1"
                  value={inputs.axisDuration}
                  onChange={handleInputChange}
                  required
                />
                <label htmlFor="axisDuration">Duration</label>
              </div>
              <div className="rotation-container">
                <input
                  type="number"
                  id="axisRotation"
                  step="0.1"
                  value={inputs.axisRotation}
                  onChange={handleInputChange}
                  required
                />
                <label htmlFor="axisRotation">Rotation</label>
              </div>
            </div>
            <div className="button-container">
              <button
                className="control-btn"
                onClick={() => handleAxesAction('simulate_with_axes')}
              >
                SIMULATE
              </button>
              <button
                className="control-btn"
                onClick={() => handleAxesAction('emit_with_axes')}
              >
                EMIT
              </button>
              <button
                className="caching-btn"
                onClick={() => handleAxesAction('schedule_with_axes')}
              >
                ADD TO SCHEDULE
              </button>
              <button
                className="undo-btn"
                onClick={() => emit('undo_last_move')}
              >
                UNDO
              </button>
              <button
                className="restore-btn"
                onClick={() => handleRestore('axes')}
              >
                <i className="bx bx-reset"></i>
              </button>
            </div>
          </div>
        </div>

        <div className="reminders">
          <div className="header">
            <i className="bx bx-wind"></i>
            <h3>Air Pump</h3>
          </div>
          <div className="button-container-group">
            <div className="button-container">
              <button
                className="pump-open-btn"
                onClick={() => emit('pump_attach')}
              >
                ATTACH
              </button>
              <button
                className="pump-close-btn"
                onClick={() => emit('pump_detach')}
              >
                DETACH
              </button>
              <button
                className="pump-shutdown-btn"
                onClick={() => emit('pump_shutdown')}
              >
                SHUTDOWN
              </button>
            </div>
            <div className="button-container">
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_pump_attach')}
              >
                SCHEDULE ATTACH
              </button>
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_pump_detach')}
              >
                SCHEDULE DETACH
              </button>
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_pump_shutdown')}
              >
                SCHEDULE SHUTDOWN
              </button>
            </div>
          </div>

          <div className="header">
            <i className="bx bx-rotate-right"></i>
            <h3>Rotation Platform</h3>
          </div>
          <div className="button-container-group">
            <div className="button-container">
              <button
                className="pump-open-btn"
                onClick={() => emit('rot_clamp')}
              >
                CLAMP
              </button>
              <button
                className="pump-close-btn"
                onClick={() => emit('rot_release')}
              >
                RELEASE
              </button>
              <button
                className="pump-shutdown-btn"
                onClick={() => emit('rot_rotate')}
              >
                ROTATE
              </button>
            </div>
            <div className="button-container">
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_rot_clamp')}
              >
                SCHEDULE CLAMP
              </button>
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_rot_release')}
              >
                SCHEDULE RELEASE
              </button>
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_rot_rotate')}
              >
                SCHEDULE ROTATE
              </button>
            </div>
          </div>

          <div className="header">
            <i className="bx bx-command"></i>
            <h3>Functional</h3>
          </div>
          <div className="button-container-group">
            <div className="button-container">
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_suspend')}
              >
                SCHEDULE SUSPEND
              </button>
              <button
                className="schedule-btn"
                onClick={() => emit('schedule_capture')}
              >
                SCHEDULE CAPTURE
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Control;
