import React, { useCallback, useEffect, useRef, useState } from 'react';

import LiveMonitor from './VideoUtils';
import useRobotStore from '../store/useRobotStore';
import ArmSimulation from '../components/three/ArmSimulation';

const keyMap = {
  KeyW: { axis: 'y', direction: 1 },
  KeyS: { axis: 'y', direction: -1 },
  KeyA: { axis: 'x', direction: -1 },
  KeyD: { axis: 'x', direction: 1 },
  KeyQ: { axis: 'z', direction: 1 },
  KeyE: { axis: 'z', direction: -1 },
  ArrowUp: { axis: 'b', direction: 1 },
  ArrowDown: { axis: 'b', direction: -1 },
  ArrowLeft: { axis: 'a', direction: -1 },
  ArrowRight: { axis: 'a', direction: 1 },
  PageUp: { axis: 'c', direction: 1 },
  PageDown: { axis: 'c', direction: -1 },
};

const Fpv = () => {
  const status = useRobotStore((state) => state.status);
  const isTracking = useRobotStore((state) => state.isTracking);
  const emit = useRobotStore((state) => state.emit);

  const [isFpvActive, setIsFpvActive] = useState(false);
  const [config, setConfig] = useState({
    samplingRate: 100,
    posIncrement: 0.005,
    orientIncrement: 0.5,
    duration: 0.1,
  });

  const moveState = useRef({ x: 0, y: 0, z: 0, a: 0, b: 0, c: 0 });
  const moveInterval = useRef(null);
  const activeKeys = useRef(new Set());

  const startSendingMoves = useCallback(() => {
    if (moveInterval.current) return;
    moveInterval.current = setInterval(() => {
      if (Object.values(moveState.current).some((v) => v !== 0)) {
        emit('fpv_move', { moveState: moveState.current, config });
      }
    }, config.samplingRate);
  }, [config, emit]);

  const stopSendingMoves = useCallback(() => {
    if (Object.values(moveState.current).every((v) => v === 0)) {
      clearInterval(moveInterval.current);
      moveInterval.current = null;
    }
  }, []);

  const updateMoveState = useCallback((axis, direction) => {
    if (!isFpvActive) return;
    moveState.current[axis] = direction;
    if (direction !== 0) {
      startSendingMoves();
    } else {
      stopSendingMoves();
    }
  }, [isFpvActive, startSendingMoves, stopSendingMoves]);

  const handleButtonAction = updateMoveState;

  const handleConfigChange = (e) => {
    const { id, value } = e.target;
    setConfig((prev) => ({
      ...prev,
      [id.replace('fpv-', '')]: parseFloat(value),
    }));
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (
        !isFpvActive ||
        activeKeys.current.has(e.code) ||
        e.target.tagName === 'INPUT'
      )
        return;
      const mapping = keyMap[e.code];
      if (mapping) {
        e.preventDefault();
        activeKeys.current.add(e.code);
        updateMoveState(mapping.axis, mapping.direction);
      }
    };
    const handleKeyUp = (e) => {
      if (!isFpvActive) return;
      const mapping = keyMap[e.code];
      if (mapping) {
        e.preventDefault();
        activeKeys.current.delete(e.code);
        updateMoveState(mapping.axis, 0);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
      clearInterval(moveInterval.current);
      moveInterval.current = null;
    };
  }, [isFpvActive, updateMoveState]);

  useEffect(() => {
    emit('request_tracking_status');
  }, [emit]);

  const toggleFpv = (active) => {
    setIsFpvActive(active);
    if (!active) {
      moveState.current = { x: 0, y: 0, z: 0, a: 0, b: 0, c: 0 };
      clearInterval(moveInterval.current);
      moveInterval.current = null;
    }
  };

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>FPV</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                FPV
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
            {/* Live Sim Toggle Here */}
          </div>
          <div id="plot-container-fpv" className="plot-container-shared">
            <ArmSimulation />
          </div>
          <div className="fpv-status-grid">
            <div className="fpv-status-row">
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
            </div>
            <div className="fpv-status-row">
              <div className="status-item">
                <span className="status-label">Axis Angles</span>
                <span className="status-value">
                  {status.thetas.map((a) => a.toFixed(2)).join(', ')}
                </span>
              </div>
            </div>
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
            <i className="bx bx-game"></i>
            <h3>FPV Control</h3>
          </div>
          <div className="fpv-session-controls">
            <button
              onClick={() => toggleFpv(true)}
              className="action-btn start-fpv"
              disabled={isFpvActive}
            >
              <i className="bx bx-play-circle"></i>
              <span>START FPV</span>
            </button>
            <button
              onClick={() => toggleFpv(false)}
              className="action-btn end-fpv"
              disabled={!isFpvActive}
            >
              <i className="bx bx-stop-circle"></i>
              <span>END FPV</span>
            </button>
          </div>
          <div className="fpv-panels">
            {!isFpvActive && (
              <div className="fpv-control-overlay">
                <span>FPV Inactive</span>
              </div>
            )}
            <div className="fpv-panel">
              <h4>Positional Control (W/A/S/D/Q/E)</h4>
              <div className="d-pad-container">
                <div className="d-pad">
                  {/* W Button (Y+) */}
                  <button
                    className="d-pad-btn up"
                    onMouseDown={() => handleButtonAction('y', 1)}
                    onMouseUp={() => handleButtonAction('y', 0)}
                    onMouseLeave={() => handleButtonAction('y', 0)}
                  >
                    W
                  </button>
                  {/* A Button (X-) */}
                  <button
                    className="d-pad-btn left"
                    onMouseDown={() => handleButtonAction('x', -1)}
                    onMouseUp={() => handleButtonAction('x', 0)}
                    onMouseLeave={() => handleButtonAction('x', 0)}
                  >
                    A
                  </button>
                  {/* S Button (Y-) */}
                  <button
                    className="d-pad-btn down"
                    onMouseDown={() => handleButtonAction('y', -1)}
                    onMouseUp={() => handleButtonAction('y', 0)}
                    onMouseLeave={() => handleButtonAction('y', 0)}
                  >
                    S
                  </button>
                  {/* D Button (X+) */}
                  <button
                    className="d-pad-btn right"
                    onMouseDown={() => handleButtonAction('x', 1)}
                    onMouseUp={() => handleButtonAction('x', 0)}
                    onMouseLeave={() => handleButtonAction('x', 0)}
                  >
                    D
                  </button>
                  <div className="d-pad-center"></div>
                </div>
                <div className="z-controls">
                  {/* Q Button (Z+) */}
                  <button
                    className="d-pad-btn z-up"
                    onMouseDown={() => handleButtonAction('z', 1)}
                    onMouseUp={() => handleButtonAction('z', 0)}
                    onMouseLeave={() => handleButtonAction('z', 0)}
                  >
                    Q
                  </button>
                  {/* E Button (Z-) */}
                  <button
                    className="d-pad-btn z-down"
                    onMouseDown={() => handleButtonAction('z', -1)}
                    onMouseUp={() => handleButtonAction('z', 0)}
                    onMouseLeave={() => handleButtonAction('z', 0)}
                  >
                    E
                  </button>
                </div>
              </div>
            </div>
            <div className="fpv-panel">
              <h4>Orientational Control (Arrows)</h4>
              <div className="d-pad-container">
                <div className="d-pad">
                  {/* UP Button (B+) */}
                  <button
                    className="d-pad-btn up"
                    onMouseDown={() => handleButtonAction('b', 1)}
                    onMouseUp={() => handleButtonAction('b', 0)}
                    onMouseLeave={() => handleButtonAction('b', 0)}
                  >
                    ↑
                  </button>
                  {/* LEFT Button (A-) */}
                  <button
                    className="d-pad-btn left"
                    onMouseDown={() => handleButtonAction('a', -1)}
                    onMouseUp={() => handleButtonAction('a', 0)}
                    onMouseLeave={() => handleButtonAction('a', 0)}
                  >
                    ←
                  </button>
                  {/* DOWN Button (B-) */}
                  <button
                    className="d-pad-btn down"
                    onMouseDown={() => handleButtonAction('b', -1)}
                    onMouseUp={() => handleButtonAction('b', 0)}
                    onMouseLeave={() => handleButtonAction('b', 0)}
                  >
                    ↓
                  </button>
                  {/* RIGHT Button (A+) */}
                  <button
                    className="d-pad-btn right"
                    onMouseDown={() => handleButtonAction('a', 1)}
                    onMouseUp={() => handleButtonAction('a', 0)}
                    onMouseLeave={() => handleButtonAction('a', 0)}
                  >
                    →
                  </button>
                  <div className="d-pad-center"></div>
                </div>
                <div className="z-controls">
                  {/* PgUp Button (C+) */}
                  <button
                    className="d-pad-btn z-up"
                    onMouseDown={() => handleButtonAction('c', 1)}
                    onMouseUp={() => handleButtonAction('c', 0)}
                    onMouseLeave={() => handleButtonAction('c', 0)}
                  >
                    PgUp
                  </button>
                  {/* PgDn Button (C-) */}
                  <button
                    className="d-pad-btn z-down"
                    onMouseDown={() => handleButtonAction('c', -1)}
                    onMouseUp={() => handleButtonAction('c', 0)}
                    onMouseLeave={() => handleButtonAction('c', 0)}
                  >
                    PgDn
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="reminders">
          <div className="header">
            <i className="bx bx-trip"></i>
            <h3>Tracking</h3>
          </div>
          <div className="tracking-panel">
            <div className="tracking-status-container">
              <h4>Tracking Status</h4>
              <p
                id="tracking-status"
                className={isTracking ? 'active' : 'inactive'}
              >
                {isTracking ? 'ACTIVE' : 'INACTIVE'}
              </p>
            </div>
            <button
              className="action-btn start-tracking"
              onClick={() => emit('start_tracking')}
              disabled={isTracking}
            >
              <i className="bx bx-play-circle"></i> START TRACKING
            </button>
            <button
              className="action-btn end-tracking"
              onClick={() => emit('end_tracking')}
              disabled={!isTracking}
            >
              <i className="bx bx-stop-circle"></i> END TRACKING
            </button>
            <p className="tracking-info">
              Recorded tracks can be managed on the 'Schedule' page.
            </p>
          </div>
        </div>
      </div>
      <div className="bottom-data">
        <div className="orders">
          <div className="control-group">
            <h4>FPV Configuration</h4>
            <div className="input-group">
              <div className="input-container">
                <input
                  type="number"
                  id="samplingRate"
                  value={config.samplingRate}
                  onChange={handleConfigChange}
                  required
                />
                <label htmlFor="samplingRate">Sampling Rate (ms)</label>
              </div>
              <div className="input-container">
                <input
                  type="number"
                  id="posIncrement"
                  step="0.001"
                  value={config.posIncrement}
                  onChange={handleConfigChange}
                  required
                />
                <label htmlFor="posIncrement">Pos Increment</label>
              </div>
              <div className="input-container">
                <input
                  type="number"
                  id="orientIncrement"
                  step="0.1"
                  value={config.orientIncrement}
                  onChange={handleConfigChange}
                  required
                />
                <label htmlFor="orientIncrement">Orient Increment</label>
              </div>
              <div className="input-container">
                <input
                  type="number"
                  id="duration"
                  step="0.1"
                  value={config.duration}
                  onChange={handleConfigChange}
                  required
                />
                <label htmlFor="duration">Move Duration (s)</label>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Fpv;
