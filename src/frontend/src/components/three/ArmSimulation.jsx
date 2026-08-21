import React, { Suspense, useEffect, useRef, useState } from 'react';

import { Canvas, useFrame } from '@react-three/fiber';
import { Box, Grid, OrbitControls, TransformControls } from '@react-three/drei';

import useRobotStore from '../../store/useRobotStore';
import RobotModel from './RobotModel';

const jointLimits = 174.5;
const clampJoint = (value) =>
  Math.max(-jointLimits, Math.min(jointLimits, Number(value)));
const sceneToRobot = ([x, y, z]) => [x, -z, y];

const Loader = () => (
  <Box position={[0, 0.5, 0]}>
    <boxGeometry args={[0.5, 0.5, 0.5]} />
    <meshBasicMaterial color="tomato" wireframe />
  </Box>
);

const EndEffectorTarget = ({ endEffectorRef, onChange, onCommit, resetToken }) => {
  const targetRef = useRef();
  const [targetObject, setTargetObject] = useState(null);
  const initialized = useRef(false);

  useEffect(() => {
    initialized.current = false;
  }, [resetToken]);

  useFrame(() => {
    if (!initialized.current && endEffectorRef.current && targetRef.current) {
      endEffectorRef.current.getWorldPosition(targetRef.current.position);
      onChange(sceneToRobot(targetRef.current.position.toArray()));
      initialized.current = true;
    }
  });

  const reportPosition = () => {
    if (targetRef.current) {
      onChange(sceneToRobot(targetRef.current.position.toArray()));
    }
  };

  return (
    <>
      {targetObject && (
        <TransformControls
          object={targetObject}
          mode="translate"
          size={0.65}
          onObjectChange={reportPosition}
          onMouseUp={() => {
            if (targetRef.current) {
              onCommit(sceneToRobot(targetRef.current.position.toArray()));
            }
          }}
        />
      )}
      <group
        ref={(object) => {
          targetRef.current = object;
          setTargetObject(object);
        }}
      >
        <mesh renderOrder={20}>
          <sphereGeometry args={[0.018, 24, 24]} />
          <meshBasicMaterial color="#2d74c8" depthTest={false} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={19}>
          <torusGeometry args={[0.032, 0.0025, 8, 36]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>
      </group>
    </>
  );
};

const InteractionPanel = ({
  mode,
  setMode,
  target,
  thetas,
  selectedJoint,
  setSelectedJoint,
  updateJoint,
  onPreview,
  onMove,
  onReset,
  isOnline,
  error,
}) => (
  <div className="arm-interaction-panel">
    <div className="arm-mode-switch" aria-label="Manipulator mode">
      <button
        type="button"
        className={mode === 'target' ? 'active' : ''}
        onClick={() => setMode('target')}
      >
        <i className="bx bx-move"></i>
        End effector
      </button>
      <button
        type="button"
        className={mode === 'joints' ? 'active' : ''}
        onClick={() => setMode('joints')}
      >
        <i className="bx bx-rotate-right"></i>
        Joints
      </button>
    </div>

    {mode === 'target' ? (
      <div className="arm-target-readout">
        {['X', 'Y', 'Z'].map((axis, index) => (
          <span key={axis}>
            <b>{axis}</b> {target[index].toFixed(3)} m
          </span>
        ))}
      </div>
    ) : (
      <div className="arm-joint-editor">
        <div className="arm-joint-tabs">
          {thetas.map((_, index) => (
            <button
              type="button"
              key={index}
              className={selectedJoint === index ? 'active' : ''}
              onClick={() => setSelectedJoint(index)}
              aria-label={`Select joint ${index + 1}`}
            >
              J{index + 1}
            </button>
          ))}
        </div>
        <label>
          <span>Joint {selectedJoint + 1}</span>
          <div className="arm-joint-value">
            <button
              type="button"
              onClick={() => updateJoint(selectedJoint, thetas[selectedJoint] - 5)}
              aria-label="Decrease joint angle"
            >
              <i className="bx bx-minus"></i>
            </button>
            <output>{thetas[selectedJoint].toFixed(1)} deg</output>
            <button
              type="button"
              onClick={() => updateJoint(selectedJoint, thetas[selectedJoint] + 5)}
              aria-label="Increase joint angle"
            >
              <i className="bx bx-plus"></i>
            </button>
          </div>
          <input
            type="range"
            min={-jointLimits}
            max={jointLimits}
            step="0.5"
            value={thetas[selectedJoint]}
            onChange={(event) => updateJoint(selectedJoint, event.target.value)}
          />
        </label>
      </div>
    )}

    {error && <p className="arm-interaction-error">{error}</p>}

    <div className="arm-interaction-actions">
      <button type="button" onClick={onReset} title="Reset preview">
        <i className="bx bx-reset"></i>
      </button>
      <button type="button" onClick={onPreview} disabled={!isOnline}>
        Preview
      </button>
      <button
        type="button"
        className="primary"
        onClick={onMove}
        disabled={!isOnline}
      >
        Move
      </button>
    </div>
  </div>
);

const ArmSimulation = ({ interactive = false }) => {
  const status = useRobotStore((state) => state.status);
  const isOnline = useRobotStore((state) => state.isOnline);
  const interactivePreview = useRobotStore((state) => state.interactivePreview);
  const interactiveError = useRobotStore((state) => state.interactiveError);
  const emit = useRobotStore((state) => state.emit);
  const clearInteractivePreview = useRobotStore(
    (state) => state.clearInteractivePreview
  );

  const [mode, setMode] = useState('target');
  const [selectedJoint, setSelectedJoint] = useState(0);
  const [draftThetas, setDraftThetas] = useState(status.thetas);
  const [target, setTarget] = useState(status.position);
  const [resetToken, setResetToken] = useState(0);
  const endEffectorRef = useRef();

  const previewThetas =
    interactivePreview?.thetas?.length === 6
      ? interactivePreview.thetas.map(clampJoint)
      : null;
  const displayedThetas = previewThetas ?? draftThetas;

  const updateJoint = (index, value) => {
    const currentThetas = displayedThetas;
    clearInteractivePreview();
    setDraftThetas(() =>
      currentThetas.map((theta, jointIndex) =>
        jointIndex === index ? clampJoint(value) : theta
      )
    );
  };

  const cartesianPayload = (position = target) => ({
    x: position[0],
    y: position[1],
    z: position[2],
    a: status.orientation[0],
    b: status.orientation[1],
    c: status.orientation[2],
    duration: 2,
    rotation: 0,
  });

  const axesPayload = () => ({
    ...Object.fromEntries(
      displayedThetas.map((theta, index) => [
        `axis${index + 1}`,
        theta - status.thetas[index],
      ])
    ),
    duration: 2,
    rotation: 0,
  });

  const preview = (position = target) => {
    if (mode === 'target') {
      emit('preview_interactive_cartesian', cartesianPayload(position));
    } else {
      emit('simulate_with_axes', axesPayload());
    }
  };

  const move = () => {
    const event = mode === 'target' ? 'emit_with_cartesian' : 'emit_with_axes';
    emit(event, mode === 'target' ? cartesianPayload() : axesPayload());
  };

  const reset = () => {
    setDraftThetas(status.thetas);
    setTarget(status.position);
    setResetToken((token) => token + 1);
    clearInteractivePreview();
  };

  return (
    <div className={`plot-container-shared${interactive ? ' is-interactive' : ''}`}>
      <Canvas camera={{ position: [1.3, 1, 2.2], fov: 15 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} />
        <axesHelper args={[1]} />

        <Suspense fallback={<Loader />}>
          <RobotModel
            displayThetas={interactive ? displayedThetas : undefined}
            interactive={interactive && mode === 'joints'}
            selectedJoint={selectedJoint}
            onJointSelect={setSelectedJoint}
            onJointChange={updateJoint}
            endEffectorRef={endEffectorRef}
          />
          {interactive && mode === 'target' && (
            <EndEffectorTarget
              endEffectorRef={endEffectorRef}
              onChange={setTarget}
              onCommit={(position) => {
                setTarget(position);
                if (isOnline) preview(position);
              }}
              resetToken={resetToken}
            />
          )}
        </Suspense>

        <Grid position={[0, 0, 0]} infiniteGrid sectionColor="#555" />
        <OrbitControls makeDefault target={[-0.2, 0.18, 0]} />
      </Canvas>

      {interactive && (
        <InteractionPanel
          mode={mode}
          setMode={setMode}
          target={target}
          thetas={displayedThetas}
          selectedJoint={selectedJoint}
          setSelectedJoint={setSelectedJoint}
          updateJoint={updateJoint}
          onPreview={() => preview()}
          onMove={move}
          onReset={reset}
          isOnline={isOnline}
          error={interactiveError}
        />
      )}
    </div>
  );
};

export default ArmSimulation;
