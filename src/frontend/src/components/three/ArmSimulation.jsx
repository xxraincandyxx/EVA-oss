import React, { Suspense, useEffect, useRef, useState } from 'react';

import { Canvas, useFrame } from '@react-three/fiber';
import { Box, Grid, OrbitControls, TransformControls } from '@react-three/drei';
import * as THREE from 'three';

import useRobotStore from '../../store/useRobotStore';
import RobotModel from './RobotModel';

const jointLimits = 174.5;
const clampJoint = (value) =>
  Math.max(-jointLimits, Math.min(jointLimits, Number(value)));
const sceneToRobot = ([x, y, z]) => [x, -z, y];
const robotToScene = ([x, y, z]) => new THREE.Vector3(x, z, -y);

const jointTransforms = [
  { position: [0, 0, 0.181], rotation: [0, 0, 0] },
  { position: [0, 0, 0], rotation: [Math.PI / 2, 0, 0] },
  { position: [-0.203, 0, 0], rotation: [0, 0, 0] },
  { position: [-0.188, 0, 0], rotation: [0, 0, 0] },
  { position: [0, 0, 0.073], rotation: [Math.PI / 2, 0, 0] },
  { position: [0, 0, 0.06825], rotation: [-Math.PI / 2, 0, 0] },
];

const forwardEndEffector = (radians) => {
  const transform = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

  jointTransforms.forEach((joint, index) => {
    transform.multiply(new THREE.Matrix4().makeTranslation(...joint.position));
    transform.multiply(
      new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(...joint.rotation, 'XYZ')
      )
    );
    transform.multiply(new THREE.Matrix4().makeRotationZ(radians[index]));
  });

  transform.multiply(new THREE.Matrix4().makeTranslation(0, 0, 0.045));
  return new THREE.Vector3().setFromMatrixPosition(transform);
};

const solvePositionIk = (startThetas, targetPosition) => {
  const target = robotToScene(targetPosition);
  const radians = startThetas.map(THREE.MathUtils.degToRad);
  const limit = THREE.MathUtils.degToRad(jointLimits);
  const epsilon = 0.0001;
  const damping = 0.000025;

  for (let iteration = 0; iteration < 80; iteration += 1) {
    const current = forwardEndEffector(radians);
    const error = target.clone().sub(current);
    if (error.lengthSq() < 0.00000225) break;

    const jacobian = radians.map((_, index) => {
      const sample = [...radians];
      sample[index] += epsilon;
      return forwardEndEffector(sample).sub(current).multiplyScalar(1 / epsilon);
    });

    const system = new THREE.Matrix3().set(
      jacobian.reduce((sum, column) => sum + column.x * column.x, damping),
      jacobian.reduce((sum, column) => sum + column.x * column.y, 0),
      jacobian.reduce((sum, column) => sum + column.x * column.z, 0),
      jacobian.reduce((sum, column) => sum + column.y * column.x, 0),
      jacobian.reduce((sum, column) => sum + column.y * column.y, damping),
      jacobian.reduce((sum, column) => sum + column.y * column.z, 0),
      jacobian.reduce((sum, column) => sum + column.z * column.x, 0),
      jacobian.reduce((sum, column) => sum + column.z * column.y, 0),
      jacobian.reduce((sum, column) => sum + column.z * column.z, damping)
    );
    const weightedError = error.applyMatrix3(system.invert());

    jacobian.forEach((column, index) => {
      const step = THREE.MathUtils.clamp(column.dot(weightedError), -0.14, 0.14);
      radians[index] = THREE.MathUtils.clamp(radians[index] + step, -limit, limit);
    });
  }

  return radians.map(THREE.MathUtils.radToDeg);
};

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
  updateTarget,
  thetas,
  selectedJoint,
  setSelectedJoint,
  updateJoint,
  onPreview,
  onMove,
  onReset,
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
          <label key={axis}>
            <b>{axis}</b>
            <input
              type="number"
              step="0.005"
              value={target[index].toFixed(3)}
              onChange={(event) => updateTarget(index, event.target.value)}
              aria-label={`${axis} target in meters`}
            />
            <span>m</span>
          </label>
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
      <button type="button" onClick={onPreview}>
        Preview
      </button>
      <button
        type="button"
        className="primary"
        onClick={onMove}
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

  const updateTarget = (index, value) => {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return;
    setTarget((current) =>
      current.map((coordinate, coordinateIndex) =>
        coordinateIndex === index ? numericValue : coordinate
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
      clearInteractivePreview();
      setDraftThetas(solvePositionIk(displayedThetas, position));
      if (isOnline) {
        emit('preview_interactive_cartesian', cartesianPayload(position));
      }
    } else if (isOnline) {
      emit('simulate_with_axes', axesPayload());
    }
  };

  const move = () => {
    if (!isOnline) {
      preview();
      return;
    }
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
          updateTarget={updateTarget}
          thetas={displayedThetas}
          selectedJoint={selectedJoint}
          setSelectedJoint={setSelectedJoint}
          updateJoint={updateJoint}
          onPreview={() => preview()}
          onMove={move}
          onReset={reset}
          error={interactiveError}
        />
      )}
    </div>
  );
};

export default ArmSimulation;
