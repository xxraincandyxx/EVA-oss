import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react';

import { Canvas, useThree } from '@react-three/fiber';
import { Box, Grid, OrbitControls, TransformControls } from '@react-three/drei';
import * as THREE from 'three';

import useRobotStore from '../../store/useRobotStore';
import {
  clampJointAngles,
  resolveInitialState,
  robotPoseToSceneMatrix,
  sceneMatrixToRobotPose,
  solvePoseIk,
} from './kinematics';
import RobotModel from './RobotModel';
import { evaRobotModel } from './robotModels/eva';
import { evaRobotVisuals } from './robotModels/evaVisuals';

const poseFields = [
  { group: 'position', index: 0, label: 'X', step: 0.005, digits: 3, unit: 'm' },
  { group: 'position', index: 1, label: 'Y', step: 0.005, digits: 3, unit: 'm' },
  { group: 'position', index: 2, label: 'Z', step: 0.005, digits: 3, unit: 'm' },
  { group: 'orientation', index: 0, label: 'A', step: 1, digits: 1, unit: 'deg' },
  { group: 'orientation', index: 1, label: 'B', step: 1, digits: 1, unit: 'deg' },
  { group: 'orientation', index: 2, label: 'C', step: 1, digits: 1, unit: 'deg' },
];

const Loader = () => (
  <Box position={[0, 0.5, 0]}>
    <boxGeometry args={[0.5, 0.5, 0.5]} />
    <meshBasicMaterial color="tomato" wireframe />
  </Box>
);

const EndEffectorTarget = ({
  model,
  pose,
  active,
  transformMode,
  onChange,
  onCommit,
  resetToken,
}) => {
  const targetRef = useRef();
  const dragging = useRef(false);
  const [targetObject, setTargetObject] = useState(null);
  const invalidate = useThree((state) => state.invalidate);

  useEffect(() => {
    if (dragging.current || !targetRef.current) return;
    const matrix = robotPoseToSceneMatrix(model, pose);
    matrix.decompose(
      targetRef.current.position,
      targetRef.current.quaternion,
      targetRef.current.scale
    );
    invalidate();
  }, [invalidate, model, pose, resetToken, targetObject]);

  const currentPose = () => {
    targetRef.current.updateMatrix();
    return sceneMatrixToRobotPose(model, targetRef.current.matrix);
  };

  return (
    <>
      {active && targetObject && (
        <TransformControls
          object={targetObject}
          mode={transformMode}
          space="world"
          size={0.65}
          onMouseDown={() => {
            dragging.current = true;
          }}
          onObjectChange={() => onChange(currentPose())}
          onMouseUp={() => {
            dragging.current = false;
            onCommit(currentPose());
          }}
        />
      )}
      <group
        visible={active}
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
        <axesHelper args={[0.065]} />
      </group>
    </>
  );
};

const InteractionPanel = ({
  model,
  mode,
  setMode,
  pose,
  updatePoseValue,
  transformMode,
  setTransformMode,
  traceTarget,
  setTraceTarget,
  thetas,
  selectedJoint,
  setSelectedJoint,
  updateJoint,
  onPreview,
  onMove,
  onReset,
  error,
}) => {
  const selectedDefinition = model.joints[selectedJoint];

  return (
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
        <>
          <div className="arm-target-tools">
            <div className="arm-transform-switch" aria-label="End-effector transform">
              <button
                type="button"
                className={transformMode === 'translate' ? 'active' : ''}
                onClick={() => setTransformMode('translate')}
                title="Move end effector"
              >
                <i className="bx bx-move"></i>
                Position
              </button>
              <button
                type="button"
                className={transformMode === 'rotate' ? 'active' : ''}
                onClick={() => setTransformMode('rotate')}
                title="Rotate end effector"
              >
                <i className="bx bx-rotate-right"></i>
                Gesture
              </button>
            </div>
            <label className="arm-trace-toggle">
              <input
                type="checkbox"
                checked={traceTarget}
                onChange={(event) => setTraceTarget(event.target.checked)}
              />
              <span>Trace target</span>
            </label>
          </div>
          <div className="arm-target-readout">
            {poseFields.map((field) => (
              <label key={field.label}>
                <b>{field.label}</b>
                <input
                  type="number"
                  step={field.step}
                  value={pose[field.group][field.index].toFixed(field.digits)}
                  onChange={(event) =>
                    updatePoseValue(field.group, field.index, event.target.value)
                  }
                  aria-label={`${field.label} ${field.group} target`}
                />
                <span>{field.unit}</span>
              </label>
            ))}
          </div>
        </>
      ) : (
        <div className="arm-joint-editor">
          <div className="arm-joint-tabs">
            {model.joints.map((joint, index) => (
              <button
                type="button"
                key={joint.id}
                className={selectedJoint === index ? 'active' : ''}
                onClick={() => setSelectedJoint(index)}
                aria-label={`Select ${joint.label}`}
              >
                {joint.label}
              </button>
            ))}
          </div>
          <label>
            <span>{selectedDefinition.label}</span>
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
              min={selectedDefinition.limits[0]}
              max={selectedDefinition.limits[1]}
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
        <button type="button" className="primary" onClick={onMove}>
          Move
        </button>
      </div>
    </div>
  );
};

const ArmSimulation = ({
  interactive = false,
  model = evaRobotModel,
  visuals = evaRobotVisuals,
}) => {
  const status = useRobotStore((state) => state.status);
  const isOnline = useRobotStore((state) => state.isOnline);
  const interactivePreview = useRobotStore((state) => state.interactivePreview);
  const interactiveError = useRobotStore((state) => state.interactiveError);
  const savedInitialPose = useRobotStore(
    (state) => state.robotInitialPoses[model.id]
  );
  const emit = useRobotStore((state) => state.emit);
  const clearInteractivePreview = useRobotStore(
    (state) => state.clearInteractivePreview
  );

  const configuredInitialState = useMemo(
    () =>
      resolveInitialState({
        ...model,
        initialPose: savedInitialPose ?? model.initialPose,
      }),
    [model, savedInitialPose]
  );
  const currentInitialState = () =>
    isOnline
      ? {
          pose: {
            position: status.position.slice(0, 3),
            orientation: status.orientation.slice(0, 3),
          },
          thetas: clampJointAngles(model, status.thetas),
        }
      : configuredInitialState;
  const [mode, setMode] = useState('target');
  const [transformMode, setTransformMode] = useState('translate');
  const [traceTarget, setTraceTargetState] = useState(true);
  const [selectedJoint, setSelectedJoint] = useState(0);
  const [draftThetas, setDraftThetas] = useState(configuredInitialState.thetas);
  const [targetPose, setTargetPose] = useState(configuredInitialState.pose);
  const [resetToken, setResetToken] = useState(0);
  const displayedThetasRef = useRef(draftThetas);
  const traceFrameRef = useRef(null);
  const pendingPoseRef = useRef(null);

  const previewThetas =
    interactivePreview?.thetas?.length === model.joints.length
      ? clampJointAngles(model, interactivePreview.thetas)
      : null;
  const displayedThetas = previewThetas ?? draftThetas;

  useEffect(() => {
    displayedThetasRef.current = displayedThetas;
  }, [displayedThetas]);

  useEffect(
    () => () => {
      if (traceFrameRef.current) cancelAnimationFrame(traceFrameRef.current);
    },
    []
  );

  const updateJoint = (index, value) => {
    clearInteractivePreview();
    const nextThetas = clampJointAngles(
      model,
      displayedThetasRef.current.map((theta, jointIndex) =>
        jointIndex === index ? Number(value) : theta
      )
    );
    displayedThetasRef.current = nextThetas;
    setDraftThetas(nextThetas);
  };

  const cartesianPayload = (pose = targetPose) => ({
    x: pose.position[0],
    y: pose.position[1],
    z: pose.position[2],
    a: pose.orientation[0],
    b: pose.orientation[1],
    c: pose.orientation[2],
    duration: 2,
    rotation: 0,
  });

  const axesPayload = () => ({
    ...Object.fromEntries(
      displayedThetas.map((theta, index) => [
        `axis${index + 1}`,
        theta - Number(status.thetas[index] ?? 0),
      ])
    ),
    duration: 2,
    rotation: 0,
  });

  const previewPose = (pose, requestBackend = false, fast = false) => {
    clearInteractivePreview();
    const solution = solvePoseIk(model, displayedThetasRef.current, pose, {
      maxIterations: fast ? 24 : 120,
      restarts: !fast,
    });
    displayedThetasRef.current = solution;
    setDraftThetas(solution);
    if (requestBackend && isOnline) {
      emit('preview_interactive_cartesian', cartesianPayload(pose));
    }
  };

  const updateTargetPose = (pose, commit = false) => {
    setTargetPose(pose);
    if (!traceTarget) return;

    pendingPoseRef.current = pose;
    if (commit) {
      if (traceFrameRef.current) cancelAnimationFrame(traceFrameRef.current);
      traceFrameRef.current = null;
      previewPose(pose, true);
      return;
    }
    if (traceFrameRef.current) return;
    traceFrameRef.current = requestAnimationFrame(() => {
      traceFrameRef.current = null;
      previewPose(pendingPoseRef.current, false, true);
    });
  };

  const updatePoseValue = (group, index, value) => {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return;
    const pose = {
      position: [...targetPose.position],
      orientation: [...targetPose.orientation],
    };
    pose[group][index] = numericValue;
    updateTargetPose(pose);
  };

  const setTraceTarget = (enabled) => {
    setTraceTargetState(enabled);
    if (enabled) previewPose(targetPose);
  };

  const preview = () => {
    if (mode === 'target') {
      previewPose(targetPose, true);
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
    const initialState = currentInitialState();
    displayedThetasRef.current = initialState.thetas;
    setDraftThetas(initialState.thetas);
    setTargetPose(initialState.pose);
    setResetToken((token) => token + 1);
    clearInteractivePreview();
  };

  return (
    <div className={`plot-container-shared${interactive ? ' is-interactive' : ''}`}>
      <Canvas
        camera={{ position: [1.3, 1, 2.2], fov: 15 }}
        dpr={[1, 1.5]}
        frameloop="demand"
        gl={{ antialias: false, powerPreference: 'low-power' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} />
        <axesHelper args={[1]} />

        <Suspense fallback={<Loader />}>
          <RobotModel
            model={model}
            visuals={visuals}
            displayThetas={interactive ? displayedThetas : undefined}
            interactive={interactive && mode === 'joints'}
            selectedJoint={selectedJoint}
            onJointSelect={setSelectedJoint}
            onJointChange={updateJoint}
          />
          {interactive && (
            <EndEffectorTarget
              model={model}
              pose={targetPose}
              active={mode === 'target'}
              transformMode={transformMode}
              onChange={(pose) => updateTargetPose(pose)}
              onCommit={(pose) => updateTargetPose(pose, true)}
              resetToken={resetToken}
            />
          )}
        </Suspense>

        <Grid position={[0, 0, 0]} infiniteGrid sectionColor="#555" />
        <OrbitControls makeDefault target={[-0.2, 0.18, 0]} />
      </Canvas>

      {interactive && (
        <InteractionPanel
          model={model}
          mode={mode}
          setMode={setMode}
          pose={targetPose}
          updatePoseValue={updatePoseValue}
          transformMode={transformMode}
          setTransformMode={setTransformMode}
          traceTarget={traceTarget}
          setTraceTarget={setTraceTarget}
          thetas={displayedThetas}
          selectedJoint={selectedJoint}
          setSelectedJoint={setSelectedJoint}
          updateJoint={updateJoint}
          onPreview={preview}
          onMove={move}
          onReset={reset}
          error={interactiveError}
        />
      )}
    </div>
  );
};

export default ArmSimulation;
