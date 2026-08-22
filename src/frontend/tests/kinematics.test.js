import assert from 'node:assert/strict';
import test from 'node:test';

import * as THREE from 'three';

import {
  clampJointAngles,
  forwardKinematics,
  robotPoseToSceneMatrix,
  sceneMatrixToRobotPose,
  solvePoseIk,
} from '../src/components/three/kinematics.js';
import { evaRobotModel } from '../src/components/three/robotModels/eva.js';

const poseError = (model, angles, target) => {
  const actual = forwardKinematics(model, angles);
  const expectedMatrix = robotPoseToSceneMatrix(model, target);
  const expectedPosition = new THREE.Vector3();
  const expectedQuaternion = new THREE.Quaternion();
  expectedMatrix.decompose(
    expectedPosition,
    expectedQuaternion,
    new THREE.Vector3()
  );
  return {
    position: actual.position.distanceTo(expectedPosition),
    orientation: actual.quaternion.angleTo(expectedQuaternion),
  };
};

test('EVA forward kinematics round-trips through a robot pose', () => {
  const angles = [12, -24, 36, -18, 9, 42];
  const forward = forwardKinematics(evaRobotModel, angles);
  const pose = sceneMatrixToRobotPose(evaRobotModel, forward.matrix);
  const restored = robotPoseToSceneMatrix(evaRobotModel, pose);

  const largestDifference = Math.max(
    ...forward.matrix.elements.map((value, index) =>
      Math.abs(value - restored.elements[index])
    )
  );
  assert.ok(largestDifference < 1e-10);
});

test('pose IK solves position and orientation from a singular starting pose', () => {
  const zeroAngles = Array(evaRobotModel.joints.length).fill(0);
  const initial = sceneMatrixToRobotPose(
    evaRobotModel,
    forwardKinematics(evaRobotModel, zeroAngles).matrix
  );
  const target = { ...initial, orientation: [60, 0, 0] };
  const solution = solvePoseIk(evaRobotModel, zeroAngles, target);
  const error = poseError(evaRobotModel, solution, target);

  assert.ok(error.position < 0.002, `position error was ${error.position}`);
  assert.ok(error.orientation < 0.02, `orientation error was ${error.orientation}`);
});

test('kinematics follows arbitrary joint counts, axes, and limits', () => {
  const planarModel = {
    root: { position: [0, 0, 0], rotation: [0, 0, 0] },
    joints: [
      {
        position: [0, 0, 0],
        rotation: [0, 0, 0],
        axis: [0, 0, 1],
        limits: [-90, 90],
      },
      {
        position: [1, 0, 0],
        rotation: [0, 0, 0],
        axis: [0, 1, 0],
        limits: [-45, 45],
      },
    ],
    tool: { position: [1, 0, 0], rotation: [0, 0, 0] },
  };

  assert.deepEqual(clampJointAngles(planarModel, [120, -80]), [90, -45]);
  const pose = forwardKinematics(planarModel, [90, 0]);
  assert.ok(pose.position.distanceTo(new THREE.Vector3(0, 2, 0)) < 1e-10);
});
