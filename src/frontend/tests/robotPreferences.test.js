import assert from 'node:assert/strict';
import test from 'node:test';

import { evaRobotModel } from '../src/components/three/robotModels/eva.js';
import {
  clearInitialPose,
  loadInitialPose,
  normalizeInitialPose,
  saveInitialPose,
} from '../src/store/robotPreferences.js';

test('initial pose preferences validate numbers and normalize angles', () => {
  const normalized = normalizeInitialPose(
    {
      position: ['0.2', 'invalid', 0.4],
      orientation: [370, -190, 45],
    },
    evaRobotModel.initialPose
  );

  assert.deepEqual(normalized.position, [0.2, -0.118, 0.4]);
  assert.deepEqual(normalized.orientation, [10, 170, 45]);
});

test('initial pose preferences persist and restore model defaults', () => {
  const values = new Map();
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
    localStorage: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
      removeItem: (key) => values.delete(key),
    },
    },
  });

  try {
    const pose = { position: [0.1, 0.2, 0.3], orientation: [10, 20, 30] };
    assert.deepEqual(saveInitialPose(evaRobotModel, pose), pose);
    assert.deepEqual(loadInitialPose(evaRobotModel), pose);
    assert.deepEqual(clearInitialPose(evaRobotModel), evaRobotModel.initialPose);
    assert.deepEqual(loadInitialPose(evaRobotModel), evaRobotModel.initialPose);
  } finally {
    Reflect.deleteProperty(globalThis, 'window');
  }
});
