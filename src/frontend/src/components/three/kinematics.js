import * as THREE from 'three';

const vector = (value = [0, 0, 0]) => new THREE.Vector3(...value);
const eulerQuaternion = (value = [0, 0, 0]) =>
  new THREE.Quaternion().setFromEuler(new THREE.Euler(...value, 'XYZ'));

const transformMatrix = ({ position, rotation } = {}) =>
  new THREE.Matrix4().compose(
    vector(position),
    eulerQuaternion(rotation),
    new THREE.Vector3(1, 1, 1)
  );

const rotationVector = (from, to) => {
  const delta = to.clone().multiply(from.clone().invert()).normalize();
  if (delta.w < 0) {
    delta.x *= -1;
    delta.y *= -1;
    delta.z *= -1;
    delta.w *= -1;
  }

  const angle = 2 * Math.acos(THREE.MathUtils.clamp(delta.w, -1, 1));
  const scale = Math.sqrt(Math.max(1 - delta.w * delta.w, 0));
  if (scale < 0.000001 || angle < 0.000001) return new THREE.Vector3();
  return new THREE.Vector3(delta.x, delta.y, delta.z).multiplyScalar(angle / scale);
};

const solveLinearSystem = (matrix, vectorValues) => {
  const size = vectorValues.length;
  const rows = matrix.map((row, index) => [...row, vectorValues[index]]);

  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    if (Math.abs(rows[column][column]) < 1e-10) return Array(size).fill(0);

    for (let row = column + 1; row < size; row += 1) {
      const ratio = rows[row][column] / rows[column][column];
      for (let entry = column; entry <= size; entry += 1) {
        rows[row][entry] -= ratio * rows[column][entry];
      }
    }
  }

  const solution = Array(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    const known = rows[row]
      .slice(row + 1, size)
      .reduce((sum, value, index) => sum + value * solution[row + index + 1], 0);
    solution[row] = (rows[row][size] - known) / rows[row][row];
  }
  return solution;
};

export const clampJointAngles = (model, angles = []) =>
  model.joints.map((joint, index) => {
    const [minimum, maximum] = joint.limits;
    const value = Number(angles[index] ?? 0);
    return THREE.MathUtils.clamp(Number.isFinite(value) ? value : 0, minimum, maximum);
  });

export const forwardKinematics = (model, angles) => {
  const radians = clampJointAngles(model, angles).map(THREE.MathUtils.degToRad);
  const matrix = transformMatrix(model.root);

  model.joints.forEach((joint, index) => {
    matrix.multiply(transformMatrix(joint));
    matrix.multiply(
      new THREE.Matrix4().makeRotationAxis(
        vector(joint.axis).normalize(),
        radians[index]
      )
    );
  });
  matrix.multiply(transformMatrix(model.tool));

  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  matrix.decompose(position, quaternion, new THREE.Vector3());
  return { matrix, position, quaternion };
};

export const sceneMatrixToRobotPose = (model, sceneMatrix) => {
  const matrix = transformMatrix(model.root).invert().multiply(sceneMatrix.clone());
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  matrix.decompose(position, quaternion, new THREE.Vector3());
  const euler = new THREE.Euler().setFromQuaternion(quaternion, 'XYZ');
  return {
    position: position.toArray(),
    orientation: euler.toArray().slice(0, 3).map(THREE.MathUtils.radToDeg),
  };
};

export const robotPoseToSceneMatrix = (model, pose) => {
  const robotPose = new THREE.Matrix4().compose(
    vector(pose.position),
    new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        ...pose.orientation.map(THREE.MathUtils.degToRad),
        'XYZ'
      )
    ),
    new THREE.Vector3(1, 1, 1)
  );
  return transformMatrix(model.root).multiply(robotPose);
};

export const solvePoseIk = (
  model,
  startAngles,
  targetPose,
  { maxIterations = 120, restarts = true } = {}
) => {
  const targetMatrix = robotPoseToSceneMatrix(model, targetPose);
  const targetPosition = new THREE.Vector3();
  const targetQuaternion = new THREE.Quaternion();
  targetMatrix.decompose(targetPosition, targetQuaternion, new THREE.Vector3());

  const epsilon = 0.0001;
  const orientationWeight = 0.16;
  const damping = 0.0025;
  const solveFrom = (seed) => {
    const radians = clampJointAngles(model, seed).map(THREE.MathUtils.degToRad);

    for (let iteration = 0; iteration < maxIterations; iteration += 1) {
      const angles = radians.map(THREE.MathUtils.radToDeg);
      const current = forwardKinematics(model, angles);
      const positionError = targetPosition.clone().sub(current.position);
      const orientationError = rotationVector(current.quaternion, targetQuaternion);
      if (positionError.length() < 0.0015 && orientationError.length() < 0.02) break;

      const error = [
        ...positionError.toArray(),
        ...orientationError.multiplyScalar(orientationWeight).toArray(),
      ];
      const jacobian = model.joints.map((_, index) => {
        const sample = [...radians];
        sample[index] += epsilon;
        const sampledPose = forwardKinematics(
          model,
          sample.map(THREE.MathUtils.radToDeg)
        );
        const positionDerivative = sampledPose.position
          .clone()
          .sub(current.position)
          .multiplyScalar(1 / epsilon);
        const orientationDerivative = rotationVector(
          current.quaternion,
          sampledPose.quaternion
        ).multiplyScalar(orientationWeight / epsilon);
        return [...positionDerivative.toArray(), ...orientationDerivative.toArray()];
      });

      const normalMatrix = model.joints.map((_, row) =>
        model.joints.map((__, column) =>
          jacobian[row].reduce(
            (sum, value, index) => sum + value * jacobian[column][index],
            row === column ? damping : 0
          )
        )
      );
      const gradient = jacobian.map((column) =>
        column.reduce((sum, value, index) => sum + value * error[index], 0)
      );
      const steps = solveLinearSystem(normalMatrix, gradient);

      model.joints.forEach((joint, index) => {
        const [minimum, maximum] = joint.limits.map(THREE.MathUtils.degToRad);
        const step = Number.isFinite(steps[index])
          ? THREE.MathUtils.clamp(steps[index], -0.12, 0.12)
          : 0;
        radians[index] = THREE.MathUtils.clamp(
          radians[index] + step,
          minimum,
          maximum
        );
      });
    }

    const angles = clampJointAngles(model, radians.map(THREE.MathUtils.radToDeg));
    const result = forwardKinematics(model, angles);
    const positionError = result.position.distanceTo(targetPosition);
    const orientationError = rotationVector(
      result.quaternion,
      targetQuaternion
    ).length();
    return {
      angles,
      positionError,
      orientationError,
      score: positionError + orientationError * orientationWeight,
    };
  };

  const initial = clampJointAngles(model, startAngles);
  const seeds = [initial];
  if (restarts && model.joints.length > 1) {
    seeds.push(
      initial.map((angle, index) => angle + (index % 2 === 0 ? 5 : -5)),
      initial.map((angle, index) => angle + (index % 2 === 0 ? -5 : 5))
    );
  }

  let best = solveFrom(seeds[0]);
  for (let index = 1; index < seeds.length && best.score > 0.005; index += 1) {
    const candidate = solveFrom(seeds[index]);
    if (candidate.score < best.score) best = candidate;
  }
  return best.angles;
};

export const resolveInitialState = (model) => {
  const seed = clampJointAngles(
    model,
    model.initialJointAngles ?? Array(model.joints.length).fill(0)
  );
  if (model.initialPose) {
    const pose = {
      position: [...model.initialPose.position],
      orientation: [...model.initialPose.orientation],
    };
    return { pose, thetas: solvePoseIk(model, seed, pose) };
  }

  return {
    pose: sceneMatrixToRobotPose(model, forwardKinematics(model, seed).matrix),
    thetas: seed,
  };
};
