const storageKey = (modelId) => `eva.robot.${modelId}.initial-pose.v1`;

const finiteTriple = (values, fallback) =>
  fallback.map((defaultValue, index) => {
    const value = Number(values?.[index]);
    return Number.isFinite(value) ? value : defaultValue;
  });

const wrapDegrees = (value) => ((value + 180) % 360 + 360) % 360 - 180;

export const normalizeInitialPose = (pose, fallback) => ({
  position: finiteTriple(pose?.position, fallback.position),
  orientation: finiteTriple(pose?.orientation, fallback.orientation).map(
    wrapDegrees
  ),
});

export const loadInitialPose = (model) => {
  if (typeof window === 'undefined') return normalizeInitialPose(null, model.initialPose);
  try {
    const saved = JSON.parse(window.localStorage.getItem(storageKey(model.id)));
    return normalizeInitialPose(saved, model.initialPose);
  } catch {
    return normalizeInitialPose(null, model.initialPose);
  }
};

export const saveInitialPose = (model, pose) => {
  const normalized = normalizeInitialPose(pose, model.initialPose);
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(storageKey(model.id), JSON.stringify(normalized));
    } catch {
      // The in-memory preference still works when storage is unavailable.
    }
  }
  return normalized;
};

export const clearInitialPose = (model) => {
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.removeItem(storageKey(model.id));
    } catch {
      // The in-memory preference can still be reset.
    }
  }
  return normalizeInitialPose(null, model.initialPose);
};
