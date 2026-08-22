import React, { useState } from 'react';

import { evaRobotModel } from '../components/three/robotModels/eva';
import useRobotStore from '../store/useRobotStore';

const fields = [
  { group: 'position', index: 0, id: 'home-x', label: 'X', step: '0.005', unit: 'm' },
  { group: 'position', index: 1, id: 'home-y', label: 'Y', step: '0.005', unit: 'm' },
  { group: 'position', index: 2, id: 'home-z', label: 'Z', step: '0.005', unit: 'm' },
  { group: 'orientation', index: 0, id: 'home-a', label: 'A', step: '1', unit: 'deg' },
  { group: 'orientation', index: 1, id: 'home-b', label: 'B', step: '1', unit: 'deg' },
  { group: 'orientation', index: 2, id: 'home-c', label: 'C', step: '1', unit: 'deg' },
];

const poseToForm = (pose) => ({
  position: pose.position.map(String),
  orientation: pose.orientation.map(String),
});

const Settings = () => {
  const initialPose = useRobotStore(
    (state) => state.robotInitialPoses[evaRobotModel.id]
  );
  const setRobotInitialPose = useRobotStore((state) => state.setRobotInitialPose);
  const resetRobotInitialPose = useRobotStore(
    (state) => state.resetRobotInitialPose
  );
  const [form, setForm] = useState(() => poseToForm(initialPose));
  const [saved, setSaved] = useState(false);

  const updateField = (group, index, value) => {
    setSaved(false);
    setForm((current) => ({
      ...current,
      [group]: current[group].map((entry, entryIndex) =>
        entryIndex === index ? value : entry
      ),
    }));
  };

  const save = (event) => {
    event.preventDefault();
    const pose = {
      position: form.position.map(Number),
      orientation: form.orientation.map(Number),
    };
    const savedPose = setRobotInitialPose(evaRobotModel, pose);
    setForm(poseToForm(savedPose));
    setSaved(true);
  };

  const restoreDefaults = () => {
    const defaultPose = resetRobotInitialPose(evaRobotModel);
    setForm(poseToForm(defaultPose));
    setSaved(true);
  };

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Settings</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">Settings</a>
            </li>
          </ul>
        </div>
      </div>
      <div className="bottom-data">
        <div
          className="orders settings-container"
          role="region"
          aria-labelledby="robot-home-pose-title"
        >
          <div className="header">
            <i className="bx bx-target-lock"></i>
            <h3 id="robot-home-pose-title">Robot Home Pose</h3>
          </div>
          <form className="settings-form" onSubmit={save}>
            <fieldset>
              <legend>Position</legend>
              <div className="pose-settings-grid">
                {fields.slice(0, 3).map((field) => (
                  <label key={field.id} htmlFor={field.id}>
                    <span>{field.label}</span>
                    <input
                      id={field.id}
                      type="number"
                      required
                      step={field.step}
                      value={form[field.group][field.index]}
                      onChange={(event) =>
                        updateField(field.group, field.index, event.target.value)
                      }
                    />
                    <small>{field.unit}</small>
                  </label>
                ))}
              </div>
            </fieldset>
            <fieldset>
              <legend>Gesture</legend>
              <div className="pose-settings-grid">
                {fields.slice(3).map((field) => (
                  <label key={field.id} htmlFor={field.id}>
                    <span>{field.label}</span>
                    <input
                      id={field.id}
                      type="number"
                      required
                      step={field.step}
                      value={form[field.group][field.index]}
                      onChange={(event) =>
                        updateField(field.group, field.index, event.target.value)
                      }
                    />
                    <small>{field.unit}</small>
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="settings-actions">
              <button type="button" onClick={restoreDefaults}>
                <i className="bx bx-reset"></i>
                Defaults
              </button>
              <button type="submit" className="save-btn">
                <i className="bx bx-save"></i>
                Save Pose
              </button>
            </div>
            <output className="settings-save-status" aria-live="polite">
              {saved ? 'Saved' : ''}
            </output>
          </form>
        </div>
      </div>
    </>
  );
};

export default Settings;
