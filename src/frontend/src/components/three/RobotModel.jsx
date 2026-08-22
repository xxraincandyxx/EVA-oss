import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

import useRobotStore from '../../store/useRobotStore';
import { clampJointAngles } from './kinematics';
import { evaRobotModel } from './robotModels/eva';
import { Model as BaseLink } from './meshes/BaseLink';
import { Model as Link1 } from './meshes/Link1';
import { Model as Link2 } from './meshes/Link2';
import { Model as Link3 } from './meshes/Link3';
import { Model as Link4 } from './meshes/Link4';
import { Model as Link5 } from './meshes/Link5';
import { Model as Link6 } from './meshes/Link6';

const visuals = {
  base: BaseLink,
  'link-1': Link1,
  'link-2': Link2,
  'link-3': Link3,
  'link-4': Link4,
  'link-5': Link5,
  'link-6': Link6,
};

const jointColors = ['#2d74c8', '#3c9b68', '#e0a32f', '#d25353', '#7c62c9', '#2698a8'];

const JointHandle = ({ joint, index, angle, selected, onSelect, onChange }) => {
  const drag = useRef(null);
  const quaternion = useMemo(
    () =>
      new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(...joint.axis).normalize()
      ),
    [joint.axis]
  );

  if (!onChange) return null;
  const pointerX = (event) => event.nativeEvent?.clientX ?? event.clientX ?? 0;

  return (
    <mesh
      quaternion={quaternion}
      onPointerDown={(event) => {
        event.stopPropagation();
        event.target.setPointerCapture(event.pointerId);
        drag.current = { x: pointerX(event), angle };
        onSelect(index);
      }}
      onPointerMove={(event) => {
        if (!drag.current) return;
        event.stopPropagation();
        onChange(index, drag.current.angle + (pointerX(event) - drag.current.x) * 0.45);
      }}
      onPointerUp={(event) => {
        event.stopPropagation();
        event.target.releasePointerCapture(event.pointerId);
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
    >
      <torusGeometry args={[selected ? 0.043 : 0.038, 0.0045, 10, 40]} />
      <meshBasicMaterial
        color={jointColors[index % jointColors.length]}
        transparent
        opacity={selected ? 1 : 0.62}
        depthTest={false}
      />
    </mesh>
  );
};

const RobotJoint = ({
  model,
  index,
  thetas,
  setJointRef,
  interactive,
  selectedJoint,
  onJointSelect,
  onJointChange,
  endEffectorRef,
}) => {
  if (index >= model.joints.length) {
    return <group ref={endEffectorRef} {...model.tool} />;
  }

  const joint = model.joints[index];
  const Visual = visuals[joint.visual.key];
  return (
    <group position={joint.position} rotation={joint.rotation}>
      <group
        ref={(object) => {
          setJointRef(index, object);
        }}
      >
        {interactive && (
          <JointHandle
            joint={joint}
            index={index}
            angle={thetas[index]}
            selected={selectedJoint === index}
            onSelect={onJointSelect}
            onChange={onJointChange}
          />
        )}
        {Visual && <Visual {...joint.visual} />}
        <RobotJoint
          model={model}
          index={index + 1}
          thetas={thetas}
          setJointRef={setJointRef}
          interactive={interactive}
          selectedJoint={selectedJoint}
          onJointSelect={onJointSelect}
          onJointChange={onJointChange}
          endEffectorRef={endEffectorRef}
        />
      </group>
    </group>
  );
};

const RobotModel = ({
  model = evaRobotModel,
  displayThetas,
  interactive = false,
  selectedJoint = 0,
  onJointSelect,
  onJointChange,
  endEffectorRef,
}) => {
  const statusThetas = useRobotStore((state) => state.status.thetas);
  const thetas = clampJointAngles(model, displayThetas ?? statusThetas);
  const jointRefs = useRef([]);
  const BaseVisual = visuals[model.baseVisual.key];
  const setJointRef = (index, object) => {
    jointRefs.current[index] = object;
  };

  useFrame(() => {
    model.joints.forEach((joint, index) => {
      jointRefs.current[index]?.quaternion.setFromAxisAngle(
        new THREE.Vector3(...joint.axis).normalize(),
        THREE.MathUtils.degToRad(thetas[index])
      );
    });
  });

  return (
    <group position={model.root.position} rotation={model.root.rotation}>
      {BaseVisual && <BaseVisual {...model.baseVisual} />}
      <RobotJoint
        model={model}
        index={0}
        thetas={thetas}
        setJointRef={setJointRef}
        interactive={interactive}
        selectedJoint={selectedJoint}
        onJointSelect={onJointSelect}
        onJointChange={onJointChange}
        endEffectorRef={endEffectorRef}
      />
    </group>
  );
};

export default RobotModel;
