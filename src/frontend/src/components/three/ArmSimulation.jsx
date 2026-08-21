import React, { Suspense } from 'react';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Box } from '@react-three/drei';
import RobotModel from './RobotModel';

const Loader = () => (
  <Box position={[0, 0.5, 0]}>
    <boxGeometry args={[0.5, 0.5, 0.5]} />
    <meshBasicMaterial color="tomato" wireframe />
  </Box>
);

const ArmSimulation = () => {
  return (
    <div className="plot-container-shared">
      <Canvas camera={{ position: [1.5, 1, 2], fov: 15 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} />

        {/* --- Add a visual coordinate system at the origin --- */}
        <axesHelper args={[1]} />

        <Suspense fallback={<Loader />}>
          <RobotModel />
        </Suspense>

        <Grid position={[0, 0, 0]} infiniteGrid sectionColor={'#555'} />
        <OrbitControls makeDefault target={[0, 0.18, 0]} />
      </Canvas>
    </div>
  );
};

export default ArmSimulation;
