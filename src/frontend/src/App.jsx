import React, { lazy, Suspense, useEffect } from 'react';

import { io } from 'socket.io-client';
import { kBackendUrl } from './config';
import useRobotStore from './store/useRobotStore';

import MainLayout from './components/layout/MainLayout';

const pages = {
  agent: lazy(() => import('./pages/Agent')),
  analytics: lazy(() => import('./pages/Analytics')),
  control: lazy(() => import('./pages/Control')),
  fpv: lazy(() => import('./pages/Fpv')),
  results: lazy(() => import('./pages/Results')),
  schedule: lazy(() => import('./pages/Schedule')),
  settings: lazy(() => import('./pages/Settings')),
  vision: lazy(() => import('./pages/Vision')),
};

function App() {
  const activePage = useRobotStore((state) => state.activePage);
  const setSocket = useRobotStore((state) => state.setSocket);
  const initializeListeners = useRobotStore(
    (state) => state.initializeListeners
  );

  useEffect(() => {
    const newSocket = io(kBackendUrl || undefined, {
      transports: ['websocket', 'polling'],
    });
    setSocket(newSocket);
    const removeListeners = initializeListeners(newSocket);
    newSocket.emit('refresh_arm_sim');

    return () => {
      removeListeners();
      newSocket.disconnect();
      setSocket(null);
    };
  }, [setSocket, initializeListeners]);

  useEffect(() => {
    const { emit } = useRobotStore.getState();
    let refreshTimer;

    if (activePage === 'control' || activePage === 'fpv') {
      refreshTimer = window.setTimeout(() => emit('refresh_arm_sim'), 50);
    }
    if (activePage === 'fpv') {
      emit('request_tracking_status');
    }

    return () => window.clearTimeout(refreshTimer);
  }, [activePage]);

  const ActivePage = pages[activePage] ?? pages.analytics;

  return (
    <MainLayout>
      <Suspense fallback={<div className="loader" aria-label="Loading" />}>
        <ActivePage />
      </Suspense>
    </MainLayout>
  );
}

export default App;
