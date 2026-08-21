import React, { useEffect, useRef, useState } from 'react';

import { kBackendUrl } from '../config';

// --- Helper Component for Robust Streaming ---
const LiveMonitor = ({ title, endpoint, icon, style }) => {
  // Initialize with the base URL
  const [streamUrl, setStreamUrl] = useState(`${kBackendUrl}${endpoint}`);
  const retryTimer = useRef(null);

  useEffect(
    () => () => window.clearTimeout(retryTimer.current),
    []
  );

  const handleStreamError = () => {
    console.log(
      `[${title}] Stream disconnected for stream url: ${streamUrl}. Retrying...`
    );
    window.clearTimeout(retryTimer.current);
    retryTimer.current = window.setTimeout(() => {
      setStreamUrl(`${kBackendUrl}${endpoint}?retry=${Date.now()}`);
    }, 1500);
  };

  return (
    <>
      <div className="header" style={style}>
        <i className={icon}></i>
        <h3>{title}</h3>
      </div>
      <div className="monitor-container">
        <img
          src={streamUrl}
          alt={title}
          onError={handleStreamError}
          style={{
            width: '100%',
            display: 'block',
            minHeight: '200px',
            background: '#000',
          }}
        />
      </div>
    </>
  );
};

export default LiveMonitor;
