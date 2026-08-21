import React, { useEffect, useRef, useState } from 'react';

import { kBackendUrl } from '../config';
import { publicUrl } from '../publicUrl';

// --- Helper Component for Robust Streaming ---
const LiveMonitor = ({ title, endpoint, icon, style }) => {
  // Initialize with the base URL
  const [streamUrl, setStreamUrl] = useState(`${kBackendUrl}${endpoint}`);
  const [isUnavailable, setIsUnavailable] = useState(true);
  const retryTimer = useRef(null);

  useEffect(
    () => () => window.clearTimeout(retryTimer.current),
    []
  );

  const handleStreamError = () => {
    setIsUnavailable(true);
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
        {isUnavailable && (
          <div className="monitor-placeholder" role="status">
            <img
              src={publicUrl('figures/logo.png')}
              alt=""
              aria-hidden="true"
            />
            <i className={icon} aria-hidden="true"></i>
            <span>{title} unavailable</span>
          </div>
        )}
        <img
          src={streamUrl}
          alt={title}
          className={isUnavailable ? 'monitor-stream is-loading' : 'monitor-stream'}
          onLoad={() => setIsUnavailable(false)}
          onError={handleStreamError}
        />
      </div>
    </>
  );
};

export default LiveMonitor;
