import React, { useState, useEffect } from 'react';

import LiveMonitor from './VideoUtils';
import useRobotStore from '../store/useRobotStore';

const Vision = () => {
  const socket = useRobotStore((state) => state.socket);

  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleDetectClick = () => {
    if (!socket) {
      console.error('Socket is not connected. Cannot start detection.');
      setError('Not connected to the server.');
      return;
    }
    setIsLoading(true);
    setImageUrl(null);
    setError(null);
    socket.emit('start_detection');
  };

  useEffect(() => {
    if (!socket) return;

    const onDetectionComplete = (data) => {
      setIsLoading(false);
      if (data.imageUrl) {
        // Add a timestamp to break browser cache
        setImageUrl(data.imageUrl + '?t=' + new Date().getTime());
      }
      if (data.error) {
        setError(data.error);
        setImageUrl(null); // Clear any previous image
      }
    };

    socket.on('detection_complete', onDetectionComplete);

    // Cleanup function is crucial to prevent memory leaks
    return () => socket.off('detection_complete', onDetectionComplete);
  }, [socket]); // The effect depends on the socket instance

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Vision</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                Vision
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="bottom-data">
        <div className="orders">
          <LiveMonitor
            title="Real-time Monitor"
            endpoint="/realtime_monitor"
            icon="bx bx-video"
          />
        </div>
        <div className="reminders">
          <div className="header">
            <i className="bx bx-scan"></i>
            <h3>Detection</h3>
          </div>
          <div className="detection-panel">
            <div className="detection-result-container">
              {/* Conditional Rendering based on state */}
              {!imageUrl && !error && !isLoading && (
                <>
                  <i className="bx bx-image-add"></i>
                  <p>Click "Detect" to start analysis</p>
                </>
              )}
              {imageUrl && (
                <img
                  src={imageUrl}
                  alt="Detection Result"
                  // The image now shows via the state change, no need for style
                />
              )}
              {error && (
                <p style={{ color: 'var(--danger)' }}>Error: {error}</p>
              )}
            </div>
            {isLoading && (
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: '70%' }}></div>
              </div>
            )}
            <button
              id="detect-btn"
              className="detect-btn"
              onClick={handleDetectClick}
              disabled={isLoading}
            >
              <i className="bx bx-bullseye"></i>
              <span>{isLoading ? 'Processing...' : 'DETECT'}</span>
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Vision;
