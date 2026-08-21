/* Analytics.jsx */

import React, { useState, useRef } from 'react';
import { publicUrl } from '../publicUrl';
import useRobotStore from '../store/useRobotStore';

const Analytics = () => {
  // Get global state from the Zustand store
  const isOnline = useRobotStore((state) => state.isOnline);

  // Manage local component state for the image browser with useState
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [view, setView] = useState('folderSelect'); // 'folderSelect' or 'imageView'
  const fileInputRef = useRef(null); // To trigger the hidden file input

  // Replace vanilla JS functions with React event handlers
  const handleFolderSelect = (event) => {
    const files = Array.from(event.target.files).filter((file) =>
      file.type.startsWith('image/')
    );
    if (files.length > 0) {
      const imageObjects = files.map((file) => ({
        url: URL.createObjectURL(file),
        name: file.name,
      }));
      setImages(imageObjects);
      setCurrentIndex(0);
      setView('imageView');
    }
  };

  const changeImage = (direction) => {
    setCurrentIndex((prevIndex) => {
      const newIndex = prevIndex + direction;
      if (newIndex < 0) return images.length - 1;
      if (newIndex >= images.length) return 0;
      return newIndex;
    });
  };

  const showFolderView = () => {
    setImages([]);
    setView('folderSelect');
  };

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Analytics</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                Analytics
              </a>
            </li>
          </ul>
        </div>
      </div>

      <ul className="insights">
        {/* UI updates reactively based on `isOnline` state from Zustand */}
        <li className={isOnline ? 'online' : 'offline'}>
          <i className="bx bx-show-alt"></i>
          <span className="info">
            <h3 className="system-status-indicator">
              {isOnline ? 'Online' : 'Offline'}
            </h3>
            <p>System Status</p>
          </span>
        </li>
        {/* Other static insights */}
        <li>
          <i className="bx bx-code-block"></i>
          <span className="info">
            <h3>v0.1.0</h3>
            <p>Version</p>
          </span>
        </li>
        <li>
          <i className="bx bx-time"></i>
          <span className="info">
            <h3>--:--:--</h3>
            <p>Uptime</p>
          </span>
        </li>
        <li>
          <i className="bx bx-wifi"></i>
          <span className="info">
            <h3>WebSocket</h3>
            <p>Connection</p>
          </span>
        </li>
      </ul>

      <div className="bottom-data">
        <div className="orders">
          <div className="header">
            <i className="bx bx-images"></i>
            <h3>Image Browser</h3>
          </div>
          <div className="image-browser">
            <div className="image-browser-container">
              {/*  Conditional rendering based on local state */}
              {view === 'folderSelect' && (
                <button
                  className="select-folder-btn"
                  onClick={() => fileInputRef.current.click()}
                >
                  <i className="bx bx-folder-open"></i>
                  <span>Choose Object</span>
                </button>
              )}
              {view === 'imageView' && images.length > 0 && (
                <div className="image-view" style={{ display: 'flex' }}>
                  <button className="back-button" onClick={showFolderView}>
                    <i className="bx bx-arrow-back"></i>
                    <span>Return</span>
                  </button>
                  <div className="image-container">
                    <img
                      src={images[currentIndex].url}
                      alt={images[currentIndex].name}
                      id="currentImage"
                    />
                  </div>
                  <button
                    className="nav-button prev"
                    onClick={() => changeImage(-1)}
                  >
                    <i className="bx bx-chevron-left"></i>
                  </button>
                  <button
                    className="nav-button next"
                    onClick={() => changeImage(1)}
                  >
                    <i className="bx bx-chevron-right"></i>
                  </button>
                  <div className="counter">{`${currentIndex + 1}/${
                    images.length
                  }`}</div>
                </div>
              )}
            </div>
            {/* The hidden input is controlled by the ref */}
            <input
              type="file"
              ref={fileInputRef}
              webkitdirectory=""
              directory=""
              multiple
              hidden
              onChange={handleFolderSelect}
            />
          </div>
        </div>

        <div className="reminders">
          <div className="header">
            <i className="bx bxl-flutter"></i>
            <h3 data-i18n="evaControlSystem">EVA Control System</h3>
          </div>
          <div className="system-info">
            <img
              src={publicUrl('figures/logo.png')}
              alt="EVA Icon"
              className="eva-icon"
            />
            <p className="system-tagline" data-i18n="evaTagline">
              AI-Powered Robotic Arm Interface
            </p>
            <div className="info-grid">
              <div>
                <strong data-i18n="backend">Backend</strong> PyTorch + Flask +
                Socket.IO
              </div>
              <div>
                <strong data-i18n="controlProtocol">Control Protocol</strong>{' '}
                Realtime PID via API
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Analytics;
