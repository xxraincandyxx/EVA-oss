import React, { useState, useEffect, useMemo } from 'react';

import useRobotStore from '../store/useRobotStore';

// --- Accordion Item Component for the Modal ---
const AccordionItem = ({ id, item, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const handleToggle = () => setIsOpen(!isOpen);

  // Smooth animation for accordion body
  const bodyStyle = {
    maxHeight: isOpen ? '500px' : '0px',
    paddingTop: isOpen ? '15px' : '0',
    paddingBottom: isOpen ? '15px' : '0',
  };

  return (
    <div className="accordion-item">
      <div
        className={`accordion-header ${isOpen ? 'active' : ''}`}
        onClick={handleToggle}
      >
        <span className="accordion-title">ID: {id}</span>
        <span
          className="accordion-arrow"
          style={{ transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)' }}
        >
          ▶
        </span>
      </div>
      <div className="accordion-body" style={bodyStyle}>
        <p className="item-description">{item.Desc}</p>
        <button className="select-button" onClick={() => onSelect(id)}>
          Select This Schedule
        </button>
        {/*
          Optional: Add the full details table here if you want to display it
        */}
      </div>
    </div>
  );
};

// --- Modal Component ---
const ScheduleModal = ({ onClose }) => {
  const socket = useRobotStore((state) => state.socket);
  const [dbData, setDbData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!socket) return;

    const handleResponse = (data) => {
      setDbData(data);
      setLoading(false);
    };
    socket.on('details_response', handleResponse);
    socket.emit('request_details');
    return () => socket.off('details_response', handleResponse);
  }, [socket]);

  const handleSelect = (id) => {
    if (socket) socket.emit('item_selected', { id });
    onClose();
  };

  return (
    <div id="modal-overlay">
      <div id="modal-content">
        <div className="modal-header">
          <h2>Choose a Schedule from DB</h2>
          <span onClick={onClose} className="close-btn">
            ×
          </span>
        </div>
        <div id="modal-data-container">
          {loading && <div className="loader"></div>}
          {!loading && dbData && Object.keys(dbData).length > 0
            ? Object.entries(dbData).map(([id, item]) => (
                <AccordionItem
                  key={id}
                  id={id}
                  item={item}
                  onSelect={handleSelect}
                />
              ))
            : !loading && <p>No saved schedules found.</p>}
        </div>
      </div>
    </div>
  );
};

// --- Main Schedule Page Component ---
const Schedule = () => {
  const schedule = useRobotStore((state) => state.schedule);
  const isRepeating = useRobotStore((state) => state.isRepeating);
  const socket = useRobotStore((state) => state.socket);
  const emit = useRobotStore((state) => state.emit);

  const [interval, setInterval] = useState(10);
  const [isModalOpen, setModalOpen] = useState(false);

  const isConnected = socket && socket.connected;

  const handleSave = () => {
    if (!isConnected) {
      alert('Not connected to server');
      return;
    }
    const description = prompt('Enter a description:', 'My Saved Schedule');
    if (description) emit('save_schedule', { description });
  };

  const handleRepeat = () => {
    if (!isConnected) {
      alert('Not connected to server');
      return;
    }

    if (isRepeating) {
      emit('stop_repeating_schedule');
    } else {
      const validInterval = Math.max(1, Math.min(interval, 3600));
      if (validInterval > 0 && Number.isFinite(validInterval)) {
        emit('start_repeating_schedule', { interval: validInterval });
        setInterval(validInterval);
      } else {
        alert('Please enter a valid interval between 1 and 3600 seconds.');
      }
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const resultText = e.target.result;

        // Parse the JSON string into an object
        const parsedObject = JSON.parse(resultText);

        // Extract the actual schedule array from the 'schedule' key
        const scheduleArray = parsedObject.schedule;

        // Add robustness check: Check if the extracted property is an array
        if (!Array.isArray(scheduleArray)) {
          // Changed the error message to reflect the expected structure
          throw new Error(
            'File content must be a JSON object with a "schedule" array key.'
          );
        }

        // Emit the actual array data
        // NOTE: We are emitting the array, but keeping the 'schedule' key
        // in the emitted object for consistency with your previous fix.
        emit('load_schedule_from_file', { schedule: scheduleArray });

        console.log('Schedule loaded successfully from file.');
      } catch (error) {
        alert('Error parsing or validating file content: ' + error.message);
      }
    };

    reader.onerror = () => {
      alert('Error reading file.');
    };

    reader.readAsText(file);
    event.target.value = '';
  };

  useEffect(() => {
    if (!socket) return;

    const handleScheduleData = (data) => {
      // Data is expected to be the schedule array/object itself
      if (data && Object.keys(data).length > 0) {
        // Convert JSON data to a string with '2' for nice formatting
        const jsonString = JSON.stringify(data, null, 2);

        // Create a Blob (a file-like object)
        const blob = new Blob([jsonString], { type: 'application/json' });

        // Create a temporary URL for the Blob
        const url = URL.createObjectURL(blob);

        // Create a hidden link element to trigger the download
        const a = document.createElement('a');
        a.href = url;
        a.download = `schedule_${new Date().toISOString().slice(0, 10)}.json`;

        // Append, click, and clean up the temporary element
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Revoke the temporary object URL
        URL.revokeObjectURL(url);

        console.log('File download initiated.');
      } else {
        alert('Schedule is empty or invalid. Cannot save.');
      }
    };

    // Listen for the backend response containing the schedule data
    socket.on('schedule_to_save', handleScheduleData);

    return () => {
      socket.off('schedule_to_save', handleScheduleData);
    };
  }, [socket]); // Run only when the socket connection changes

  const scheduleList = useMemo(() => {
    if (!schedule || schedule.length === 0) {
      return <p className="empty-schedule">No commands scheduled yet.</p>;
    }
    return schedule.map((item, index) => {
      let details = `Action: ${item.action}`;
      if (item.action === 'CARTESIAN')
        details = `X:${item.X}, Y:${item.Y}, Z:${item.Z}`;
      else if (item.action === 'AXES')
        details = `A1:${item.Axis1}, A2:${item.Axis2}, ...`;

      return (
        <div key={index} className="scheduled-item">
          <span className="index">{index + 1}</span>
          <span className="name">{item.name}</span>
          <span className="details">{details}</span>
        </div>
      );
    });
  }, [schedule]); // Dependency on the 'schedule' state from the store

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Schedule</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                Schedule
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="bottom-data">
        <div className="orders">
          <div className="header">
            <i className="bx bx-list-check"></i>
            <h3>Scheduled Commands</h3>
          </div>
          <div className="cache-container" id="cacheContainer">
            {scheduleList}
          </div>
        </div>

        <div className="reminders">
          <div className="header">
            <i className="bx bx-calendar-edit"></i>
            <h3>Schedule Actions</h3>
          </div>
          <div className="schedule-actions">
            <button
              className="action-btn clear"
              onClick={() => emit('clear_schedule')}
            >
              <i className="bx bx-trash"></i>
              <span>CLEAR</span>
            </button>
            <button
              className="action-btn pop"
              onClick={() => emit('pop_schedule')}
            >
              <i className="bx bx-chevrons-up"></i>
              <span>POP</span>
            </button>
            <button
              className="action-btn emit"
              onClick={() => emit('emit_schedule')}
              disabled={isRepeating}
            >
              <i className="bx bx-rocket"></i>
              <span>EMIT SCHEDULE</span>
            </button>
            <button
              className="action-btn eliminate"
              onClick={() => emit('eliminate_schedule')}
              disabled={isRepeating}
            >
              <i className="bx bx-x-circle"></i>
              <span>ELIMINATE</span>
            </button>
            <button
              className="action-btn save"
              onClick={handleSave}
              disabled={isRepeating}
            >
              <i className="bx bx-data"></i>
              <span>SAVE TO DB</span>
            </button>
            <button
              className="action-btn load"
              onClick={() => setModalOpen(true)}
              disabled={isRepeating}
            >
              <i className="bx bx-folder-open"></i>
              <span>LOAD FROM DB</span>
            </button>
            <button
              className="action-btn save"
              onClick={() => emit('request_schedule_for_save')}
              disabled={isRepeating}
            >
              <i className="bx bxs-download"></i>
              <span>SAVE TO FILE</span>
            </button>
            <button
              className="action-btn load"
              onClick={() =>
                document.getElementById('schedule-file-input')?.click()
              }
              disabled={isRepeating}
            >
              <i className="bx bxs-upload"></i>
              <span>LOAD FROM FILE</span>
            </button>
          </div>

          <div className="schedule-repeat-action">
            <button
              className={`action-btn ${isRepeating ? 'stop-repeat' : 'repeat'}`}
              onClick={handleRepeat}
            >
              <i
                className={`bx ${isRepeating ? 'bx-stop-circle' : 'bx-repost'}`}
              ></i>
              <span>{isRepeating ? 'STOP' : 'REPEAT'}</span>
            </button>
            <div className="repeat-interval-container">
              <input
                type="number"
                id="repeat-interval"
                value={interval}
                min="1"
                onChange={(e) => setInterval(Number(e.target.value))}
                disabled={isRepeating}
                required
              />
              <label htmlFor="repeat-interval">Interval (s)</label>
            </div>
          </div>
        </div>
      </div>
      {isModalOpen && (
        <ScheduleModal onClose={() => setModalOpen(false)} />
      )}
      {/* Hidden file input can be placed here */}
      <input
        type="file"
        id="schedule-file-input"
        style={{ display: 'none' }}
        accept=".json,.txt"
        onChange={handleFileUpload}
      />
    </>
  );
};

export default Schedule;
