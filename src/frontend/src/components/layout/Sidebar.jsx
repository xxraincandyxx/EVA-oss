import React from 'react';
import useRobotStore from '../../store/useRobotStore';

const navItems = [
  { key: 'analytics', icon: 'bx-analyse', label: 'Analytics' },
  { key: 'control', icon: 'bx-joystick-button', label: 'Control' },
  { key: 'fpv', icon: 'bx-game', label: 'FPV' },
  { key: 'vision', icon: 'bx-camera', label: 'Vision' },
  { key: 'results', icon: 'bx-task', label: 'Results' },
  { key: 'schedule', icon: 'bx-list-ul', label: 'Schedule' },
  { key: 'agent', icon: 'bx-conversation', label: 'Agent' },
  { key: 'settings', icon: 'bx-cog', label: 'Settings' },
];

const Sidebar = ({ isCollapsed }) => {
  const activePage = useRobotStore((state) => state.activePage);
  const setActivePage = useRobotStore((state) => state.setActivePage);

  return (
    <div className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <a href="#" className="logo">
        <i className="bx bx-code-alt"></i>
        {!isCollapsed && (
          <div className="logo-name">
            <span>Eva</span>System
          </div>
        )}
      </a>
      <ul className="side-menu">
        {navItems.map((item) => (
          <li
            key={item.key}
            className={activePage === item.key ? 'active' : ''}
          >
            <a
              href="#"
              aria-label={item.label}
              title={item.label}
              onClick={(e) => {
                e.preventDefault();
                setActivePage(item.key);
              }}
            >
              <i className={`bx ${item.icon}`}></i>
              {!isCollapsed && <span>{item.label}</span>}
            </a>
          </li>
        ))}
      </ul>
      <ul className="side-menu">
        <li>
          <a href="#" className="logout" aria-label="Logout" title="Logout">
            <i className="bx bx-log-out-circle"></i>
            {!isCollapsed && <span>Logout</span>}
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;
