import React, { useState, useEffect } from 'react';
import { publicUrl } from '../../publicUrl';

const Navbar = ({ onToggleSidebar }) => {
  const [isDarkMode, setIsDarkMode] = useState(
    localStorage.getItem('theme') === 'dark'
  );

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.body.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  const handleThemeToggle = () => {
    setIsDarkMode((prev) => !prev);
  };

  return (
    <nav>
      <i className="bx bx-menu" onClick={onToggleSidebar} style={{cursor: 'pointer'}}></i>
      <form action="#">
        <div className="form-input">
          <input type="search" placeholder="Search..." />
          <button className="search-btn" type="submit">
            <i className="bx bx-search"></i>
          </button>
        </div>
      </form>

      <input
        type="checkbox"
        id="theme-toggle"
        hidden
        checked={isDarkMode}
        onChange={handleThemeToggle}
      />
      <label htmlFor="theme-toggle" className="theme-toggle">
        <i className="bx bxs-sun"></i>
        <i className="bx bxs-moon"></i>
      </label>

      <a href="#" className="notif">
        <i className="bx bx-bell"></i>
        <span className="count">12</span>
      </a>
      <a href="#" className="profile">
        <img src={publicUrl('figures/profile.png')} alt="Profile" />
      </a>
    </nav>
  );
};

export default Navbar;
