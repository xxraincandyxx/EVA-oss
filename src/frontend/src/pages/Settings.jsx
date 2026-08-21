import React, { useState } from 'react';

const Settings = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real app, you would send this data to the backend
    console.log('Saving settings:', formData);
    alert('Settings saved (see console).');
  };

  return (
    <>
      <div className="header">{/* ... */}</div>
      <div className="bottom-data">
        <div className="orders settings-container">
          <div className="header">
            <i className="bx bx-user-circle"></i>
            <h3>Profile Settings</h3>
          </div>
          <form className="settings-form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Your Username"
              />
            </div>
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="your.email@example.com"
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">New Password</label>
              <input
                type="password"
                id="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter new password"
              />
            </div>
            <button type="submit" className="save-btn">
              Save Changes
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default Settings;
