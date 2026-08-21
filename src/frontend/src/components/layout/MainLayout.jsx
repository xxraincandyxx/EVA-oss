import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Navbar from './Navbar';

const MainLayout = ({ children }) => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  return (
    <>
      <Sidebar isCollapsed={isSidebarCollapsed} />
      <div className={`content ${isSidebarCollapsed ? 'content-expanded' : ''}`}>
        <Navbar onToggleSidebar={toggleSidebar} />
        <main>{children}</main>
      </div>
    </>
  );
};

export default MainLayout;