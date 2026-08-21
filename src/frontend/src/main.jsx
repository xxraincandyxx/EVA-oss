import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Main Application Styles
import './assets/css/style.css';

// Component-Specific Styles
import './assets/css/agent.css';
import './assets/css/analytics.css';
import './assets/css/control.css';
import './assets/css/fpv.css';
import './assets/css/results.css';
import './assets/css/schedule.css';
import './assets/css/settings.css';
import './assets/css/vision.css';
import './assets/css/widgets.css';

// Font & Icon Assets
import 'boxicons/css/boxicons.min.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
