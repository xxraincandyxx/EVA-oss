import React from 'react';
import { publicUrl } from '../publicUrl';

// Keeping dummy data as in the original JS file
const dummyResults = [
  {
    id: 'WP-0078A',
    timestamp: '2025-10-27 14:32:15',
    imageUrl: publicUrl('figures/logo.png'),
    status: 'Passed',
    details: 'No surface flaws detected.',
  },
  {
    id: 'WP-0078B',
    timestamp: '2025-10-27 14:35:02',
    imageUrl: publicUrl('figures/logo.png'),
    status: 'Failed',
    details: 'Detected 2 micro-cracks.',
  },
  {
    id: 'WP-0079B',
    timestamp: '2025-10-27 14:40:15',
    imageUrl: publicUrl('figures/logo.png'),
    status: 'Failed',
    details: 'Detected 2 micro-cracks.',
  },
  {
    id: 'WP-0080A',
    timestamp: '2025-10-30 18:35:11',
    imageUrl: publicUrl('figures/logo.png'),
    status: 'Passed',
    details: 'Detected 2 micro-cracks.',
  },
  // ... more data
];

const Results = () => {
  return (
    <>
      <div className="header">{/* ... */}</div>
      <div className="results-grid">
        {dummyResults.map((result) => (
          <div key={result.id} className="result-card">
            <div className="card-header">
              <span className="card-id">{result.id}</span>
              <span className="card-timestamp">{result.timestamp}</span>
            </div>
            <div className="card-image-container">
              <img
                src={result.imageUrl}
                alt="Workpiece"
                className="card-image"
              />
            </div>
            <div className="card-body">
              <div className="card-status">
                <span className="card-status-label">Status:</span>
                <span
                  className={`card-status-value ${result.status.toLowerCase()}`}
                >
                  {result.status.toUpperCase()}
                </span>
              </div>
              <p className="card-details">{result.details}</p>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default Results;
