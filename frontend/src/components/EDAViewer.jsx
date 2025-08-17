import React, { useEffect, useState } from 'react';
import axios from 'axios';

function EDAViewer() {
  const [edaData, setEdaData] = useState(null);

  useEffect(() => {
    const fetchEDA = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/eda");
        setEdaData(res.data);
      } catch (error) {
        setEdaData({ error: "Failed to load EDA: " + error.message });
      }
    };
    fetchEDA();
  }, []);

  return (
    <div>
      <h2>📈 EDA Summary</h2>
      {edaData ? (
        <div>
          {edaData.error ? (
            <p>{edaData.error}</p>
          ) : (
            <>
              <p><strong>Rows:</strong> {edaData.rows}</p>
              <p><strong>Columns:</strong> {edaData.columns}</p>
              <p><strong>Column Names:</strong></p>
              <ul>
                {edaData.column_names.map((col, idx) => <li key={idx}>{col}</li>)}
              </ul>
            </>
          )}
        </div>
      ) : (
        <p>Loading EDA...</p>
      )}
    </div>
  );
}

export default EDAViewer;
