import React, { useState } from 'react';
import axios from 'axios';

function ModelPrediction() {
  const [inputData, setInputData] = useState({});
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/predict", inputData);
      setResult(res.data);
    } catch (error) {
      setResult({ error: "Prediction failed: " + error.message });
    }
  };

  return (
    <div style={{ marginBottom: '2rem' }}>
      <h2>🔍 Predict Insurance Metrics</h2>
      <input name="make" placeholder="Vehicle Make" onChange={handleChange} />
      <input name="Model" placeholder="Vehicle Model" onChange={handleChange} />
      <input name="cubiccapacity" placeholder="Cubic Capacity" onChange={handleChange} />
      {/* Add more inputs based on your model features */}
      <button onClick={handleSubmit}>Predict</button>
      {result && (
        <div>
          <h4>Result:</h4>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default ModelPrediction;
