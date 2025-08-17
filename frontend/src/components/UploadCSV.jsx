import React, { useState } from "react";
import axios from "axios";

function UploadCSV() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMessage("Processing CSV, please wait...");

    try {
      const res = await axios.post(
        "http://localhost:5000/api/predict_csv",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setPredictions(res.data);
      setMessage("✅ Predictions completed!");
    } catch (error) {
      console.error(error);
      setMessage("❌ Prediction failed: " + error.message);
    }

    setLoading(false);
  };

  const downloadCSV = () => {
    if (predictions.length === 0) return;
    const csvHeader = Object.keys(predictions[0]).join(",");
    const csvRows = predictions.map((row) =>
      Object.values(row)
        .map((val) => `"${val}"`)
        .join(",")
    );
    const csvContent = [csvHeader, ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ marginBottom: "2rem" }}>
      <h2>📁 Upload Customer CSV</h2>
      <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>
      {message && <p>{message}</p>}

      {predictions.length > 0 && (
        <>
          <button onClick={downloadCSV} style={{ marginTop: "1rem" }}>
            📥 Download Predictions CSV
          </button>
          <div style={{ maxHeight: "300px", overflowY: "scroll", marginTop: "1rem" }}>
            <table border="1" cellPadding="5">
              <thead>
                <tr>
                  {Object.keys(predictions[0]).map((col, idx) => (
                    <th key={idx}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 20).map((row, idx) => (
                  <tr key={idx}>
                    {Object.values(row).map((val, i) => (
                      <td key={i}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p>Showing first 20 rows.</p>
          </div>
        </>
      )}
    </div>
  );
}

export default UploadCSV;
