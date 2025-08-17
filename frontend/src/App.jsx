import React, { useState } from "react";
import axios from "axios";
import {
  Container, Row, Col, Table, Button, Card, Spinner
} from "react-bootstrap";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState([]);
  const [eda, setEda] = useState({});
  const [page, setPage] = useState(0);
  const [totalRows, setTotalRows] = useState(0);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const uploadCSV = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post("http://127.0.0.1:5000/api/predict_csv", formData);
      setData(res.data.preview);
      setEda(res.data.eda_preview);
      setTotalRows(res.data.total_rows || res.data.preview.length);
      setPage(0);
    } catch (err) {
      console.error(err);
      alert("Upload failed!");
    }
    setLoading(false);
  };

  const loadPage = async (newPage) => {
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/api/get_chunk", { page: newPage });
      setData(res.data.rows);
      setEda(res.data.eda_preview);
      setPage(res.data.page);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const numericCols = eda.numeric_summary ? Object.keys(eda.numeric_summary) : [];
  const catCols = eda.top_categories ? Object.keys(eda.top_categories) : [];

  return (
    <Container fluid className="mt-4" style={{ minHeight: "100vh", backgroundColor: "#f8f9fa" }}>
      
      {/* Pre-upload landing page */}
      {data.length === 0 && (
        <Row className="align-items-center mb-5">
          <Col xs={12} md={6}>
            <h1 style={{ fontWeight: "700", color: "#0d6efd", marginBottom: "1rem" }}>
              Insurance Risk Analytics Dashboard
            </h1>
            <p style={{ fontSize: "1.1rem", color: "#555", marginBottom: "2rem" }}>
              Analyze insurance risk data, predict claim probabilities, severities,
              and premiums with a single upload. Make informed decisions faster and smarter.
            </p>

            {/* Marketing highlights in cards */}
            <Row>
              <Col xs={12} className="mb-3">
                <Card className="shadow-sm" style={{ borderLeft: "5px solid #0d6efd" }}>
                  <Card.Body>
                    <Card.Title style={{ fontWeight: "600" }}>Predictive Insights</Card.Title>
                    <Card.Text>Get precise claim probability, severity, and premium predictions to understand risk better.</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
              <Col xs={12} className="mb-3">
                <Card className="shadow-sm" style={{ borderLeft: "5px solid #198754" }}>
                  <Card.Body>
                    <Card.Title style={{ fontWeight: "600" }}>Easy Upload</Card.Title>
                    <Card.Text>Simply upload your insurance dataset in CSV format and visualize predictions instantly.</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
              <Col xs={12} className="mb-3">
                <Card className="shadow-sm" style={{ borderLeft: "5px solid #ffc107" }}>
                  <Card.Body>
                    <Card.Title style={{ fontWeight: "600" }}>Data-Driven Decisions</Card.Title>
                    <Card.Text>Transform raw insurance data into actionable insights to optimize risk management and policy strategies.</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Upload form */}
            <div className="d-flex align-items-center mt-4">
              <input type="file" accept=".csv" onChange={handleFileChange} className="form-control"/>
              <Button variant="primary" className="ms-2" onClick={uploadCSV} disabled={loading}>
                {loading ? <Spinner animation="border" size="sm"/> : "Upload CSV"}
              </Button>
            </div>

            <p className="mt-4 text-muted">Developed by <strong>Nurye Nigus</strong></p>
          </Col>

          <Col xs={12} md={6} className="text-center mt-4 mt-md-0">
            <img
              src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
              alt="Insurance Analytics Illustration"
              style={{ maxWidth: "90%", height: "auto", borderRadius: "10px", boxShadow: "0 5px 15px rgba(0,0,0,0.1)" }}
            />
          </Col>
        </Row>
      )}

      {/* Predictions Table */}
      {data.length > 0 && (
        <Row>
          <Col>
            <Card className="mb-4 shadow-sm">
              <Card.Header style={{ fontWeight: "600" }}>
                Predictions (Page {page + 1} of {Math.ceil(totalRows / 10)})
              </Card.Header>
              <Card.Body style={{ maxHeight: "350px", overflowY: "auto" }}>
                <Table striped bordered hover size="sm">
                  <thead>
                    <tr>
                      {Object.keys(data[0]).map((col) => <th key={col}>{col}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {data.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((val, i) => <td key={i}>{val}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </Table>
                <div className="d-flex justify-content-between mt-2">
                  <Button disabled={page === 0} onClick={() => loadPage(page - 1)}>Previous</Button>
                  <Button disabled={(page + 1) * 10 >= totalRows} onClick={() => loadPage(page + 1)}>Next</Button>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

      {/* Numeric EDA Charts */}
      {numericCols.length > 0 && (
        <Row>
          {numericCols.map((col) => {
            const stats = eda.numeric_summary[col];
            const chartData = [
              { label: 'Min', value: stats.min },
              { label: '25%', value: stats['25%'] },
              { label: '50%', value: stats['50%'] },
              { label: '75%', value: stats['75%'] },
              { label: 'Max', value: stats.max },
            ];
            return (
              <Col md={4} key={col} className="mb-3">
                <Card className="shadow-sm">
                  <Card.Body>
                    <h6>{col}</h6>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="label"/>
                        <YAxis/>
                        <Tooltip/>
                        <Bar dataKey="value" fill="#0d6efd"/>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card.Body>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}

      {/* Categorical EDA Charts */}
      {catCols.length > 0 && (
        <Row>
          {catCols.map((col) => {
            const catData = Object.entries(eda.top_categories[col]).map(([key,value]) => ({name:key, value}));
            return (
              <Col md={4} key={col} className="mb-3">
                <Card className="shadow-sm">
                  <Card.Body>
                    <h6>{col}</h6>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={catData}>
                        <XAxis dataKey="name" tick={{ fontSize: 10 }}/>
                        <YAxis/>
                        <Tooltip/>
                        <Bar dataKey="value" fill="#198754"/>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card.Body>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}
    </Container>
  );
}

export default App;
