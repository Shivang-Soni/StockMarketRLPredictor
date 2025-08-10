import React, { useState } from 'react';
import { Chart as ChartJS, LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend } from "chart.js";
import "./App.css";
import PredictChart from "./predictChart";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend);

function App() {
  const debugging_sample={
  dates: ['2024-07-01', '2024-07-02', '2024-07-03'],
  actual_prices: [150, 155, 160],
  predictions: [152, 157, 162]
}
  const [predictionData, setPredictionData] = useState(null);
  const [response, setResponse] = useState(null);
  const [symbol, setSymbol] = useState("AAPL");
  const [loadState, setLoadstate] = useState(false);
  const [progress, setProgress] = useState(0)
  const makePrediction = async () => {
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });
      const data = await res.json();
      setPredictionData({
        dates: data.dates,
        actual_prices: data.actual_prices,
        predictions: data.predictions,
      });
      console.log(predictionData)
    } catch (err) {
      console.log("Error during prediction:", err);
    }
  };

  const handleTraining = async () => {
    try {
      setLoadstate(true);
      const res = await fetch("http://localhost:5000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });
      const data = await res.json();
      setResponse(data.status);
      if (data.status === "Training finished.") {
        alert("⚒️ Training finished!");
      }
    } catch (err) {
      console.error(err);
      setResponse("Error during API Call");
    } finally {
      setLoadstate(false);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial", backgroundColor: "blue", minHeight: "100vh" }}>
      <h1 style={{ color: "white" }}>📈 Stock RL Trainer</h1>

      {/* Horizontal Layout */}
      <div style={{ marginBottom: "1rem" }}>
        <input
          value={symbol}
          placeholder="Enter symbol"
          onChange={(e) => setSymbol(e.target.value)}
          style={{ padding: "10px", marginRight: "10px" }}
        />
        <button
          onClick={handleTraining}
          disabled={loadState}
          style={{
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            backgroundColor: loadState ? "skyblue" : "white",
            cursor: "pointer",
          }}
        >
          {loadState ? "In Progress": "Start Training"}
        </button>
      </div>
      <button
        onClick={makePrediction}
        style={{ padding: "8px 16px", backgroundColor: "white", cursor: "pointer" }}
      >
        Show Predictions
      </button>
      {!predictionData && <p style={{color: 'white'}}>Prediction Data hasn't been loaded yet!</p>}
      {predictionData && (
        <div style={{ backgroundColor: "white", width: "80%", margin: "20px auto", padding: "20px", borderRadius: "10px" }}>
          <PredictChart
            dates={predictionData.dates}
            predictions={predictionData.predictions}
            actual_prices={predictionData.actual_prices}
          />
        </div>
      )}

      <p style={{ marginTop: "2rem", color: "white" }}>{response}</p>
    </div>
  );
}

export default App;
