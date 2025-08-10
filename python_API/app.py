from flask import Flask, request, jsonify
import pandas as pd
import os
from flask_cors import CORS

from data_loader import load_stock_data 
from agent import DQNAgent, load_agent_model  
from environment import StockEnv
from main import train

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Flask API running successfully!"

@app.route("/train", methods=["POST"])
def train_route():
    data = request.get_json()
    symbol = data.get("symbol", "AAPL")
    try:
        summary = train(symbol)
        return jsonify({"status": "Training finished.", "result": summary})
    except Exception as e:
        print("Error during Training:", str(e))
        return jsonify({"status": "Error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get("symbol", "AAPL")
        df = load_stock_data(symbol)
        recent_df = df.tail(10).copy()

        env = StockEnv(df)
        state = env.reset()
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
        agentpath = os.path.join(os.path.dirname(__file__), "dqn_model.h5")
        agent = load_agent_model(agent, agentpath)

        predictions = []
        dates = recent_df.index.strftime('%Y-%m-%d').tolist()
        actual_prices = recent_df['Close'].dropna().astype(float).squeeze().tolist()

        for _ in range(len(actual_prices)):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if env.current_step < len(df):
                price = float(df.iloc[env.current_step]['Close'])
                predictions.append(price)
            else:
                break

            state = next_state
            if done:
                break

        while len(predictions) < len(actual_prices):
            predictions.append(None)

        return jsonify({
            "dates": list(dates),
            "predictions": [float(p) if p is not None else None for p in predictions],
            "actual_prices": list(actual_prices)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
