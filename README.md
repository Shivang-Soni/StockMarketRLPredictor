# DQN Stock Trading Agent (TensorFlow 2.0)

This is a self-contained reinforcement learning project where I trained a Deep Q-Network (DQN) agent to simulate trading behavior on real historical stock data. The goal was to understand how RL can be applied to financial time series using minimalistic assumptions.

The entire codebase is modularized but kept in a single folder for simplicity.

## Description

The project consists of:

- Downloading real historical stock prices using the `yfinance` library
- Building a custom trading environment where the agent receives percentage daily returns as state input
- Using a basic MLP neural network as the Q-function approximator
- Training the DQN agent using experience replay and ε-greedy exploration
- Logging the performance using TensorBoard
- A Flask server to serve training and prediction as API endpoints
- A React frontend to trigger training and show prediction charts in browser

## Folder Contents

All files are placed in a single folder for convenience:

- `main.py`: Runs the training loop and sets up logging
- `env.py`: Custom trading environment with simple reward logic
- `agent.py`: DQN agent implementation and replay buffer
- `data.py`: Handles downloading and cleaning stock price data
- `logger.py`: Handles TensorBoard summaries
- `data_loader.py`: Loads data from Borse
- `app.py`: Flask API backend that exposes `/train` and `/predict` endpoints
- `frontend/`: React frontend folder with user interface
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Installation

Make sure you have Python 3.7+ and TensorFlow 2.x installed. Then run:

```bash
pip install -r requirements.txt
`