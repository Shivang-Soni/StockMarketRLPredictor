import numpy as np
import pandas as pd

class StockEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_step = len(df) - 1
        self.current_step = 0
        self.state_dim = 5
        self.action_dim = 3
        self.action_space = [0, 1, 2]
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_buy_cost = 0.0

    def get_state(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.state_dim, dtype=np.float32)
        state = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values
        return state.astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_buy_cost = 0.0
        return self.get_state()

    def step(self, action):
        if self.current_step >= len(self.df):
            raise IndexError("current_step is out of bounds")

        price = float(self.df.iloc[self.current_step]['Close'].item())

        reward = 0

        # Buy
        if action == 1 and self.balance >= price:
            self.shares_held += 1
            self.balance -= price
            self.total_shares_bought += 1
            self.total_buy_cost += price

        # Sell
        elif action == 2 and self.shares_held > 0:
            avg_buy_price = self.total_buy_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
            reward = price - avg_buy_price
            self.shares_held -= 1
            self.balance += price
            self.total_shares_bought -= 1
            self.total_buy_cost -= avg_buy_price

        self.current_step += 1
        done = self.current_step >= self.n_step

        return self.get_state(), reward, done, {}
