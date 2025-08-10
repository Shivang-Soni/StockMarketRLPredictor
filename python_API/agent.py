import numpy as np
from model import mlp
from replay_buffer import ReplayBuffer
from tensorflow.keras.models import load_model


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(1000)
        self.model = mlp(state_dim, action_dim)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample_batch(self.batch_size)

        # Extrahiere und konvertiere alle Komponenten
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([float(t[2]) for t in batch])  # Robust gegen falsche Typen
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)  # Sicherstellen, dass bool → float geht

        # Vorhersage der Q-Werte
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        # Zielwerte aktualisieren
        targets = q_values.copy()
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Modell trainieren
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Epsilon reduzieren (Exploration vs Exploitation)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def save_agent_model(agent, filepath):
    agent.model.save(filepath, include_optimizer = False)

def load_agent_model(agent, filepath):
    agent.model = load_model(filepath, compile = False)
    return agent