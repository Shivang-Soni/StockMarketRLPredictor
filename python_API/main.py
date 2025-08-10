# main.py
import os   
from data_loader import load_stock_data
from environment import StockEnv
from agent import DQNAgent, save_agent_model, load_agent_model
import tensorflow as tf

def train(symbol="AAPL"):
    #return ({"symbol" : symbol, "status" : "Succesful"})
    df = load_stock_data(symbol, period="1y")
    env = StockEnv(df)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = DQNAgent(state_dim, action_dim)

    episodes = 50
    summary = {
            "symbol" : symbol,
            "episode_rewards" : [],
            "total_episodes" : episodes
        }
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        summary['episode_rewards'].append(total_reward)

        print(f"Episode {episode+1}/{episodes} — Total Reward: {total_reward}")
    agent_path = os.path.join(os.path.dirname(__file__), "dqn_model.h5")
    print(f"Speichere Modell unter: {agent_path}")
    save_agent_model(agent,agent_path)
    
    return summary

if __name__ == "__main__":
    train("AAPL")  
