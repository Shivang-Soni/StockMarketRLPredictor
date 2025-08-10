from data_loader import load_stock_data
from environment import StockEnv
from agent import DQNAgent, load_agent_model, save_agent_model
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def train_with_RL():
    df = load_stock_data("AAPL", "1y")
    env = StockEnv(df)
    agent = DQNAgent(env.state_dim, env.action_dim)

    log_dir = "logs"
    summary_writer = tf.summary.create_file_writer(log_dir)

    rewards = []
    for episode in range(40):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode+1} — Total Reward: {total_reward:.2f}")
                with summary_writer.as_default():
                    tf.summary.scalar("Total Reward", total_reward, step=episode)
                    tf.summary.scalar("Epsilon", agent.epsilon, step=episode)
                rewards.append(total_reward)
                break

    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    agent_path = os.path.join(os.path.dirname(__file__), "dqn_model.h5")
    print(f"Speichere Modell unter: {agent_path}")
    save_agent_model(agent,agent_path)
