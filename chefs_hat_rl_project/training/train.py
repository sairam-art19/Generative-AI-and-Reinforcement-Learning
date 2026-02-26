import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from utils.reward_shaping import sparse_reward, shaped_reward

def create_result_dirs():
    base = os.path.join(os.path.dirname(__file__), "..", "results")

    model_dir = os.path.join(base, "model")
    csv_dir = os.path.join(base, "csv")
    plot_dir = os.path.join(base, "plot")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return model_dir, csv_dir, plot_dir


def reset_env(env):
    reset_output = env.reset()

    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def step_env(env, action):
    step_output = env.step(action)

    if len(step_output) == 5:
        next_state, reward, terminated, truncated, info = step_output
        done = terminated or truncated
    else:
        next_state, reward, done, info = step_output

    return next_state, reward, done, info


def train(episodes=500, use_shaping=False):

    model_dir, csv_dir, plot_dir = create_result_dirs()

    # Temporary stable environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    rewards_log = []

    for episode in range(episodes):

        state = reset_env(env)
        done = False
        total_reward = 0

        while not done:

            action = agent.select_action(state)
            next_state, reward, done, info = step_env(env, action)

            if use_shaping:
                reward = shaped_reward(reward, info)
            else:
                reward = sparse_reward(reward)

            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        agent.update()
        rewards_log.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")

    env.close()

    # ---------- Save CSV ----------
    df = pd.DataFrame({
        "episode": np.arange(len(rewards_log)),
        "reward": rewards_log
    })

    csv_path = os.path.join(csv_dir, "training_results.csv")
    df.to_csv(csv_path, index=False)

    # ---------- Save Plot ----------
    plt.figure()
    plt.plot(rewards_log)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve")
    plot_path = os.path.join(plot_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # ---------- Save Model ----------
    model_path = os.path.join(model_dir, "ppo_model.pt")
    agent.save(model_path)

    print("\nTraining Complete.")
    print(f"Model saved to: {model_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    train()