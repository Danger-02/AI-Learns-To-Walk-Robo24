import gym
import pybullet, pybullet_envs
from stable_baselines3 import PPO
import numpy as np

model_path = "AntBulletEnv-v0_PPO_100.zip"
model = PPO.load(model_path)


env = gym.make("AntBulletEnv-v0")
env.render(mode="human")

num_episodes = 3

def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate a trained RL model on a given environment.
    Args:
        model: The trained RL model.
        env: The environment to evaluate on.
        num_episodes: Number of episodes to run the evaluation.
    Returns:
        avg_reward: The average reward over the evaluation episodes.
        rewards: A list of rewards for each episode.
    """
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward, rewards


evaluate_model(model, env, num_episodes=num_episodes)

env.close()
