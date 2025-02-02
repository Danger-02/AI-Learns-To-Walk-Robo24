import streamlit as st
import gym, pybullet_envs
import numpy as np
import cv2
from stable_baselines3 import PPO
import time
import matplotlib.pyplot as plt

model_paths = [
    "AntBulletEnv-v0_PPO_10.zip",
    "AntBulletEnv-v0_PPO_50.zip",
    "AntBulletEnv-v0_PPO_100.zip"
]
models = [PPO.load(path, device="cpu") for path in model_paths]

env = gym.make("AntBulletEnv-v0")
env.reset()

st.title("🚀 AI Learns to Walk - PyBullet RL Simulation 🐜")
st.write("AntBulletEnv-v0 is a Reinforcement Learning environment based on the popular PyBullet physics engine. It's a variant of the 'Ant' environment, where an AI agent learns to control a simulated quadrupedal robot (often called the 'Ant') to walk through trial and error.")
st.write("Below Button will show the trained Ant Reinforcement Model at Initial, Intermediate and Final Stage")

frame_placeholders = [st.empty() for _ in range(3)]
reward_metrics = [st.empty() for _ in range(3)]

rewards_list = []

def run_model(model, frame_placeholder, reward_placeholder, model_name):
    obs = env.reset()
    total_reward = 0

    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated = env.step(action)
        total_reward += reward
        
        frame = env.render(mode="rgb_array")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_LINEAR)
        frame_placeholder.image(frame, caption=f"{model_name} Simulation", channels="BGR", use_container_width=True)
        
        if terminated or truncated:
            break
        
        time.sleep(0.03)
    
    reward_placeholder.metric(f"{model_name} Total Reward", f"{total_reward:.2f}")

    rewards_list.append(total_reward)


if st.button("Start Simulation"):

    for i, model in enumerate(models):
        model_name = ["Initial Model", "50% Trained Model", "Fully Trained Model"][i]
        run_model(model, frame_placeholders[i], reward_metrics[i], model_name)

    env.close()

    st.subheader("Model Performance Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Model Avg Reward", f"{rewards_list[0]:.2f}")
    col2.metric("50% Trained Model Avg Reward", f"{rewards_list[1]:.2f}")
    col3.metric("Fully Trained Model Avg Reward", f"{rewards_list[2]:.2f}")

    st.subheader("Reward Progression")
    fig, ax = plt.subplots()
    ax.bar(["Initial", "Mid", "Final"], rewards_list, color=["red", "orange", "green"])
    ax.set_ylabel("Average Reward")
    ax.set_title("Comparison of Model Performance")
    st.pyplot(fig)