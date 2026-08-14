# AI Learns to Walk - Reinforcement Learning for Quadruped Locomotion

Training and visualizing a physics-based quadruped locomotion agent using Deep Reinforcement Learning.

[![Watch the demo](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://youtu.be/B8nOpcykLDk?si=asJCUD4xz6yJCcg2)
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://www.python.org/)
[![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![PyBullet](https://img.shields.io/badge/PyBullet-Physics-lightgrey)](https://pybullet.org/)

---

## Overview :

This project trains an AI agent to walk from scratch using **Proximal Policy Optimization (PPO)** in the **AntBulletEnv-v0** physics simulation environment (PyBullet). The agent starts with no knowledge of locomotion and, through iterative reinforcement learning, learns to coordinate its legs to move forward efficiently.

The trained policy is then loaded into an interactive **Streamlit** dashboard, where the agent's walking behavior can be visualized and monitored in real time.

This project was developed and presented at the **Robotics Club, MNNIT Allahabad**. A demo video is linked above.

---

## 🚀 Features

- **Reinforcement Learning** — Agent trained using PPO (Stable-Baselines3) on a continuous-control locomotion task.
- **Physics-Based Simulation** — Realistic quadruped dynamics via PyBullet's AntBulletEnv-v0.
- **Interactive Dashboard** — Streamlit interface to visualize and monitor the trained agent's performance.
- **Configurable Training** — Hyperparameters, episode count, and training duration are easily adjustable.
- **Checkpointing** — Automatically saves the best-performing model across training episodes.

---

## 📊 Results

The agent was trained for 100 episodes (~10,240 timesteps per episode, ~1M total timesteps). Mean episodic reward improved substantially over the course of training:

| Stage | Episode | Mean Reward |
|---|---|---|
| Initial | 1 | ~11 |
| Mid-training | 44 | ~13,700 |
| Best performance | 95 | ~27,250 |

Policy convergence was tracked using standard PPO diagnostics: KL divergence, clip fraction, explained variance, and value loss.

---

## 📦 Dependencies

- Python 3.9
- [OpenAI Gym](https://github.com/openai/gym)
- [PyBullet](https://pybullet.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

Install all dependencies:

```bash
pip install gym pybullet pybullet_envs stable-baselines3 streamlit tqdm
```

> **Note:** This project uses the legacy OpenAI Gym API. Stable-Baselines3 will automatically wrap it in a compatibility layer; migrating to Gymnasium is recommended for future development.

---

## 🛠️ Usage

### Train the agent

```bash
python train.py
```

This runs 100 training episodes, evaluates the policy after each one, and saves the best-performing model as `AntBulletEnv-v0_PPO_Best`.

### Launch the visualization dashboard

```bash
streamlit run app.py
```

Load a saved model checkpoint to watch the trained agent walk in real time.

---

## 🎥 Demo

Watch the trained agent in action: **[YouTube Demo](https://youtu.be/B8nOpcykLDk?si=asJCUD4xz6yJCcg2)**

Presented at the **Robotics Club, MNNIT Allahabad**.

---

## 📄 License

This project is open source and available for educational and research purposes.
