# CS839-Fall2025 Reinforcement Learning Assignment 1

This repo contains code and experiments for the **PointMassDiscrete-v0** environment using **DQN** and **PPO** from CleanRL.

---

## 🧩 Environment Setup

```bash
conda create -n rl python=3.10 -y
conda activate rl
pip install -r requirements.txt
````

(Optional) for logging:

```bash
wandb login
```

---

## 🚀 Run Experiments

### Main experiments

Run both DQN and PPO:

```bash
python run_all.py
```

### Hyperparameter tuning (learning rate)

Run learning rate sweeps:

```bash
python run_lr.py
```

---

All experiment results (videos, logs, and W&B runs) are automatically saved under:

```
wandb/
videos/
```

---

**Author:** Easton
**Course:** CS839 – Reinforcement Learning (Fall 2025)


