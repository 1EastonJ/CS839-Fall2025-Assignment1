import subprocess

# Define the algorithms and their base commands
base_commands = {
    "dqn": (
        "python cleanrl/dqn.py --env-id PointMassDiscrete-v0 "
        "--total-timesteps 550000 --capture-video --track --wandb-project-name cs839_proj_hyper "
    ),
    "ppo": (
        "python cleanrl/ppo.py --env-id PointMassDiscrete-v0 "
        "--total-timesteps 1500000 --capture-video --track --wandb-project-name cs839_proj_hyper "
    ),
}

# Choose the hyperparameter you want to sweep
# Example: learning rate
learning_rates_dqn = [1e-4, 5e-4, 1e-3]
learning_rates_ppo = [1e-5, 3e-5, 1e-4]

# Use a single fixed seed for all runs
seed = 1

# Loop over algorithms and parameter values
for algo, base_cmd in base_commands.items():
    if algo == "dqn":
        for lr in learning_rates_dqn:
            print(f"\n🚀 Running DQN with lr={lr}, seed={seed}")
            cmd = f"{base_cmd} --learning-rate {lr} --seed {seed}"
            subprocess.run(cmd, shell=True, check=True)

    elif algo == "ppo":
        for lr in learning_rates_ppo:
            print(f"\n🔥 Running PPO with lr={lr}, seed={seed}")
            cmd = f"{base_cmd} --learning-rate {lr} --seed {seed}"
            subprocess.run(cmd, shell=True, check=True)

print("\n✅ Hyperparameter sweep complete!")
