import subprocess

# Define commands and parameters
dqn_command = (
    "python cleanrl/dqn.py --env-id PointMassDiscrete-v0 "
    "--total-timesteps 550000 --capture-video --track --wandb-project-name cs839_proj"
)

ppo_command = (
    "python cleanrl/ppo.py --env-id PointMassDiscrete-v0 "
    "--total-timesteps 1500000 --capture-video --track --wandb-project-name cs839_proj"
)

# Run for seeds 1–10
for seed in range(1, 11):
    print(f"\n🚀 Running DQN seed {seed}...")
    subprocess.run(f"{dqn_command} --seed {seed}", shell=True, check=True)

for seed in range(1, 11):
    print(f"\n🔥 Running PPO seed {seed}...")
    subprocess.run(f"{ppo_command} --seed {seed}", shell=True, check=True)

print("\n✅ All runs completed successfully!")
