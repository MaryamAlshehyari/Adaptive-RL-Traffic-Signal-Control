import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_rl import SumoEnvironment

# Make sure SUMO_HOME is properly set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Setup SUMO-RL environment (single-agent for the main traffic light)
env = DummyVecEnv([
    lambda: SumoEnvironment(
        net_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.net.xml",
        route_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml",
        use_gui=False,
        num_seconds=3000,
        single_agent=True,          # Specify the single TLS
        out_csv_name="/sumo-rl/outputs/ppo/cologne1/ppo_cologne1_final/ppo_cologne1",  # Output directory for logs
        delta_time=5,  # Action frequency
        yellow_time=3,
        min_green=5,
        max_green=60,
        # reward_fn = -sum(vehicle_waiting_times),

    )
])

# Create PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="/sumo-rl/tensorboard_logs/ppo_cologne1_1",
)

# Train the model
model.learn(total_timesteps=5000)

# Save the model
model.save("ppo_model_cologne1")
  