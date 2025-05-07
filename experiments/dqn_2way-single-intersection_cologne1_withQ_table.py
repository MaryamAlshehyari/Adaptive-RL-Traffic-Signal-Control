import os
import sys
import pickle

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment
import torch


# === Load the normalized Q-table ===
q_table_path = "/sumo-rl/outputs/ql_2way-single-intersection_cologne1/2025-05-04 22:42:06_alpha0.1_gamma0.99_eps0.05_decay1.0_rewardwait/q_tables/normalized_q_table.pkl"
with open(q_table_path, "rb") as f:
    q_table = pickle.load(f)

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.net.xml",
        route_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml",
        out_csv_name="/sumo-rl/outputs/dqn_with_Qtable/cologne1/dqn_cologne1_withQtable",
        single_agent=True,
        use_gui=False,
        num_seconds=1000,
    )

    # === Create DQN model ===
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.1,   #  Higher epsilon for early exploration
        exploration_final_eps=0.05,
        verbose=1,
    )
    import numpy as np

    policy_net = model.q_net
    device = policy_net.device

    # Inject as many as we can fit (assume same action space)
    injected = 0
    for state, q_values in q_table.items():
        if isinstance(q_values, list):
            q_values = torch.tensor(q_values, dtype=torch.float32).to(device)
            if len(q_values) == model.action_space.n:
                with torch.no_grad():
                    model.q_net.q_net[-1].bias.data = q_values
                    injected += 1
        if injected >= 1:  # inject once globally (since we can't map states to NN input directly)
            break

    print(f"Injected 1 Q-table vector as initial bias in output layer")



    print(f" Injected Q-values into DQN for {injected} states")

    # === Train DQN ===
    model.learn(total_timesteps=100000)
