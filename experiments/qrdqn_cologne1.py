import os
import sys

import gymnasium as gym


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from sb3_contrib import QRDQN
from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    # net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    net_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.net.xml",
    route_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml",
    single_agent=True,
    # route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    out_csv_name="/sumo-rl/outputs/qrdqn/cologne1/qrdqn_cologne1_2time",
    use_gui=False,
    num_seconds=2000,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = QRDQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    learning_starts=1000,
    buffer_size=10000,
    train_freq=1,
    target_update_interval=250,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="/sumo-rl/tensorboard_logs/ddqn_ad",

)

model.learn(total_timesteps=10000)


