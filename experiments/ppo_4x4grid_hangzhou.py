import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl


if __name__ == "__main__":
    # Use:
    # ray[rllib]==2.7.0
    # numpy == 1.23.4
    # Pillow>=9.4.0
    # ray[rllib]==2.7.0
    # SuperSuit>=3.9.0
    # torch>=1.13.1
    # tensorflow-probability>=0.19.0
    ray.init()

    env_name = "4x4grid"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.net.xml",
                route_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.rou.sorted.xml",
                out_csv_name="/sumo-rl/outputs/ppo/hangzhou/hangzhou_1",
                use_gui=False,
                num_seconds=5000,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        # .multi_agent(
        #     policies={"default_policy"},
        #     policy_mapping_fn=mapping_fn,
        # )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
