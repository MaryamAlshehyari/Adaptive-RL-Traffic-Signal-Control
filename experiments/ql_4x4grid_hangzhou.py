import argparse
import os
import sys

import pandas as pd

import pickle
import os
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 15
    episodes = 4

    env = SumoEnvironment(
        net_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.net.xml",
        route_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.rou.sorted.xml",
        use_gui=False,
        num_seconds=5000,
        # min_grepythonen=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(f"/sumo-rl/outputs/ql/4x4/ql_hangzhou/ql-4x4grid_run{run}_3times", episode)
            # Save Q-tables after finishing this run
        q_table_dir = f"/sumo-rl/outputs/ql/4x4/ql_hangzhou/ql-4x4grid_run{run}/q_tables"
        os.makedirs(q_table_dir, exist_ok=True)
        for ts_id, agent in ql_agents.items():
            with open(os.path.join(q_table_dir, f"{ts_id}_q_table.pkl"), "wb") as f:
                pickle.dump(agent.q_table, f)

    env.close()
