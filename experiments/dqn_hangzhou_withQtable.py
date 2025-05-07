


import gym
import torch
import numpy as np
import random
from collections import deque
from sumo_rl import SumoEnvironment
import os
from torch import nn
import torch.optim as optim
# os.makedirs("/sumo-rl/outputs/qrdqn", exist_ok=True)
# ===== DQN Agent (per traffic light) =====
class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.q_network[-1].out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.q_network(state).argmax().item()

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)

        q_vals = self.q_network(s).gather(1, a).squeeze()
        with torch.no_grad():
            target = r + self.gamma * self.target_network(s_next).max(1)[0] * (1 - done)

        loss = self.criterion(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def supervised_pretrain(self, state_action_values: list, epochs=5):
        self.q_network.train()
        for epoch in range(epochs):
            random.shuffle(state_action_values)
            for state, q_vals in state_action_values:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_target = torch.FloatTensor(q_vals).unsqueeze(0)

                q_pred = self.q_network(state_tensor)
                loss = self.criterion(q_pred, q_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



# ===== Environment Setup =====
env = SumoEnvironment(
    net_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.net.xml",
    route_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.rou.sorted.xml",
    out_csv_name="/sumo-rl/outputs/dqn_with_Qtable/hangzhou/dqn_hangzhou_withqtable",
    use_gui=False,
    num_seconds=5000,
    yellow_time=3,
    min_green=5,
    delta_time=5,
    single_agent=False,  # ✅ Enable multi-agent mode
    
)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

agents = {}
action_space = env.action_space

# Create a DQN agent for each traffic light
print("action_space type:", type(env.action_space))
print("env.ts_ids:", env.ts_ids)

obs = env.reset()

# Determine if it's multi-agent
multiagent = isinstance(env.action_space, dict)

for tls_id in env.ts_ids:
    obs_dim = len(obs[tls_id])

    if multiagent:
        n_actions = env.action_space[tls_id].n  # Multi-agent
    else:
        n_actions = env.action_space.n          # Single-agent fallback

    agents[tls_id] = DQNAgent(obs_dim, n_actions)

import pickle

q_tables = {}
for tls_id in env.ts_ids:
    with open(f"/sumo-rl/outputs/ql/4x4/qtable_4x4/{tls_id}_q_table_normlized.pkl", "rb") as f:
        q_tables[tls_id] = pickle.load(f)
for tls_id in env.ts_ids:
    q_table = q_tables[tls_id]
    state_action_values = []

    
    for state, q_vector in q_table.items():
        state_vector = list(state)
        # Pad/truncate
        if len(state_vector) != obs_dim:
            print(f"⚠️ State mismatch at {tls_id}: Q-table dim {len(state_vector)} vs env dim {obs_dim}")
        if len(state_vector) < obs_dim:
            state_vector += [0.0] * (obs_dim - len(state_vector))
        elif len(state_vector) > obs_dim:
            state_vector = state_vector[:obs_dim]

        state_action_values.append((state_vector, q_vector))

    agents[tls_id].supervised_pretrain(state_action_values)
    print(f"[{tls_id}] DQN pre-trained with {len(state_action_values)} Q-table entries.")


# ===== Training Loop =====
n_episodes = 200

for episode in range(n_episodes):
    obs = env.reset()
    done = {"__all__": False}
    total_reward = {tls: 0 for tls in agents}

    while not done["__all__"]:
        actions = {
            tls_id: agents[tls_id].act(obs[tls_id])
            for tls_id in env.ts_ids
        }

        next_obs, rewards, done, _ = env.step(actions)

        for tls_id in env.ts_ids:
            agents[tls_id].remember(
                obs[tls_id],
                actions[tls_id],
                rewards[tls_id],
                next_obs[tls_id],
                done[tls_id]
            )
            total_reward[tls_id] += rewards[tls_id]

        obs = next_obs

        # Train each agent
        for tls_id in env.ts_ids:
            agents[tls_id].train()

    for tls_id in env.ts_ids:
        agents[tls_id].decay_epsilon()
        agents[tls_id].update_target()

    print(f"[Episode {episode+1}] Reward: {total_reward}")

env.close()
