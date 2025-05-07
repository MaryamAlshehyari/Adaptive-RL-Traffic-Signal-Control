#!/bin/bash

mkdir -p plots

# ==== QL ====
# QL - Cologne1
python plot.py -f ql/ql_2way-single-intersection_cologne1/ql_cologne_conn0_ep1.csv -t ql_cologne1 -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/ql_cologne1_waiting -xmax 50000
python plot.py -f ql/ql_2way-single-intersection_cologne1/ql_cologne_conn0_ep1.csv -t ql_cologne1 -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/ql_cologne1_arrived
python plot.py -f ql/ql_2way-single-intersection_cologne1/ql_cologne_conn0_ep1.csv -t ql_cologne1 -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/ql_cologne1_stopped -xmax 50000

# QL - Hangzhou
python plot.py -f ql/4x4/ql_hangzhou/ql-4x4grid_run3_3times_conn0_ep4.csv -t ql_hangzhou -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/ql_hangzhou_waiting
python plot.py -f ql/4x4/ql_hangzhou/ql-4x4grid_run3_3times_conn0_ep4.csv -t ql_hangzhou -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/ql_hangzhou_arrived -xmax 4500
python plot.py -f ql/4x4/ql_hangzhou/ql-4x4grid_run3_3times_conn0_ep4.csv -t ql_hangzhou -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/ql_hangzhou_stopped


# ==== DQN ====
# DQN - Cologne1
python plot.py -f dqn/cologne1/dqn_cologne1_conn0_ep5.csv -t dqn_cologne1 -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/dqn_cologne1_waiting -xmax 40000
python plot.py -f dqn/cologne1/dqn_cologne1_conn0_ep5.csv -t dqn_cologne1 -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/dqn_cologne1_arrived
python plot.py -f dqn/cologne1/dqn_cologne1_conn0_ep5.csv -t dqn_cologne1 -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/dqn_cologne1_stopped -xmax 40000

# DQN - Hangzhou
python plot.py -f dqn/hangzhou/dqn_hangzhou_conn0_ep200.csv -t dqn_hangzhou -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/dqn_hangzhou_waiting
python plot.py -f dqn/hangzhou/dqn_hangzhou_conn0_ep200.csv -t dqn_hangzhou -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/dqn_hangzhou_arrived
python plot.py -f dqn/hangzhou/dqn_hangzhou_conn0_ep200.csv -t dqn_hangzhou -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/dqn_hangzhou_stopped


# ==== DQN with Q-Table Init ====
# DQN-QTable - Cologne1
python plot.py -f dqn_with_Qtable/cologne1/dqn_cologne1_withQtable_conn0_ep500.csv -t dqnq_cologne1 -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/dqnq_cologne1_waiting
python plot.py -f dqn_with_Qtable/cologne1/dqn_cologne1_withQtable_conn0_ep500.csv -t dqnq_cologne1 -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/dqnq_cologne1_arrived
python plot.py -f dqn_with_Qtable/cologne1/dqn_cologne1_withQtable_conn0_ep500.csv -t dqnq_cologne1 -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/dqnq_cologne1_stopped

# DQN-QTable - Hangzhou
python plot.py -f dqn_with_Qtable/hangzhou/dqn_hangzhou_withqtable_conn0_ep200.csv -t dqnq_hangzhou -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/dqnq_hangzhou_waiting
python plot.py -f dqn_with_Qtable/hangzhou/dqn_hangzhou_withqtable_conn0_ep200.csv -t dqnq_hangzhou -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/dqnq_hangzhou_arrived
python plot.py -f dqn_with_Qtable/hangzhou/dqn_hangzhou_withqtable_conn0_ep200.csv -t dqnq_hangzhou -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/dqnq_hangzhou_stopped


# ==== PPO ====
# PPO - Cologne1
python plot.py -f ppo/cologne1/ppo_cologne1_final/ppo_cologne1_conn0_ep166.csv -t ppo_cologne1 -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/ppo_cologne1_waiting -xmax 1000
python plot.py -f ppo/cologne1/ppo_cologne1_final/ppo_cologne1_conn0_ep166.csv -t ppo_cologne1 -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/ppo_cologne1_arrived
python plot.py -f ppo/cologne1/ppo_cologne1_final/ppo_cologne1_conn0_ep166.csv -t ppo_cologne1 -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/ppo_cologne1_stopped -xmax 1000

# PPO - Hangzhou 
python plot.py -f ppo/hangzhou/hangzhou_1_conn0_ep26.csv -t ppo_hangzhou -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/ppo_hangzhou_waiting 
python plot.py -f ppo/hangzhou/hangzhou_1_conn0_ep26.csv -t ppo_hangzhou -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/ppo_hangzhou_arrived
python plot.py -f ppo/hangzhou/hangzhou_1_conn0_ep26.csv -t ppo_hangzhou -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/ppo_hangzhou_stopped 


# ==== QRDQN ====
# QRDQN - Cologne1
python plot.py -f qrdqn/cologne1/qrdqn_cologne1_2time_conn0_ep24.csv -t qrdqn_cologne1 -yaxis system_mean_waiting_time -ma 50 -ylabel system_mean_waiting_time -output plots/qrdqn_cologne1_waiting
python plot.py -f qrdqn/cologne1/qrdqn_cologne1_2time_conn0_ep24.csv -t qrdqn_cologne1 -yaxis system_total_arrived -ma 50 -ylabel system_total_arrived -output plots/qrdqn_cologne1_arrived
python plot.py -f qrdqn/cologne1/qrdqn_cologne1_2time_conn0_ep24.csv -t qrdqn_cologne1 -yaxis agents_total_stopped -ma 50 -ylabel agents_total_stopped -output plots/qrdqn_cologne1_stopped
