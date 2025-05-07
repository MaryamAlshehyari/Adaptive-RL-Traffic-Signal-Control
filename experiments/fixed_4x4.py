from sumo_rl import SumoEnvironment
import numpy as np

# Create fixed-time SUMO environment on the 4x4 grid
env = SumoEnvironment(
    net_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.net.xml",
    route_file="/sumo-rl/sumo_rl/nets/hangzhou/hangzhou.rou.sorted.xml",
    use_gui=False,
    num_seconds=5000,
    delta_time=5,
    fixed_ts=True,            # Enable fixed-time control
    single_agent=False,       # Multi-agent setup
    add_system_info=True,
    add_per_agent_info=True
)

observations = env.reset()
done = {"__all__": False}

# Accumulators
total_waiting_times = []
total_stopped_counts = []
total_arrivals = []
step_count = 0

while not done["__all__"]:
    observations, rewards, done, info = env.step(None)

    # System-wide metrics
    total_waiting_times.append(info["system_total_waiting_time"])
    total_stopped_counts.append(info["system_total_stopped"])
    total_arrivals.append(info["system_total_arrived"])
    step_count += 1

env.close()

# Convert to arrays for stats
waiting_array = np.array(total_waiting_times)
stopped_array = np.array(total_stopped_counts)
arrived_array = np.array(total_arrivals)


# Convert to arrays for stats
arrived_array = np.array(total_arrivals)

# Final arrival count = last value
final_arrival = arrived_array[-1]

# Compute mean and median of cumulative arrival values across all steps
mean_arrival_across_steps = arrived_array.mean()
median_arrival_across_steps = np.median(arrived_array)

# Also compute arrival rate (change between steps) if needed
arrival_rate_per_step = np.diff(arrived_array)
mean_arrival_rate = np.mean(arrival_rate_per_step)
median_arrival_rate = np.median(arrival_rate_per_step)

# Print results
print(" FIXED-TIME BASELINE EVALUATION (4x4 Grid Multi-Agent)")
print(f"Total Steps: {step_count}")
print(f"Final Arrived Vehicles: {final_arrival}")
print(f"Mean Total Arrived Vehicles Across Steps: {mean_arrival_across_steps:.2f}")
print(f"Median Total Arrived Vehicles Across Steps: {median_arrival_across_steps:.2f}")
print(f"Mean Per-Step Arrival Rate: {mean_arrival_rate:.2f}")
print(f"Median Per-Step Arrival Rate: {median_arrival_rate:.2f}")
# Compute stats
print("FIXED-TIME BASELINE EVALUATION (4x4 Grid Multi-Agent)")
print(f"Total Steps: {step_count}")
print(f"Final Arrived Vehicles: {arrived_array[-1]}")
print(f"Mean Waiting Time: {waiting_array.mean():.2f}")
print(f"Median Waiting Time: {np.median(waiting_array):.2f}")
print(f"Mean Stopped Vehicles: {stopped_array.mean():.2f}")
print(f"Median Stopped Vehicles: {np.median(stopped_array):.2f}")
print(f"Mean Arrivals per Step: {np.mean(np.diff(arrived_array)):.2f}")
print(f"Median Arrivals per Step: {np.median(np.diff(arrived_array)):.2f}")
