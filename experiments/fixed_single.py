from sumo_rl import SumoEnvironment

# Set up the fixed-time SUMO environment
env = SumoEnvironment(
    net_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.net.xml",
    route_file="/sumo-rl/sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml",
    use_gui=False,
    single_agent=True,
    num_seconds=3000,
    delta_time=5,
    yellow_time=3,
    min_green=5,
    max_green=60,
    fixed_ts=True  # Enables fixed-time control
)

obs, info = env.reset()
done = False

# Metrics
total_waiting_time = 0
total_stopped = 0
arrived = 0
step_count = 0

while not done:
    obs, reward, terminated, truncated, info = env.step(None)
    total_waiting_time += info["system_mean_waiting_time"]
    total_stopped += info["system_total_stopped"]
    arrived = info["system_total_arrived"]
    step_count += 1
    done = truncated

env.close()

print("FIXED-TIME BASELINE EVALUATION (Cologne1)")
print(f"Mean Waiting Time: {total_waiting_time:.2f}")
print(f"Total Stopped Vehicles: {total_stopped}")
print(f"Total Arrived Vehicles: {arrived}")
print(f"Finished at Step: {step_count}")

arrival_times = dict()
step_count = 0

while not done:
    obs, reward, terminated, truncated, info = env.step(None)
    
    sim_time = env.sim_step
    arrived_vehicles = env.sumo.simulation.getArrivedIDList()
    
    for veh in arrived_vehicles:
        arrival_times[veh] = sim_time

    step_count += 1
    done = truncated

env.close()

# Compute stats
arrival_seconds = list(arrival_times.values())

if arrival_seconds:
    mean_arrival_time = np.mean(arrival_seconds)
    median_arrival_time = np.median(arrival_seconds)
    print(f"Total Arrived Vehicles: {len(arrival_seconds)}")
    print(f"Mean Arrival Time (s): {mean_arrival_time:.2f}")
    print(f"Median Arrival Time (s): {median_arrival_time:.2f}")
else:
    print("No vehicles arrived.")
