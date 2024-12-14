import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import pandas as pd

# Parse date columns for proper time handling
date_columns = [
    'Arrival', 'Berth_claimed', 'Finished_loading',
    'Berth_released', 'Departure_Time'
]
simulation_data = pd.read_csv('2025_simulation.csv')
for col in date_columns:
    simulation_data[col] = pd.to_datetime(simulation_data[col])

# Sort events by time to ensure sequential animation
simulation_data = simulation_data.sort_values(by='Arrival')

# Prepare a time window for the animation (from first to last event)
start_time = simulation_data['Arrival'].min()
end_time = simulation_data['Departure_Time'].max()

# Generate a list of unique vessels and berths
unique_vessels = simulation_data['vessel_id'].unique()
num_vessels = len(unique_vessels)
num_berths = simulation_data['berth occupied'].max()

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, num_berths + 1)  # Berth positions
ax.set_ylim(0, num_vessels + 1)  # Vessel positions

# Add labels and legend
ax.set_title("Port Simulation Animation")
ax.set_xlabel("Berth Positions")
ax.set_ylabel("Vessels")

# Dictionary to track vessel positions and status
vessel_positions = {vessel: [0, idx + 1] for idx, vessel in enumerate(unique_vessels)}
vessel_circles = {}

# Draw initial vessel positions
for vessel, position in vessel_positions.items():
    vessel_circles[vessel] = ax.plot(position[0], position[1], 'o', label=vessel)[0]

# Animation function
def update(frame):
    current_time = start_time + pd.Timedelta(minutes=frame)
    for _, row in simulation_data.iterrows():
        vessel = row['vessel_id']
        berth = row['berth occupied']
        if row['Arrival'] <= current_time < row['Berth_claimed']:
            vessel_positions[vessel][0] = 0  # Waiting outside
        elif row['Berth_claimed'] <= current_time < row['Berth_released']:
            vessel_positions[vessel][0] = berth  # At berth
        elif current_time >= row['Departure_Time']:
            vessel_positions[vessel][0] = 0  # Departed

        # Update vessel positions on the plot
        vessel_circles[vessel].set_data(vessel_positions[vessel][0], vessel_positions[vessel][1])

# Create animation
num_frames = int((end_time - start_time).total_seconds() // 60)
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)

plt.legend(loc='upper right')
plt.show()
