import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv  # Add import for CSV file handling
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)

# Simulation parameters
tf = 400  # Total simulation time (seconds)
dt = 0.5  # Time interval (seconds)
t = np.arange(0, tf + dt, dt)
n = 4  # State dimension
m = 3  # Measurement dimension

# Object parameters
target_height = 0.5  # km
sensor_height = 0.2  # km
turning_speed = np.deg2rad(0.5)  # 0.5 degrees/s

# Initial object state (position in km, velocity in km/s)
x0_true = np.array([2000, 0, 10000, -40], dtype=np.float32)  # Initial position and velocity

# Set up sensor network
# Static sensor positions
static_sensors = [
    [2500, 1000, sensor_height],  # Sensor 1
    [3500, 1000, sensor_height],  # Sensor 2
    [4500, 1000, sensor_height],  # Sensor 3
]
# Mobile sensor initial position and velocity
mobile_sensor = [1500, 0, sensor_height]  # Sensor 4
final_mobile_sensor_pos = [5000, 6000]  # Final destination for the mobile sensor
mobile_sensor_vel = [
    (final_mobile_sensor_pos[0] - mobile_sensor[0]) / tf,  # Velocity in X
    (final_mobile_sensor_pos[1] - mobile_sensor[1]) / tf,  # Velocity in Y
    0  # No change in Z
]
sensor_positions = [static_sensors + [mobile_sensor]]
num_nodes = len(static_sensors) + 1

# Calculate positions of all sensors at each time step
for k in range(1, len(t)):
    current_positions = []
    for i, sensor in enumerate(sensor_positions[-1]):
        if i == num_nodes - 1:  # Mobile sensor
            new_pos = [
                sensor[0] + mobile_sensor_vel[0] * dt,
                sensor[1] + mobile_sensor_vel[1] * dt,
                sensor[2]
            ]
            current_positions.append(new_pos)
        else:  # Static sensors
            current_positions.append(sensor.copy())
    sensor_positions.append(current_positions)

mobile_sensor_trajectory = np.array([sensor_positions[k][num_nodes - 1] for k in range(len(t))])

# Process and measurement noise
Q = np.eye(n) * 1e-5  # Process noise covariance as specified: 10^-5 * I
R_base = np.diag([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2]).astype(np.float32)

# Generate measurement noise for each node's sensors
base_var = np.array([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  
node_variances = [base_var for _ in range(num_nodes)]
R_list = [np.diag(var) for var in node_variances]

# Object motion model function with slower movement and exact final destination
def object_motion_model(x, dt):
    # Linear motion model
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    
    # Turning points and final destination
    turn_point1 = np.array([2000, 4000])
    turn_point2 = np.array([4000, 2500])
    final_dest = np.array([4000, -1000])
    
    current_pos = np.array([x[0], x[2]])
    
    # Check if we've passed the turning points
    passed_turn1 = x[2] <= 4000 and current_pos[0] >= 2000
    passed_turn2 = passed_turn1 and current_pos[0] >= 4000 and current_pos[1] <= 2500
    near_final = np.linalg.norm(current_pos - final_dest) < 300
    
    # Snap to final destination if close enough
    if near_final:
        return np.array([final_dest[0], 0, final_dest[1], 0])  # Stop at final destination
    
    # Smoothly approach the second turning point
    if passed_turn1 and not passed_turn2:
        direction = turn_point2 - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        speed = 30  # Slower constant speed between turning points
        new_vx = direction[0] * speed
        new_vy = direction[1] * speed
        return np.array([x[0] + new_vx * dt, new_vx, x[2] + new_vy * dt, new_vy])
    
    # Smoothly approach the first turning point
    elif not passed_turn1:
        direction = turn_point1 - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        speed = 30  # Slower constant speed toward the first turning point
        new_vx = direction[0] * speed
        new_vy = direction[1] * speed
        return np.array([x[0] + new_vx * dt, new_vx, x[2] + new_vy * dt, new_vy])
    
    # After the second turning point, move toward the final destination
    elif passed_turn2:
        direction = final_dest - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        speed = 30  # Slower constant speed toward the final destination
        new_vx = direction[0] * speed
        new_vy = direction[1] * speed
        return np.array([x[0] + new_vx * dt, new_vx, x[2] + new_vy * dt, new_vy])
    
    # Fallback to standard motion model
    return F @ x

# Measurement model as specified in equation (73)
def measurement_model(x, sensor_pos):
    dx = x[0] - sensor_pos[0]  # ξk - x^i_k
    dy = x[2] - sensor_pos[1]  # ηk - y^i_k
    h = target_height - sensor_pos[2]  # Height difference
    
    # Distance calculation: sqrt((ξk - x^i_k)^2 + (ηk - y^i_k)^2 + h^2)
    d_squared = dx**2 + dy**2
    r = np.sqrt(d_squared + h**2)
    
    # Azimuth angle: arctan((ηk - y^i_k)/(ξk - x^i_k))
    psi = np.arctan2(dy, dx)
    
    # Pitch angle: arctan(h/sqrt((ξk - x^i_k)^2 + (ηk - y^i_k)^2))
    theta = np.arctan2(h, np.sqrt(d_squared))
    
    return np.array([r, psi, theta])

# Generate true object trajectory
x = np.zeros((n, len(t)))
x[:, 0] = x0_true
z = np.zeros((m, len(t), num_nodes))
w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T

for k in range(1, len(t)):
    # State propagation with process noise
    x[:, k] = object_motion_model(x[:, k-1], dt) + w[:, k]
    
    # Generate measurements for each sensor
    for node in range(num_nodes):
        sensor_pos = sensor_positions[k][node]
        true_measurement = measurement_model(x[:, k], sensor_pos)
        v = np.random.multivariate_normal(mean=np.zeros(m), cov=R_list[node])
        z[:, k, node] = true_measurement + v

# Visualization of results

# 2D Visualization (Figure 2 in the paper)
plt.figure(figsize=(12, 10))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='Target Trajectory')

# Add markers at regular intervals to show progression
marker_indices = np.arange(0, len(t), 60)
plt.scatter(x[0, marker_indices], x[2, marker_indices], c='black', s=30, alpha=0.7)

# Highlight start, turning points, and final destination
plt.scatter(x[0, 0], x[2, 0], c='blue', s=200, marker='o', label='Start')
plt.scatter(2000, 4000, c='red', marker='*', s=400, edgecolors='black', label='Turning Point 1')
plt.scatter(4000, 2500, c='red', marker='*', s=400, edgecolors='black', label='Turning Point 2')
plt.scatter(4000, -1000, c='green', marker='D', s=200, label='Final Destination')

# Draw circles around turning points and final destination
circle1 = plt.Circle((2000, 4000), 300, color='r', fill=False, linestyle='--', alpha=0.5)
circle2 = plt.Circle((4000, 2500), 300, color='r', fill=False, linestyle='--', alpha=0.5)
circle3 = plt.Circle((4000, -1000), 300, color='g', fill=False, linestyle='--', alpha=0.5)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)

# Show start and end points prominently
plt.scatter(x[0, 0], x[2, 0], c='blue', s=200, marker='o', label='Start')
plt.scatter(x[0, -1], x[2, -1], c='purple', s=200, marker='X', label='End')

# Mark turning points with larger markers
plt.scatter(2000, 4000, c='blue', marker='*', s=300, label='Turning Point 1')
plt.scatter(4000, 2500, c='blue', marker='*', s=300, label='Turning Point 2')

# Plot static sensors
for i, sensor in enumerate(static_sensors):
    plt.scatter(sensor[0], sensor[1], c='green', marker='^', s=100, label=f'Sensor {i+1}')

# Plot mobile sensor trajectory
plt.plot(mobile_sensor_trajectory[:, 0], mobile_sensor_trajectory[:, 1], 'g--', linewidth=2, label='Sensor 4 (Mobile)')

plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('2D Visualization of Target Tracking Scenario')
plt.grid(True)
plt.legend()
plt.savefig('target_tracking_2d.png', dpi=300)
plt.show()

# 3D Visualization (Figure 3 in the paper)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot static sensors
for i, sensor in enumerate(static_sensors):
    ax.scatter(sensor[0], sensor[1], sensor[2], c='green', marker='^', s=100, label=f'Sensor {i+1}')

# Plot mobile sensor trajectory
ax.plot(mobile_sensor_trajectory[:, 0], mobile_sensor_trajectory[:, 1], mobile_sensor_trajectory[:, 2], 
        c='green', linestyle='--', label='Sensor 4 (Mobile)', linewidth=2)

# Plot target trajectory
ax.plot(x[0, :], x[2, :], target_height * np.ones(len(t)), 
        c='red', label='Target Trajectory', linewidth=2)

# Add labels and legend
ax.set_xlabel('X Position (km)')
ax.set_ylabel('Y Position (km)')
ax.set_zlabel('Z Position (km)')
ax.set_title('3D Visualization of Target Tracking Scenario')
ax.legend()
ax.grid(True)
plt.savefig('target_tracking_3d.png', dpi=300)
plt.show()

# Azimuth angles visualization (Figure 6 in the paper)
plt.figure(figsize=(10, 6))
for i in range(num_nodes):
    plt.plot(t, np.rad2deg(z[1, :, i]), label=f'Sensor {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Azimuth Angle (degrees)')
plt.title('Azimuth Angles Measured by Sensors')
plt.legend()
plt.grid(True)
plt.savefig('azimuth_angles.png', dpi=300)
plt.show()

# Save the simulation data for filter implementation
np.savez('simulation_data.npz', 
         time=t, 
         true_states=x, 
         measurements=z, 
         sensor_positions=sensor_positions,
         process_noise_cov=Q,
         measurement_noise_cov=R_list)

from ekf import EKF

# Consensus weight matrix (ensure it is a proper 2D array)
consensus_weights = np.array([
    [0.5, 0.25, 0.25, 0],
    [0.25, 0.5, 0.25, 0],
    [0.25, 0.25, 0.5, 0],
    [0, 0, 0, 1]
])

# Initialize EKFs for each sensor
ekfs = [EKF(Q, R_list[i], dt) for i in range(num_nodes)]

# Arrays to store estimates
ekf_estimates = np.zeros((num_nodes, n, len(t)))
consensus_estimates = np.zeros((n, len(t)))

# Run EKFs with consensus
for k in range(len(t)):
    ekf_pos = []
    neighbors_info = []

    for i in range(num_nodes):
        ekfs[i].predict()
        if k > 0:  # Start updating after first prediction
            ekfs[i].update(z[:, k, i], sensor_positions[k][i])
        ekf_estimates[i, :, k] = ekfs[i].x
        ekf_pos.append((ekfs[i].x, ekfs[i].P))

    # Apply consensus algorithm
    for i in range(num_nodes):
        neighbors_info = [(ekf_pos[j][0], ekf_pos[j][1]) for j in range(num_nodes) if j != i]
        ekfs[i].consensus_update(neighbors_info, consensus_weights[i])  # Pass the correct row of weights

    # Compute consensus state as the average of all nodes
    consensus_state = np.mean([ekfs[i].x for i in range(num_nodes)], axis=0)
    consensus_estimates[:, k] = consensus_state

# Plot comparison of individual EKF estimates before consensus
plt.figure(figsize=(12, 10))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')

for i in range(num_nodes):
    plt.plot(ekf_estimates[i, 0, :], ekf_estimates[i, 2, :], '--', alpha=0.7, label=f'EKF {i+1} Estimate')

plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Target Tracking: Individual EKF Estimates Before Consensus')
plt.grid(True)
plt.legend()
plt.savefig('ekf_individual_estimates.png', dpi=300)
plt.show()

# Plot comparison
plt.figure(figsize=(12, 10))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')
plt.plot(consensus_estimates[0, :], consensus_estimates[2, :], 'b--', linewidth=2, label='Consensus EKF Estimate')

for i in range(num_nodes):
    plt.plot(ekf_estimates[i, 0, :], ekf_estimates[i, 2, :], ':', alpha=0.5, label=f'EKF {i+1}')

# Add sensors and other plot elements
for i, sensor in enumerate(static_sensors):
    plt.scatter(sensor[0], sensor[1], c='green', marker='^', s=100, label=f'Sensor {i+1}')

plt.plot(mobile_sensor_trajectory[:, 0], mobile_sensor_trajectory[:, 1], 'g--', linewidth=2, label='Mobile Sensor')

plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Target Tracking: True vs Estimated Trajectories (Consensus EKF)')
plt.grid(True)
plt.legend()
plt.savefig('consensus_ekf_comparison.png', dpi=300)
plt.show()

# Save azimuth angles to a CSV file
azimuth_file = 'azimuth_angles.csv'
with open(azimuth_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)'] + [f'Sensor {i+1}' for i in range(num_nodes)])
    for k in range(len(t)):
        writer.writerow([t[k]] + list(np.rad2deg(z[1, k, :])))

# Save EKF estimates to individual CSV files
for i in range(num_nodes):
    ekf_file = f'ekf_estimates_sensor_{i+1}.csv'
    with open(ekf_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'X Position (km)', 'X Velocity (km/s)', 'Y Position (km)', 'Y Velocity (km/s)'])
        for k in range(len(t)):
            writer.writerow([t[k]] + list(ekf_estimates[i, :, k]))

# Save consensus EKF estimates to a CSV file
consensus_file = 'consensus_ekf_estimates.csv'
with open(consensus_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'X Position (km)', 'X Velocity (km/s)', 'Y Position (km)', 'Y Velocity (km/s)'])
    for k in range(len(t)):
        writer.writerow([t[k]] + list(consensus_estimates[:, k]))

# Create a custom UKF class to handle changing measurement functions
class CustomUKF:
    def __init__(self, dim_x, dim_z, dt, Q, R, initial_x=None):
        # Create sigma points
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=0)
        
        # Initialize the filter
        self.ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=self.fx, hx=self.hx, points=points)
        self.ukf.Q = Q
        self.ukf.R = R
        
        # Set initial state and covariance
        if initial_x is None:
            self.ukf.x = np.array([x0_true[0], 0, x0_true[2], 0])  # Use true initial state with zero velocity
        else:
            self.ukf.x = initial_x
        self.ukf.P = np.eye(dim_x) * 10.0  # Higher initial uncertainty
        
        # Store sensor position
        self.sensor_pos = np.zeros(3)
        
    def fx(self, x, dt):
        return object_motion_model(x, dt)
    
    def hx(self, x):
        return measurement_model(x, self.sensor_pos)
    
    def update_sensor_position(self, pos):
        self.sensor_pos = np.array(pos)
    
    def predict(self):
        self.ukf.predict()
    
    def update(self, z):
        self.ukf.update(z)
    
    @property
    def x(self):
        return self.ukf.x
    
    @property
    def P(self):
        return self.ukf.P
    
    # Add consensus update method similar to EKF
    def consensus_update(self, neighbors_info, weights):
        # Extract own state and covariance
        x_i = self.ukf.x
        P_i = self.ukf.P
        
        # Initialize weighted sum
        x_weighted_sum = weights[0] * x_i
        P_inv_weighted_sum = weights[0] * np.linalg.inv(P_i)
        
        # Iterate through neighbors
        for j, (x_j, P_j) in enumerate(neighbors_info):
            x_weighted_sum += weights[j+1] * x_j
            P_inv_weighted_sum += weights[j+1] * np.linalg.inv(P_j)
        
        # Update state and covariance
        self.ukf.P = np.linalg.inv(P_inv_weighted_sum)
        self.ukf.x = self.ukf.P @ (P_inv_weighted_sum @ x_weighted_sum)

# Initialize UKFs for each sensor with proper initial state
initial_state = np.array([x0_true[0], 0, x0_true[2], 0])  # Start with position and zero velocities
ukfs = [CustomUKF(n, m, dt, Q, R_list[i], initial_state) for i in range(num_nodes)]

# Arrays to store UKF estimates
ukf_estimates = np.zeros((num_nodes, n, len(t)))
ukf_consensus_estimates = np.zeros((n, len(t)))

# Run UKFs with consensus
for k in range(len(t)):
    ukf_pos = []

    for i in range(num_nodes):
        # Update the sensor position
        ukfs[i].update_sensor_position(sensor_positions[k][i])
        
        # Predict step
        ukfs[i].predict()
        
        # Update step - only update after first time step to allow prediction to run first
        if k > 0:  
            ukfs[i].update(z[:, k, i])
            
        # Store estimates
        ukf_estimates[i, :, k] = ukfs[i].x
        ukf_pos.append((ukfs[i].x, ukfs[i].P))

    # Apply consensus algorithm
    for i in range(num_nodes):
        # Extract weights for this node
        node_weights = consensus_weights[i]
        # Get neighbors info (excluding self)
        neighbors_info = [(ukf_pos[j][0], ukf_pos[j][1]) for j in range(num_nodes) if j != i]
        # Apply consensus
        ukfs[i].consensus_update(neighbors_info, node_weights)

    # Compute consensus state as the average of all nodes
    consensus_state = np.mean([ukfs[i].x for i in range(num_nodes)], axis=0)
    ukf_consensus_estimates[:, k] = consensus_state

# Plot comparison of individual UKF estimates before consensus
plt.figure(figsize=(12, 10))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')

for i in range(num_nodes):
    plt.plot(ukf_estimates[i, 0, :], ukf_estimates[i, 2, :], '--', alpha=0.7, label=f'UKF {i+1} Estimate')

plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Target Tracking: Individual UKF Estimates Before Consensus')
plt.grid(True)
plt.legend()
plt.savefig('ukf_individual_estimates.png', dpi=300)
plt.show()

# Plot comparison of UKF consensus estimates
plt.figure(figsize=(12, 10))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')
plt.plot(ukf_consensus_estimates[0, :], ukf_consensus_estimates[2, :], 'b--', linewidth=2, label='Consensus UKF Estimate')

for i in range(num_nodes):
    plt.plot(ukf_estimates[i, 0, :], ukf_estimates[i, 2, :], ':', alpha=0.5, label=f'UKF {i+1}')

# Add sensors and other plot elements
for i, sensor in enumerate(static_sensors):
    plt.scatter(sensor[0], sensor[1], c='green', marker='^', s=100, label=f'Sensor {i+1}')

plt.plot(mobile_sensor_trajectory[:, 0], mobile_sensor_trajectory[:, 1], 'g--', linewidth=2, label='Mobile Sensor')

plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Target Tracking: True vs Estimated Trajectories (Consensus UKF)')
plt.grid(True)
plt.legend()
plt.savefig('consensus_ukf_comparison.png', dpi=300)
plt.show()

# Save UKF estimates to individual CSV files
for i in range(num_nodes):
    ukf_file = f'ukf_estimates_sensor_{i+1}.csv'
    with open(ukf_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'X Position (km)', 'X Velocity (km/s)', 'Y Position (km)', 'Y Velocity (km/s)'])
        for k in range(len(t)):
            writer.writerow([t[k]] + list(ukf_estimates[i, :, k]))

# Save consensus UKF estimates to a CSV file
ukf_consensus_file = 'consensus_ukf_estimates.csv'
with open(ukf_consensus_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'X Position (km)', 'X Velocity (km/s)', 'Y Position (km)', 'Y Velocity (km/s)'])
    for k in range(len(t)):
        writer.writerow([t[k]] + list(ukf_consensus_estimates[:, k]))

# Calculate RMSE for EKF and UKF
def calculate_rmse(true_states, estimates):
    errors = true_states - estimates
    mse = np.mean(errors**2, axis=1)
    rmse = np.sqrt(mse)
    return rmse

# Compute RMSE for EKF and UKF consensus estimates
rmse_ekf = calculate_rmse(x, consensus_estimates)
rmse_ukf = calculate_rmse(x, ukf_consensus_estimates)

# Print RMSE results
print("RMSE for EKF Consensus:")
print(f"X Position: {rmse_ekf[0]:.4f}, X Velocity: {rmse_ekf[1]:.4f}, Y Position: {rmse_ekf[2]:.4f}, Y Velocity: {rmse_ekf[3]:.4f}")

print("RMSE for UKF Consensus:")
print(f"X Position: {rmse_ukf[0]:.4f}, X Velocity: {rmse_ukf[1]:.4f}, Y Position: {rmse_ukf[2]:.4f}, Y Velocity: {rmse_ukf[3]:.4f}")

# Compute mean RMSE across all state dimensions
mean_rmse_ekf = np.mean(rmse_ekf)
mean_rmse_ukf = np.mean(rmse_ukf)

# Print final mean RMSE
print(f"Final Mean RMSE for EKF Consensus: {mean_rmse_ekf:.4f}")
print(f"Final Mean RMSE for UKF Consensus: {mean_rmse_ukf:.4f}")

# Plot RMSE comparison
plt.figure(figsize=(10, 6))
plt.bar(['X Position', 'X Velocity', 'Y Position', 'Y Velocity'], rmse_ekf, alpha=0.6, label='EKF RMSE')
plt.bar(['X Position', 'X Velocity', 'Y Position', 'Y Velocity'], rmse_ukf, alpha=0.6, label='UKF RMSE')
plt.ylabel('RMSE')
plt.title('RMSE Comparison: EKF vs UKF')
plt.legend()
plt.grid(True)
plt.savefig('rmse_comparison.png', dpi=300)
plt.show()

# Calculate RMSE over time for EKF and UKF
rmse_ekf_over_time = np.sqrt((x - consensus_estimates)**2)
rmse_ukf_over_time = np.sqrt((x - ukf_consensus_estimates)**2)

# Save RMSE over time for each state to individual CSV files
states = ['X Position', 'X Velocity', 'Y Position', 'Y Velocity']
for i, state in enumerate(states):
    # EKF RMSE
    ekf_rmse_file = f'ekf_rmse_{state.lower().replace(" ", "_")}.csv'
    with open(ekf_rmse_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'RMSE'])
        for k in range(len(t)):
            writer.writerow([t[k], rmse_ekf_over_time[i, k]])
    
    # UKF RMSE
    ukf_rmse_file = f'ukf_rmse_{state.lower().replace(" ", "_")}.csv'
    with open(ukf_rmse_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'RMSE'])
        for k in range(len(t)):
            writer.writerow([t[k], rmse_ukf_over_time[i, k]])

# Calculate mean RMSE per time step for EKF and UKF
mean_rmse_ekf_per_timestep = np.mean(rmse_ekf_over_time, axis=0)
mean_rmse_ukf_per_timestep = np.mean(rmse_ukf_over_time, axis=0)

# Save mean RMSE per time step to CSV files
mean_rmse_ekf_file = 'mean_rmse_ekf_per_timestep.csv'
with open(mean_rmse_ekf_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Mean RMSE'])
    for k in range(len(t)):
        writer.writerow([t[k], mean_rmse_ekf_per_timestep[k]])

mean_rmse_ukf_file = 'mean_rmse_ukf_per_timestep.csv'
with open(mean_rmse_ukf_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Mean RMSE'])
    for k in range(len(t)):
        writer.writerow([t[k], mean_rmse_ukf_per_timestep[k]])
