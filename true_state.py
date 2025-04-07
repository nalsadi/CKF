import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
