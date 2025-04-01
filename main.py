import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D

# Satellite orbital motion model
def satellite_motion_model(x, dt):
    """
    Model satellite motion in a simplified orbital path
    x: state vector [x, vx, y, vy]
    dt: time step in seconds
    """
    # For a simplified orbital model, we'll use a circular orbit 
    # with slight perturbations to simulate a realistic satellite path
    
    # Current position and velocity
    pos_x, vel_x, pos_y, vel_y = x
    
    # Orbital parameters (simplified for 2D)
    center_x = 35  # Center of orbit in x (km)
    center_y = 30  # Center of orbit in y (km)
    
    # Distance from center
    rx = pos_x - center_x
    ry = pos_y - center_y
    r = np.sqrt(rx**2 + ry**2)
    
    # Orbital velocity (assuming circular orbit)
    orbital_period = 120  # seconds for complete orbit
    angular_velocity = 2 * np.pi / orbital_period  # rad/s
    
    # Calculate acceleration (centripetal)
    # a = -r * ω²
    ax = -rx * angular_velocity**2
    ay = -ry * angular_velocity**2
    
    # Apply motion equations
    new_pos_x = pos_x + vel_x * dt + 0.5 * ax * dt**2
    new_vel_x = vel_x + ax * dt
    new_pos_y = pos_y + vel_y * dt + 0.5 * ay * dt**2
    new_vel_y = vel_y + ay * dt
    
    return np.array([new_pos_x, new_vel_x, new_pos_y, new_vel_y])

# Measurement model: range, azimuth, elevation
def measurement_model(x, sensor_pos):
    # x: target state [x, vx, y, vy]
    # sensor_pos: [x, y, z] of sensor
    dx = x[0] - sensor_pos[0]
    dy = x[2] - sensor_pos[1]
    dz = 0.5 - sensor_pos[2]  # Target height - sensor height
    
    # Range
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Azimuth angle
    psi = np.arctan2(dy, dx)
    
    # Elevation angle
    theta = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    
    return np.array([r, psi, theta])

# Jacobian of the measurement model
def measurement_jacobian(x, sensor_pos):
    # x: target state [x, vx, y, vy]
    # sensor_pos: [x, y, z] of sensor
    dx = x[0] - sensor_pos[0]
    dy = x[2] - sensor_pos[1]
    dz = 0.5 - sensor_pos[2]  # Target height - sensor height
    
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_xy = np.sqrt(dx**2 + dy**2)
    
    # Partial derivatives for range
    dr_dx = dx / r
    dr_dy = dy / r
    
    # Partial derivatives for azimuth
    dpsi_dx = -dy / (dx**2 + dy**2)
    dpsi_dy = dx / (dx**2 + dy**2)
    
    # Partial derivatives for elevation
    dtheta_dx = -dx * dz / (r_xy * r**2)
    dtheta_dy = -dy * dz / (r_xy * r**2)
    
    # Assemble Jacobian matrix (3x4)
    H = np.array([
        [dr_dx, 0, dr_dy, 0],
        [dpsi_dx, 0, dpsi_dy, 0],
        [dtheta_dx, 0, dtheta_dy, 0]
    ])
    
    return H

# Simulation parameters
dt = 0.5  # Time step (s)
steps = 250  # Number of steps to simulate (~125 seconds, just over 1 orbit)
target_height = 0.5  # km
sensor_height = 0.2  # km

# Set up sensor network
# Format: [x, y, z, vx, vy, vz]
static_sensors = [
    [25, 10, sensor_height, 0, 0, 0],  # Sensor 1
    [35, 10, sensor_height, 0, 0, 0],  # Sensor 2
    [45, 10, sensor_height, 0, 0, 0]   # Sensor 3
]
# Mobile sensor with constant velocity (straight path)
mobile_sensor = [15, 10, sensor_height, 0.05, 0.05, 0]

sensors = static_sensors + [mobile_sensor]
num_sensors = len(sensors)

# Initial satellite state [x, vx, y, vy]
# Start satellite at a point in its orbit with appropriate velocity
orbital_radius = 15  # km
center_x = 35
center_y = 30
orbital_period = 120  # seconds
angular_velocity = 2 * np.pi / orbital_period

# Initial position at top of orbit
initial_angle = np.pi/4  # Start at 45 degrees
pos_x = center_x + orbital_radius * np.cos(initial_angle)
pos_y = center_y + orbital_radius * np.sin(initial_angle)

# Initial velocity (tangential to orbit)
vel_x = -orbital_radius * angular_velocity * np.sin(initial_angle)
vel_y = orbital_radius * angular_velocity * np.cos(initial_angle)

x0_true = np.array([pos_x, vel_x, pos_y, vel_y])

# Process and measurement noise
Q = 1e-6 * np.eye(4)  # Very small process noise for minor variations
R = np.diag([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  # Measurement noise

# Generate satellite trajectory
x_true = x0_true.copy()
true_trajectory = [x_true.copy()]

for t in range(steps):
    # Apply very small process noise for slight realism
    w = np.random.multivariate_normal(np.zeros(4), Q)
    
    # Update true state with satellite motion model
    x_true = satellite_motion_model(x_true, dt)
    x_true += w  # Add minimal noise
    true_trajectory.append(x_true.copy())

true_trajectory = np.array(true_trajectory)

# Initialize EKF for each sensor
x_estimates = [x0_true + np.random.randn(4) * 0.1 for _ in range(num_sensors)]  # Initial guess with small noise
P_estimates = [np.eye(4) * 1 for _ in range(num_sensors)]
est_trajectories = [[] for _ in range(num_sensors)]
measurements = [[] for _ in range(num_sensors)]

# Update sensor positions over time
sensor_positions = []
for t in range(steps + 1):
    current_positions = []
    for i, sensor in enumerate(sensors):
        # Update mobile sensor position
        if i == num_sensors - 1:  # Last sensor is mobile
            sensor[0] += sensor[3] * dt
            sensor[1] += sensor[4] * dt
        current_positions.append(sensor[:3].copy())  # Store x, y, z (using copy to avoid reference issues)
    sensor_positions.append(current_positions)

# Run EKF for each sensor
for t in range(steps):
    for i in range(num_sensors):
        # Current sensor position [x, y, z]
        sensor_pos = sensor_positions[t][i]
        
        # Generate true measurement from target to sensor
        z_true = measurement_model(true_trajectory[t+1], sensor_pos)
        
        # Add measurement noise
        v = np.random.multivariate_normal(np.zeros(3), R)
        z = z_true + v
        measurements[i].append(z)
        
        # Predict step
        x_pred = satellite_motion_model(x_estimates[i], dt)
        
        # Approximate the state transition matrix numerically
        F = np.zeros((4, 4))
        epsilon = 1e-4
        
        for j in range(4):
            x_perturbed = x_estimates[i].copy()
            x_perturbed[j] += epsilon
            x_perturbed_next = satellite_motion_model(x_perturbed, dt)
            F[:, j] = (x_perturbed_next - x_pred) / epsilon
            
        P_pred = F @ P_estimates[i] @ F.T + Q
        
        # Update step
        H = measurement_jacobian(x_pred, sensor_pos)
        z_pred = measurement_model(x_pred, sensor_pos)
        y = z - z_pred  # Innovation
        
        # Wrap angle differences to [-pi, pi]
        y[1:3] = np.arctan2(np.sin(y[1:3]), np.cos(y[1:3]))
        
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_estimates[i] = x_pred + K @ y
        P_estimates[i] = (np.eye(4) - K @ H) @ P_pred
        
        est_trajectories[i].append(x_estimates[i].copy())

# Convert to numpy arrays
est_trajectories = [np.array(traj) for traj in est_trajectories]

# Calculate consensus estimate by averaging all sensor estimates
consensus_trajectory = []
for t in range(steps):
    consensus_state = np.zeros(4)
    for i in range(num_sensors):
        consensus_state += est_trajectories[i][t]
    consensus_state /= num_sensors
    consensus_trajectory.append(consensus_state)
consensus_trajectory = np.array(consensus_trajectory)

# Extract mobile sensor path for visualization
mobile_sensor_path = np.array([pos[num_sensors-1] for pos in sensor_positions])

# Plotting the results
plt.figure(figsize=(15, 10))

# Plot the 2D trajectory
plt.subplot(2, 2, 1)
plt.plot(true_trajectory[:, 0], true_trajectory[:, 2], 'k-', linewidth=2, label='True trajectory')
# Draw orbit center
plt.plot(center_x, center_y, 'kx', markersize=8)
plt.text(center_x+1, center_y+1, 'Orbit Center')

for i in range(num_sensors):
    plt.plot(est_trajectories[i][:, 0], est_trajectories[i][:, 2], '--', label=f'Sensor {i+1} estimate')
plt.plot(consensus_trajectory[:, 0], consensus_trajectory[:, 2], 'r-', linewidth=2, label='Consensus estimate')

# Plot static sensor positions
for i, pos in enumerate(sensor_positions[0][:-1]):  # Exclude mobile sensor
    plt.plot(pos[0], pos[1], 'bo', markersize=8)
    plt.text(pos[0], pos[1], f'S{i+1}')

# Plot mobile sensor path
plt.plot(mobile_sensor_path[:, 0], mobile_sensor_path[:, 1], 'g-', linewidth=2, label='Mobile sensor path')

# Mark starting position
plt.plot(true_trajectory[0, 0], true_trajectory[0, 2], 'go', markersize=8)
plt.text(true_trajectory[0, 0], true_trajectory[0, 2], 'Start')

plt.xlabel('X position (km)')
plt.ylabel('Y position (km)')
plt.title('Satellite Tracking with Ground Sensor Network')
plt.grid(True)
plt.legend()

# Plot X position error over time
plt.subplot(2, 2, 2)
for i in range(num_sensors):
    error_x = est_trajectories[i][:, 0] - true_trajectory[1:steps+1, 0]
    plt.plot(range(steps), error_x, '--', label=f'Sensor {i+1}')
error_x_consensus = consensus_trajectory[:, 0] - true_trajectory[1:steps+1, 0]
plt.plot(range(steps), error_x_consensus, 'r-', linewidth=2, label='Consensus')
plt.xlabel('Time step')
plt.ylabel('X position error (km)')
plt.title('X Position Estimation Error')
plt.grid(True)
plt.legend()

# Plot Y position error over time
plt.subplot(2, 2, 3)
for i in range(num_sensors):
    error_y = est_trajectories[i][:, 2] - true_trajectory[1:steps+1, 2]
    plt.plot(range(steps), error_y, '--', label=f'Sensor {i+1}')
error_y_consensus = consensus_trajectory[:, 2] - true_trajectory[1:steps+1, 2]
plt.plot(range(steps), error_y_consensus, 'r-', linewidth=2, label='Consensus')
plt.xlabel('Time step')
plt.ylabel('Y position error (km)')
plt.title('Y Position Estimation Error')
plt.grid(True)
plt.legend()

# Plot velocity estimation
plt.subplot(2, 2, 4)
plt.plot(range(steps+1), np.sqrt(true_trajectory[:, 1]**2 + true_trajectory[:, 3]**2), 'k-', linewidth=2, label='True velocity')
for i in range(num_sensors):
    vel = np.sqrt(est_trajectories[i][:, 1]**2 + est_trajectories[i][:, 3]**2)
    plt.plot(range(1, steps+1), vel, '--', label=f'Sensor {i+1}')
vel_consensus = np.sqrt(consensus_trajectory[:, 1]**2 + consensus_trajectory[:, 3]**2)
plt.plot(range(1, steps+1), vel_consensus, 'r-', linewidth=2, label='Consensus')
plt.xlabel('Time step')
plt.ylabel('Velocity (km/s)')
plt.title('Target Velocity Estimation')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot true trajectory
ax.plot(true_trajectory[:, 0], true_trajectory[:, 2], [target_height] * len(true_trajectory), 'k-', linewidth=2, label='True trajectory')

# Plot orbit center
ax.scatter(center_x, center_y, target_height, c='black', marker='x', s=100)
ax.text(center_x, center_y, target_height, 'Orbit Center')

# Plot static sensor positions
for i, pos in enumerate(sensor_positions[0][:-1]):  # Exclude mobile sensor
    ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='o', s=100)
    ax.text(pos[0], pos[1], pos[2], f'S{i+1}')

# Plot mobile sensor path
ax.plot(mobile_sensor_path[:, 0], mobile_sensor_path[:, 1], mobile_sensor_path[:, 2], 'g-', linewidth=2, label='Mobile sensor path')

# Plot consensus estimate
ax.plot(consensus_trajectory[:, 0], consensus_trajectory[:, 2], [target_height] * len(consensus_trajectory), 'r-', linewidth=2, label='Consensus')

# Add measurement lines from sensors to target at select time points
visualization_times = [0, steps//2, steps-1]  # Fewer measurement lines for clarity
for t in visualization_times:
    for i in range(num_sensors):
        if t < len(true_trajectory):  # Ensure t is valid
            sensor_pos = sensor_positions[t][i]
            target_pos = [true_trajectory[t][0], true_trajectory[t][2], target_height]
            ax.plot([sensor_pos[0], target_pos[0]], 
                    [sensor_pos[1], target_pos[1]],
                    [sensor_pos[2], target_pos[2]], 'k:', alpha=0.3)

ax.set_xlabel('X position (km)')
ax.set_ylabel('Y position (km)')
ax.set_zlabel('Z position (km)')
ax.set_title('3D Visualization of Satellite Tracking with Mobile Sensor')
ax.legend()

# Set reasonable z-axis limits to better show height differences
ax.set_zlim(0, 1)

plt.tight_layout()
plt.show()

