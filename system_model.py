import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Random seed for reproducibility
np.random.seed(42)

# Final simulation time and radar sample rate
tf = 500
T = 5
t = np.arange(0, tf + T, T)
n = 4  # Number of states
m = 3  # Number of measurements: [r, psi, theta]

# Initial state
x_true = np.array([[25e3], [-120], [10e3], [0]])
x_kf = x_true.copy()


# Place sensors along the y = 0 axis
sensor_positions = [
    [-0.5e4, 0, 0],  # Moving sensor initial position
    [0, 0, 0],       # First stationary sensor
    [0.5e4, 0, 0],   # Second stationary sensor
    [1.0e4, 0, 0],   # Third stationary sensor
]

# Uniform motion model
UM = np.array([
    [1, T, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, T],
    [0, 0, 0, 1]
])

# Coordinated turn model
def CT(w):
    if abs(w) < 1e-6:
        return UM
    else:
        return np.array([
            [1, np.sin(w * T) / w, 0, -(1 - np.cos(w * T)) / w],
            [0, np.cos(w * T), 0, -np.sin(w * T)],
            [0, (1 - np.cos(w * T)) / w, 1, np.sin(w * T) / w],
            [0, np.sin(w * T), 0, np.cos(w * T)]
        ])

# Noise covariances
L1 = 0.16
Q1 = L1 * np.array([
    [T**3 / 3, T**2 / 2, 0, 0],
    [T**2 / 2, T, 0, 0],
    [0, 0, T**3 / 3, T**2 / 2],
    [0, 0, T**2 / 2, T]
])
R = np.diag([500**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  # Noise for [r, psi, theta]

# Initial covariance matrix
P_kf = np.diag([R[0, 0], 100, R[1, 1], 100])

# System and measurement noise
num_steps = len(t) - 1
w = np.random.multivariate_normal(np.zeros(n), Q1, num_steps).T
v = np.random.multivariate_normal(np.zeros(m), R, (len(sensor_positions), num_steps)).transpose(1, 2, 0)

# Storage
x_store = np.zeros((n, len(t)))
z_store = np.zeros((len(sensor_positions), m, len(t)))  # Shape: (num_sensors, measurements per sensor, time steps)
x_store[:, 0] = x_true[:, 0]

# Object motion model function
def object_motion_model(x, dt):
    if t[k] <= 125:
        return UM @ x
    elif 125 < t[k] <= 215:
        return CT(np.deg2rad(1)) @ x
    elif 215 < t[k] <= 340:
        return UM @ x
    elif 340 < t[k] <= 370:
        return CT(np.deg2rad(-3)) @ x
    else:
        return UM @ x


# Moving sensor trajectory
moving_sensor_start = np.array([-0.5e4, 0])  # Starting position
moving_sensor_end = np.array([1e4, 5e3])     # Ending position
moving_sensor_velocity = (moving_sensor_end - moving_sensor_start) / tf  # Velocity

moving_sensor_positions = np.array([
    np.append(moving_sensor_start + moving_sensor_velocity * t_step, 0)  # Append z = 0
    for t_step in t
])

# Update moving sensor position at each time step
sensor_positions_dynamic = [
    [moving_sensor_positions[k].tolist()] + sensor_positions[1:]
    for k in range(len(t))
]

# Define target height (assumed constant for simplicity)
target_height = 1000  # in meters

# Updated measurement model function
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

# --- Simulate True Trajectory and Measurements ---
for k in range(num_steps):
    # State propagation with process noise
    x_store[:, k + 1] = object_motion_model(x_store[:, k], T).flatten() + w[:, k]
    
    # Generate measurements for each sensor and store them individually
    for i, sensor_pos in enumerate(sensor_positions_dynamic[k]):
        z_store[i, :, k + 1] = measurement_model(x_store[:, k + 1], sensor_pos) + v[k, :, i]

# --- PLOTS ---

# 1. 2D Visualization
plt.figure(figsize=(12, 10))
plt.plot(x_store[0, :], x_store[2, :], 'r-', linewidth=2, label='True Trajectory')

# Add markers at regular intervals to show progression
marker_indices = np.arange(0, len(t), 10)
plt.scatter(x_store[0, marker_indices], x_store[2, marker_indices], c='black', s=30, alpha=0.7)

# Highlight start and end points
plt.scatter(x_store[0, 0], x_store[2, 0], c='blue', s=200, marker='o', label='Start')
plt.scatter(x_store[0, -1], x_store[2, -1], c='purple', s=200, marker='X', label='End')

# Plot sensor positions
for i, sensor in enumerate(sensor_positions):
    plt.scatter(sensor[0], sensor[1], c='green', marker='^', s=100, label=f'Sensor {i+1}')

# Plot moving sensor trajectory
plt.plot(moving_sensor_positions[:, 0], moving_sensor_positions[:, 1], 'g--', linewidth=2, label='Moving Sensor')

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Visualization of Target Tracking Scenario')
plt.grid(True)
plt.legend()
plt.savefig('target_tracking_2d.png', dpi=300)
plt.show()

# 2. 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot sensor positions
for i, sensor in enumerate(sensor_positions):
    ax.scatter(sensor[0], sensor[1], 0, c='green', marker='^', s=100, label=f'Sensor {i+1}')

# Plot moving sensor trajectory
ax.plot(moving_sensor_positions[:, 0], moving_sensor_positions[:, 1], np.zeros(len(t)), 
        c='green', linestyle='--', label='Moving Sensor', linewidth=2)

# Plot target trajectory
ax.plot(x_store[0, :], x_store[2, :], np.zeros(len(t)), c='red', label='True Trajectory', linewidth=2)

# Add labels and legend
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Visualization of Target Tracking Scenario')
ax.legend()
ax.grid(True)
plt.savefig('target_tracking_3d.png', dpi=300)
plt.show()

# 3. Azimuth Angles Visualization
plt.figure(figsize=(10, 6))

# Extract azimuth angles (second measurement component) from z_store
for i in range(len(sensor_positions)):
    azimuth_angles = z_store[i, 1, :]  # Second component (psi) for each sensor
    plt.plot(t, np.rad2deg(azimuth_angles), label=f'Sensor {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('Azimuth Angle (degrees)')
plt.title('Azimuth Angles Measured by Sensors')
plt.legend()
plt.grid(True)
plt.savefig('azimuth_angles.png', dpi=300)
plt.show()
