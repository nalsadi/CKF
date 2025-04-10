import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
tf = 500
dt = 1  # Time interval (seconds)
T = dt  # Ensure consistency
t = np.arange(0, tf + dt, dt)  # Time array
n = 4  # State dimension
m = 3  # Measurement dimension
target_height = 1000  # Target height in meters

# Uniform motion model
UM = np.array([
    [1, T, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, T],
    [0, 0, 0, 1]
])

# Process noise covariance matrix
L1 = 0.16
Q = L1 * np.array([
    [T**3 / 3, T**2 / 2, 0, 0],
    [T**2 / 2, T, 0, 0],
    [0, 0, T**3 / 3, T**2 / 2],
    [0, 0, T**2 / 2, T]
])

# Measurement noise covariance matrix
R = np.diag([
    500**2,  # Range noise
    np.deg2rad(1)**2,  # Azimuth noise
    np.deg2rad(1)**2   # Pitch noise
])

# Coordinated turn model
def CT(w):
    if abs(w) < 1e-6:  # Handle near-zero angular velocity
        return UM
    return np.array([
        [1, np.sin(w * T) / w, 0, -(1 - np.cos(w * T)) / w],
        [0, np.cos(w * T), 0, -np.sin(w * T)],
        [0, (1 - np.cos(w * T)) / w, 1, np.sin(w * T) / w],
        [0, np.sin(w * T), 0, np.cos(w * T)]
    ])

# Object motion model function
def object_motion_model(x, dt, k):
    # Determine motion model based on time
    if t[k] <= 125:
        return UM @ x
    elif 125 < t[k] <= 215:
        return CT(np.deg2rad(1)) @ x
    elif 215 < t[k] <= 340:
        return UM @ x
    elif 340 < t[k] <= 370:
        return CT(np.deg2rad(-3)) @ x
    return UM @ x

# Measurement model function
def measurement_model(x, sensor_pos):
    # Calculate differences in position
    dx = x[0] - sensor_pos[0]  # ξk - x^i_k
    dy = x[2] - sensor_pos[1]  # ηk - y^i_k
    h = target_height - sensor_pos[2]  # Height difference

    # Distance calculation
    d_squared = dx**2 + dy**2
    r = np.sqrt(d_squared + h**2)

    # Azimuth angle
    psi = np.arctan2(dy, dx)

    # Pitch angle
    theta = np.arctan2(h, np.sqrt(d_squared))

    return np.array([r, psi, theta])
