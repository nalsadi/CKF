import numpy as np

np.random.seed(42)


tf = 500
dt = 1  # Time interval (seconds)
T = dt  # Ensure consistency
t = np.arange(0, tf + dt, dt)  # Update time array
n = 4
m = 3
target_height = 1000  # in meters



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

# Object motion model function
def object_motion_model(x, dt, k):
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


L1 = 0.16
Q = L1 * np.array([
    [T**3 / 3, T**2 / 2, 0, 0],
    [T**2 / 2, T, 0, 0],
    [0, 0, T**3 / 3, T**2 / 2],
    [0, 0, T**2 / 2, T]
])
R = np.diag([500**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  # Noise for [r, psi, theta]
