import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Satellite dynamics: constant velocity model
def simulate_satellite_motion(x0, F, steps, Q):
    x = x0
    states = [x]
    for _ in range(steps):
        w = np.random.multivariate_normal(np.zeros(len(Q)), Q)
        x = F @ x + w
       



# Simulation parameters
dt = 1.0
steps = 50
num_stations = 3

# State vector: [x, y, vx, vy]
x0 = np.array([0, 0, 1, 0.5])
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
Q = 0.01 * np.eye(4)
R = 0.5 * np.eye(2)

# Simulate true satellite motion
x_true = x0.copy()
true_trajectory = [x_true[:2]]
for _ in range(steps):
    w = np.random.multivariate_normal(np.zeros(4), Q)
    x_true = F @ x_true + w
    true_trajectory.append(x_true[:2])
true_trajectory = np.array(true_trajectory)

# Each station has its own KF state
x_estimates = [x0 + np.random.randn(4) for _ in range(num_stations)]
P_estimates = [np.eye(4) for _ in range(num_stations)]
measurements = [[] for _ in range(num_stations)]
est_trajectories = [[] for _ in range(num_stations)]

for t in range(steps):
    z_true = true_trajectory[t+1]

    for i in range(num_stations):
        # Simulate noisy measurement
        z = z_true + np.random.multivariate_normal(np.zeros(2), R)
        measurements[i].append(z)

        # Predict
        x_pred = F @ x_estimates[i]
        P_pred = F @ P_estimates[i] @ F.T + Q

        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_estimates[i] = x_pred + K @ y
        P_estimates[i] = (np.eye(4) - K @ H) @ P_pred

        est_trajectories[i].append(x_estimates[i][:2])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'k-', label='True trajectory')
for i in range(num_stations):
    est = np.array(est_trajectories[i])
    plt.plot(est[:, 0], est[:, 1], '--', label=f'Station {i+1} estimate')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('Distributed Satellite Tracking with KF')
plt.grid()
plt.show()

