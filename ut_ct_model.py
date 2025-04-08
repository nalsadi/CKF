import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Random seed for reproducibility
np.random.seed(42)

# Final simulation time and radar sample rate
tf = 500
T = 5
t = np.arange(0, tf+T, T)
n = 4  # Number of states (reduced to 4)
m = 2  # Number of measurements

# Initial state (remove fifth state)
x_true = np.array([[25e3], [-120], [10e3], [0]])
x_kf = x_true.copy()
x_ekf = x_true.copy()
x_ukf = x_true.copy()

# Measurement matrix (unchanged)
H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
Hlin = H.copy()

# Uniform motion model (remove fifth row and column)
UM = np.array([
    [1, T, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, T],
    [0, 0, 0, 1]
])

# Coordinated turn model (remove fifth row and column)
def CT(w):
    if abs(w) < 1e-6:
        return np.array([
            [1, T, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, T],
            [0, 0, 0, 1]
        ])
    else:
        return np.array([
            [1, np.sin(w*T)/w, 0, -(1 - np.cos(w*T))/w],
            [0, np.cos(w*T), 0, -np.sin(w*T)],
            [0, (1 - np.cos(w*T))/w, 1, np.sin(w*T)/w],
            [0, np.sin(w*T), 0, np.cos(w*T)]
        ])

# Noise covariances (remove fifth row and column)
L1 = 0.16
L2 = 0.01
Q1 = L1 * np.array([
    [T**3/3, T**2/2, 0, 0],
    [T**2/2, T, 0, 0],
    [0, 0, T**3/3, T**2/2],
    [0, 0, T**2/2, T]
])
Q2 = L1 * np.array([
    [T**3/3, T**2/2, 0, 0],
    [T**2/2, T, 0, 0],
    [0, 0, T**3/3, T**2/2],
    [0, 0, T**2/2, T]
])
R = (500)**2 * np.eye(m)

# Initial covariance matrices (remove fifth row and column)
P_kf = np.diag([R[0,0], 100, R[1,1], 100])
P_ekf = P_kf.copy()
P_ukf = P_kf.copy()

# System and measurement noise (adjust dimensions)
num_steps = len(t) - 1
w = np.random.multivariate_normal(np.zeros(n), Q1, num_steps).T
v = np.random.multivariate_normal(np.zeros(m), R, num_steps).T

# Storage (adjust dimensions)
x_store = np.zeros((n, len(t)))
z_store = np.zeros((m, len(t)))
x_store[:, 0] = x_true[:, 0]
z_store[:, 0] = (H @ x_true).flatten()

# --- Simulate True Trajectory and Measurements ---
for k in range(num_steps):
    if k*T <= 125:
        x_store[:, k+1] = (UM @ x_store[:, k]).flatten() + w[:, k]
    elif 125 < k*T <= 215:
        x_store[:, k+1] = (CT(np.deg2rad(1)) @ x_store[:, k]).flatten() + w[:, k]
    elif 215 < k*T <= 340:
        x_store[:, k+1] = (UM @ x_store[:, k]).flatten() + w[:, k]
    elif 340 < k*T <= 370:
        x_store[:, k+1] = (CT(np.deg2rad(-3)) @ x_store[:, k]).flatten() + w[:, k]
    else:
        x_store[:, k+1] = (UM @ x_store[:, k]).flatten() + w[:, k]
        
    z_store[:, k+1] = (H @ x_store[:, k+1]).flatten() + v[:, k]

# Dummy KF, EKF, UKF update functions
def kf_predict(x, P, F, Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def kf_update(x_pred, P_pred, z, H, R):
    y = z - (H @ x_pred).flatten()
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(P_pred)) - K @ H) @ P_pred
    return x_upd, P_upd

# Estimation Results (adjust dimensions)
x_kf_store = np.zeros_like(x_store)
x_kf_store[:,0] = x_kf.flatten()
P_kf_store = np.zeros((n, n, len(t)))
P_kf_store[:,:,0] = P_kf

# Run filters
for k in range(num_steps):
    # KF prediction and update
    x_pred, P_pred = kf_predict(x_kf_store[:,k], P_kf_store[:,:,k], UM, Q1)
    x_upd, P_upd = kf_update(x_pred, P_pred, z_store[:,k+1], H, R)
    x_kf_store[:,k+1] = x_upd.flatten()
    P_kf_store[:,:,k+1] = P_upd

# RMSE Calculation (unchanged)
def rmse(x_est, x_true):
    return np.sqrt(np.mean((x_est - x_true)**2, axis=1))

rmse_kf = rmse(x_kf_store, x_store)

# Show RMSE
print("\nSimulation Results:")
print("State\tKF RMSE")
for i in range(n):
    print(f"{i+1}\t{rmse_kf[i]:.2f}")

# --- PLOTS ---

# 1. True Trajectory and Measurements
plt.figure()
plt.plot(x_store[0,:], x_store[2,:], label='True Trajectory')
plt.plot(z_store[0,:], z_store[1,:], 'rx', label='Radar Measurements')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('ATC Problem: True vs Measured')
plt.legend()
plt.grid()
plt.xlim([-20000, 30000])
plt.ylim([-25000, 15000])
plt.show()

# 2. KF Estimated Trajectory
plt.figure()
plt.plot(x_store[0,:], x_store[2,:], label='True')
plt.plot(x_kf_store[0,:], x_kf_store[2,:], 'g--x', label='KF Estimate')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('ATC Estimation Results')
plt.legend()
plt.grid()
plt.xlim([-20000, 30000])
plt.ylim([-25000, 15000])
plt.show()

# 3. Position Tracking Errors
tracking_error_kf = np.sqrt((x_store[0,:] - x_kf_store[0,:])**2 + (x_store[2,:] - x_kf_store[2,:])**2)

plt.figure()
plt.plot(t, tracking_error_kf, label='KF Tracking Error')
plt.xlabel('Time (s)')
plt.ylabel('Position Tracking Error (m)')
plt.title('Position Tracking Error Over Time')
plt.legend()
plt.grid()
plt.show()
