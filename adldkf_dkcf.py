import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
tf = 100  # Final time in simulation (seconds)
dt = 0.5  # Sample rate (seconds)
t = np.arange(0, tf + dt, dt)  # Time vector
n = 4  # Number of states [x, vx, y, vy]
m = 3  # Number of measurements [range, azimuth, elevation]
num_nodes = 4  # Number of sensor nodes (3 static + 1 mobile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Orbital parameters
center_x = 35  # Center of orbit in x (km)
center_y = 30  # Center of orbit in y (km)
orbital_radius = 15  # km
orbital_period = 120  # seconds
angular_velocity = 2 * np.pi / orbital_period
target_height = 0.5  # km
sensor_height = 0.2  # km

# Initial satellite state [x, vx, y, vy]
initial_angle = np.pi/4  # Start at 45 degrees
pos_x = center_x + orbital_radius * np.cos(initial_angle)
pos_y = center_y + orbital_radius * np.sin(initial_angle)
vel_x = -orbital_radius * angular_velocity * np.sin(initial_angle)
vel_y = orbital_radius * angular_velocity * np.cos(initial_angle)
x0_true = np.array([pos_x, vel_x, pos_y, vel_y], dtype=np.float32)

# Set up sensor network
# Format: [x, y, z]
static_sensors = [
    [25, 10, sensor_height],  # Sensor 1
    [35, 10, sensor_height],  # Sensor 2
    [45, 10, sensor_height]   # Sensor 3
]
# Mobile sensor with initial position
mobile_sensor = [15, 10, sensor_height]
mobile_sensor_vel = [0.05, 0.05, 0]  # Mobile sensor velocity

sensor_positions = [static_sensors + [mobile_sensor]]
for k in range(1, len(t)):
    current_positions = []
    for i, sensor in enumerate(sensor_positions[-1]):
        if i == num_nodes - 1:  # Mobile sensor
            new_pos = [
                sensor[0] + mobile_sensor_vel[0] * dt,
                sensor[1] + mobile_sensor_vel[1] * dt,
                sensor[2] + mobile_sensor_vel[2] * dt
            ]
            current_positions.append(new_pos)
        else:
            current_positions.append(sensor.copy())
    sensor_positions.append(current_positions)

# Convert to PyTorch tensors
sensor_positions_torch = [torch.tensor(pos_list, dtype=torch.float32, device=device) 
                          for pos_list in sensor_positions]

# Process and measurement noise
Q_np = np.diag([1e-6, 1e-6, 1e-6, 1e-6]).astype(np.float32)  # Process noise
Q = torch.tensor(Q_np, dtype=torch.float32, device=device)

# Measurement noise for range (m), azimuth (rad), elevation (rad)
R_base = np.diag([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2]).astype(np.float32)
R = torch.tensor(R_base, dtype=torch.float32, device=device)

# Generate variant measurement noise for each node's sensors
base_var = np.array([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  # Base variances
noise_variation = 0.5  # 50% maximum variation from base variance
node_variances = []

for i in range(num_nodes):
    # Generate random variations between -noise_variation and +noise_variation
    variation = 1 + np.random.uniform(-noise_variation, noise_variation, size=m)
    # Apply variation to base variances
    node_var = base_var * variation
    node_variances.append(node_var)

# Create R matrices for each node
R_list = [torch.diag(torch.tensor(var, dtype=torch.float32, device=device)) for var in node_variances]

# Find worst measurement noise for regular KF
worst_variances = np.max(node_variances, axis=0)  # Get maximum variance for each sensor across all nodes
R_worst = torch.diag(torch.tensor(worst_variances, dtype=torch.float32, device=device))

# Print the measurement noise variances for each node and worst case
print("\nMeasurement noise variances for each node:")
for i, var in enumerate(node_variances):
    print(f"Node {i+1}: {var}")
print(f"\nWorst-case measurement noise (used for regular KF): {worst_variances}")

# Satellite motion model function
def satellite_motion_model(x, dt, device=device):
    """
    PyTorch implementation of satellite motion model
    x: state tensor [x, vx, y, vy]
    dt: time step in seconds
    """
    # Current position and velocity
    pos_x, vel_x, pos_y, vel_y = x.unbind(dim=-1)
    
    # Distance from center
    rx = pos_x - center_x
    ry = pos_y - center_y
    r = torch.sqrt(rx**2 + ry**2)
    
    # Orbital velocity (assuming circular orbit)
    orbital_period_tensor = torch.tensor(orbital_period, device=device)
    angular_velocity_tensor = 2 * torch.tensor(np.pi, device=device) / orbital_period_tensor
    
    # Calculate acceleration (centripetal)
    ax = -rx * angular_velocity_tensor**2
    ay = -ry * angular_velocity_tensor**2
    
    # Apply motion equations
    new_pos_x = pos_x + vel_x * dt + 0.5 * ax * dt**2
    new_vel_x = vel_x + ax * dt
    new_pos_y = pos_y + vel_y * dt + 0.5 * ay * dt**2
    new_vel_y = vel_y + ay * dt
    
    return torch.stack([new_pos_x, new_vel_x, new_pos_y, new_vel_y], dim=-1)

# Define PyTorch measurement model
def measurement_model(x, sensor_pos):
    """
    PyTorch implementation of measurement model
    x: state tensor [x, vx, y, vy]
    sensor_pos: tensor [x, y, z]
    Returns: tensor [range, azimuth, elevation]
    """
    dx = x[..., 0] - sensor_pos[..., 0]
    dy = x[..., 2] - sensor_pos[..., 1]
    dz = target_height - sensor_pos[..., 2]  # Target height - sensor height
    
    # Range
    r = torch.sqrt(dx**2 + dy**2 + dz**2)
    
    # Azimuth angle
    psi = torch.atan2(dy, dx)
    
    # Elevation angle
    theta = torch.atan2(dz, torch.sqrt(dx**2 + dy**2))
    
    return torch.stack([r, psi, theta], dim=-1)

# Define LSTM for parameter estimation
class LSTMEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMEstimator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Initialize LSTM model, optimizer, and loss function
lstm_model = LSTMEstimator(input_dim=m, hidden_dim=32, output_dim=n+m).to(device)
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Add DKCF parameters
N = num_nodes  # Number of nodes for DKCF
c = 0.1  # Consensus gain

# Initialize variables for tracking
x = torch.zeros((n, len(t)), dtype=torch.float32, device=device)
x[:, 0] = torch.tensor(x0_true, dtype=torch.float32, device=device)
z = torch.zeros((m, len(t), N), dtype=torch.float32, device=device)
w = torch.tensor(np.random.multivariate_normal(mean=np.zeros(n), cov=Q_np, size=len(t)).T, 
                dtype=torch.float32, device=device)

# State and covariance tracking for nodes
x_kf = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device, requires_grad=False)
P_kf = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)
squared_error_kf = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device)

# Initialize DKCF variables
dkcf_states = torch.zeros((N, len(t), n), dtype=torch.float32, device=device)
P_dkcf = [torch.eye(n, dtype=torch.float32, device=device) for _ in range(N)]
squared_error_dkcf = torch.zeros((n, len(t), N), dtype=torch.float32, device=device)

# Initialize all KF with same initial state
for node in range(num_nodes):
    x_kf[:, 0, node] = torch.tensor(x0_true, dtype=torch.float32, device=device) + torch.randn(n, device=device) * 0.1
    dkcf_states[node, 0, :] = torch.tensor(x0_true, dtype=torch.float32, device=device) + torch.randn(n, device=device) * 0.1

# Kalman Filter functions
def kf_predict(x, u, P, Q, dt):
    # Predict satellite state using motion model
    x_pred = satellite_motion_model(x.unsqueeze(0), dt).squeeze(0)
    
    # Approximate the state transition matrix numerically
    F = torch.zeros((n, n), device=device)
    epsilon = 1e-4
    
    for j in range(n):
        x_perturbed = x.clone()
        x_perturbed[j] += epsilon
        x_perturbed_next = satellite_motion_model(x_perturbed.unsqueeze(0), dt).squeeze(0)
        F[:, j] = (x_perturbed_next - x_pred) / epsilon
        
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred, F

def kf_update(x_pred, z, P_pred, R, sensor_pos):
    # Get measurement prediction
    z_pred = measurement_model(x_pred.unsqueeze(0), sensor_pos.unsqueeze(0)).squeeze(0)
    
    # Calculate Jacobian
    H = torch.zeros((m, n), device=device)
    epsilon = 1e-4
    
    for j in range(n):
        x_perturbed = x_pred.clone()
        x_perturbed[j] += epsilon
        z_perturbed = measurement_model(x_perturbed.unsqueeze(0), sensor_pos.unsqueeze(0)).squeeze(0)
        H[:, j] = (z_perturbed - z_pred) / epsilon
    
    # Calculate innovation
    y = z - z_pred
    
    # Wrap angle differences to [-pi, pi]
    y[1:3] = torch.atan2(torch.sin(y[1:3]), torch.cos(y[1:3]))
    
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ torch.linalg.pinv(S)
    x_updated = x_pred + K @ y
    P_updated = (torch.eye(n, device=device) - K @ H) @ P_pred
    
    return x_updated, P_updated, z_pred

# Generate true satellite trajectory
for k in range(1, len(t)):
    # True system dynamics
    x[:, k] = satellite_motion_model(x[:, k-1].unsqueeze(0), dt).squeeze(0) + w[:, k]
    
    # Generate measurements for each sensor
    for node in range(num_nodes):
        sensor_pos = sensor_positions_torch[k][node]
        true_measurement = measurement_model(x[:, k].unsqueeze(0), sensor_pos.unsqueeze(0)).squeeze(0)
        
        # Add measurement noise based on node's R matrix
        node_cov = np.diag(node_variances[node])  # Convert variance vector to covariance matrix
        v = torch.tensor(np.random.multivariate_normal(
            mean=np.zeros(m), cov=node_cov), 
            dtype=torch.float32, device=device)
        z[:, k, node] = true_measurement + v

# Define the sliding window size
window_size = 20

# Simulation loop
for k in range(1, len(t)):
    # ----- ADL-DKF (LSTM-based approach) -----
    if k > window_size:
        # Use the last `window_size` entries for the LSTM input
        z_input_tensor = z[:, max(0, k + 1 - window_size):k + 1, 0].transpose(0, 1)  # Shape: [window_size, m]
        
        # Normalize using PyTorch functions
        z_mean = torch.mean(z_input_tensor, dim=0)
        z_std = torch.std(z_input_tensor, dim=0) + 1e-8
        z_input_tensor_normalized = (z_input_tensor - z_mean) / z_std
        
        # Add batch dimension
        z_input_tensor_normalized = z_input_tensor_normalized.unsqueeze(0)  # Shape: [1, window_size, m]
        
        # Get LSTM output
        lstm_output = lstm_model(z_input_tensor_normalized)
        
        # Extract and apply LSTM parameters
        q_diag = torch.nn.functional.softplus(lstm_output[:, :n])
        r_diag = torch.nn.functional.softplus(lstm_output[:, n:n+m])
        
        Q_adldkf = torch.diag_embed(q_diag).squeeze(0)
        R_adldkf = torch.diag_embed(r_diag).squeeze(0)
    else:
        Q_adldkf, R_adldkf = Q.clone(), R.clone()

    # ADL-DKF prediction and update
    sensor_pos = sensor_positions_torch[k][0]
    z_tensor = z[:, k, 0]
    x_pred_adldkf, P_pred_adldkf, _ = kf_predict(x_kf[:, k-1, 0], None, P_kf[0], Q_adldkf, dt)
    x_kf[:, k, 0], P_kf[0], _ = kf_update(x_pred_adldkf, z_tensor, P_pred_adldkf, R_adldkf, sensor_pos)
    squared_error_kf[:, k, 0] = (x[:, k] - x_kf[:, k, 0])**2

    # ----- DKCF -----
    for i in range(N):
        sensor_pos = sensor_positions_torch[k][i]
        z_tensor = z[:, k, i]
        
        # Update using the Kalman Filter
        x_pred_dkcf, P_pred_dkcf, _ = kf_predict(dkcf_states[i, k-1, :], None, P_dkcf[i], Q, dt)
        x_updated, P_dkcf[i], _ = kf_update(x_pred_dkcf, z_tensor, P_pred_dkcf, R_list[i], sensor_pos)
        dkcf_states[i, k, :] = x_updated.flatten()

    # Consensus step for DKCF using mean of all nodes
    mean_state = torch.mean(dkcf_states[:, k, :], dim=0)  # Mean across all nodes
    for i in range(N):
        x_i = dkcf_states[i, k, :]
        dkcf_states[i, k, :] = x_i + c * (mean_state - x_i)
        squared_error_dkcf[:, k, i] = (x[:, k] - dkcf_states[i, k, :])**2

# Compute RMSE per state over time
rmse_adldkf_state = torch.sqrt(torch.mean(squared_error_kf[:, :, 0], dim=0))  # Shape: (len(t),)
rmse_dkcf_state = torch.sqrt(torch.mean(squared_error_dkcf, dim=(0, 2)))  # Shape: (len(t),)

# Compute total RMSE over time by summing squared errors across states
rmse_adldkf_total = torch.sqrt(torch.sum(squared_error_kf[:, :, 0], dim=0))  # Shape: (len(t),)
rmse_dkcf_total = torch.sqrt(torch.sum(torch.mean(squared_error_dkcf, dim=2), dim=0))  # Shape: (len(t),)

# Convert to numpy for plotting
x_np = x.cpu().detach().numpy()
x_kf_np = x_kf.cpu().detach().numpy()
dkcf_states_np = dkcf_states.cpu().detach().numpy()

# Create figure with 3 rows and 2 columns
plt.figure(figsize=(15, 12))

# First row: State trajectories (x and y positions)
plt.subplot(3, 2, 1)
plt.plot(x_np[0, 1:], x_np[2, 1:], 'k-', label='True', linewidth=2)  # True trajectory
plt.plot(x_kf_np[0, 1:, 0], x_kf_np[2, 1:, 0], 'r--', label='ADL-DKF', linewidth=1.5)
plt.plot(dkcf_states_np[0, 1:, 0], dkcf_states_np[0, 1:, 2], 'g-.', label='DKCF', linewidth=1.5)
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Satellite Trajectory')
plt.legend()
plt.grid(True)

# Second subplot: X position error over time
plt.subplot(3, 2, 2)
plt.plot(t[1:], x_np[0, 1:] - x_kf_np[0, 1:, 0], 'r--', label='ADL-DKF')
plt.plot(t[1:], x_np[0, 1:] - dkcf_states_np[0, 1:, 0], 'g-.', label='DKCF')
plt.xlabel('Time (sec)')
plt.ylabel('X Position Error (km)')
plt.title('X Position Error')
plt.legend()
plt.grid(True)

# Third subplot: Y position error over time
plt.subplot(3, 2, 3)
plt.plot(t[1:], x_np[2, 1:] - x_kf_np[2, 1:, 0], 'r--', label='ADL-DKF')
plt.plot(t[1:], x_np[2, 1:] - dkcf_states_np[0, 1:, 2], 'g-.', label='DKCF')
plt.xlabel('Time (sec)')
plt.ylabel('Y Position Error (km)')
plt.title('Y Position Error')
plt.legend()
plt.grid(True)

# Fourth subplot: Velocity estimation
plt.subplot(3, 2, 4)
true_vel = np.sqrt(x_np[1, :]**2 + x_np[3, :]**2)
plt.plot(t, true_vel, 'k-', label='True Velocity', linewidth=2)
adldkf_vel = np.sqrt(x_kf_np[1, :, 0]**2 + x_kf_np[3, :, 0]**2)
dkcf_vel = np.sqrt(dkcf_states_np[0, :, 1]**2 + dkcf_states_np[0, :, 3]**2)
plt.plot(t, adldkf_vel, 'r--', label='ADL-DKF')
plt.plot(t, dkcf_vel, 'g-.', label='DKCF')
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (km/s)')
plt.title('Velocity Estimation')
plt.legend()
plt.grid(True)

# Fifth subplot: Total RMSE over time
plt.subplot(3, 2, 5)
plt.plot(t[1:], rmse_adldkf_total[1:].cpu().detach().numpy(), 'r-', label='ADL-DKF')
plt.plot(t[1:], rmse_dkcf_total[1:].cpu().detach().numpy(), 'g-', label='DKCF')
plt.xlabel('Time (sec)')
plt.ylabel('Total RMSE')
plt.title('Total RMSE Over Time')
plt.legend()
plt.grid(True)

# Sixth subplot: Mean RMSE over time
plt.subplot(3, 2, 6)
plt.plot(t[1:], rmse_adldkf_state[1:].cpu().detach().numpy(), 'r-', label='ADL-DKF')
plt.plot(t[1:], rmse_dkcf_state[1:].cpu().detach().numpy(), 'g-', label='DKCF')
plt.xlabel('Time (sec)')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('satellite_tracking_results.png', dpi=300)
plt.show()

# Print overall RMSE
print("\nOverall Mean RMSE:")
print(f"ADL-DKF: {torch.mean(rmse_adldkf_total[1:]).item():.6f}")
print(f"DKCF: {torch.mean(rmse_dkcf_total[1:]).item():.6f}")