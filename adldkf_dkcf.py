import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
seed = 42  # You can choose any integer value
np.random.seed(seed)
torch.manual_seed(seed)


# Simulation parameters
tf = 200  # Final time in simulation (seconds)
dt = 0.7  # Sample rate (seconds)
t = np.arange(0, tf + dt, dt)  # Time vector
n = 4  # Number of states [x, vx, y, vy]
m = 3  # Number of measurements [range, azimuth, elevation]
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
num_static_sensors = 8  # Number of static sensors
sensor_radius = 20  # Radius of the circle (km)

# Generate static sensors in a circular arrangement
static_sensors = [
    [
        center_x + sensor_radius * np.cos(2 * np.pi * i / num_static_sensors),
        center_y + sensor_radius * np.sin(2 * np.pi * i / num_static_sensors),
        sensor_height
    ]
    for i in range(num_static_sensors)
]

# Mobile sensor with initial position
mobile_sensor = [15, 10, sensor_height]
mobile_sensor_vel = [0.05, 0.05, 0]  # Mobile sensor velocity

sensor_positions = [static_sensors + [mobile_sensor]]
num_nodes = len(static_sensors) + 1  # Update the number of nodes
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

# Store the mobile sensor trajectory for 3D visualization
mobile_sensor_trajectory = np.array([sensor_positions[k][num_nodes-1] for k in range(len(t))])

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

# Define simple NN for parameter estimation
class SimpleNNEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNNEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# Define the sliding window size
window_size = 50

# Initialize NN models and optimizers for each node
nn_models = []
optimizers = []
criterions = []
for _ in range(num_nodes):
    # Change input_dim to m * window_size to match the flattened input
    model = SimpleNNEstimator(input_dim=m * window_size, hidden_dim=32*4, output_dim=n+m+m).to(device)
    
    # Improve weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    model.apply(init_weights)
    
    nn_models.append(model)

# Adjust learning rate and add regularization
optimizers = [
    optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Reduce learning rate and add weight decay
    for model in nn_models
]

# Use SmoothL1Loss instead of MSELoss
criterions = [nn.SmoothL1Loss() for _ in range(num_nodes)]  # More robust loss function

# Initialize measurement history for each node
z_node_histories = [torch.zeros((m, window_size), dtype=torch.float32, device=device) for _ in range(num_nodes)]

# Initialize optimal Q and R matrices for each node
Q_opts = [Q.clone() for _ in range(num_nodes)]
R_opts = [R.clone() for _ in range(num_nodes)]
# Initialize delta values for each node
delta_opts = [torch.ones(m, dtype=torch.float32, device=device) * 0.01 for _ in range(num_nodes)]  # Default delta values per measurement

# Add DKCF parameters
N = num_nodes  # Number of nodes for DKCF

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

# Initialize all KF with same initial state
for node in range(num_nodes):
    x_kf[:, 0, node] = torch.tensor(x0_true, dtype=torch.float32, device=device) + torch.randn(n, device=device) * 0.1

def sigmoid_saturation(innov, delta, alpha=1.0):
    return 1 / (1 + torch.exp(-alpha * (innov.abs() - delta)))

# Modified SSIF implementation for nonlinear systems with adaptive features
def ssif_adaptive(x, z, P, Q, R, delta, motion_model, measurement_model, sensor_pos, criterion):
    """
    Extended Kalman Filter with SSIF (Sigmoidal Saturated Increment Function) implementation
    with adaptive features for nonlinear systems.
    
    x: state vector
    z: measurement vector
    P: error covariance matrix
    Q: process noise covariance
    R: measurement noise covariance
    delta: sliding layer widths
    motion_model: function for state transition
    measurement_model: function for measurement prediction
    sensor_pos: sensor position for measurement model
    criterion: loss function for adaptive optimization
    """
    n = x.shape[0]
    m = z.shape[0]
    delta = delta.clone().detach()  # Sliding layer widths

    # Prediction stage
    x_pred = motion_model(x, dt)  # Predict state
    F = torch.zeros((n, n), device=x.device)  # State transition Jacobian
    epsilon = 1e-4
    for j in range(n):
        x_perturbed = x.clone()
        x_perturbed[j] += epsilon
        F[:, j] = (motion_model(x_perturbed, dt) - x_pred) / epsilon
    P_pred = F @ P @ F.T + Q  # Predict error covariance

    # Update stage
    z_pred = measurement_model(x_pred, sensor_pos)  # Predicted measurement
    z_pred = z_pred.clone().detach().requires_grad_(True)  # Make prediction differentiable for loss
    
    H = torch.zeros((m, n), device=x.device)  # Measurement Jacobian
    for j in range(n):
        x_perturbed = x_pred.clone()
        x_perturbed[j] += epsilon
        H[:, j] = (measurement_model(x_perturbed, sensor_pos) - z_pred.detach()) / epsilon
        
    innov = z - z_pred  # Innovation
    innov[1:3] = torch.atan2(torch.sin(innov[1:3]), torch.cos(innov[1:3]))  # Wrap angles to [-pi, pi]

    # Compute loss for adaptive optimization
    loss = criterion(z_pred, z)

    sat = sigmoid_saturation(innov, delta, alpha=0.7)  # Saturation terms
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ torch.linalg.pinv(S)  # Kalman gain
    x_updated = x_pred + K @ (sat * innov.detach())  # Update state estimate
    P_updated = (torch.eye(n, device=x.device) - K @ H) @ P_pred  # Update error covariance

    return x_updated, P_updated, z_pred, loss

# Generate true satellite trajectory
for k in range(1, len(t)):
    # Check for fault condition - occurs halfway through simulation
    fault_time_index = len(t) // 1
    is_fault_active = k >= fault_time_index
    
    # True system dynamics with fault
    if is_fault_active:
        # System fault: Add bias to acceleration and introduce unexpected trajectory change
        # This simulates a thruster malfunction or unexpected gravitational disturbance
        fault_bias = torch.tensor([0.05, 0.003, -0.04, 0.002], dtype=torch.float32, device=device)
        fault_process_noise_multiplier = 5.0  # 5x more process noise during fault
        
        # Apply model with fault
        x[:, k] = satellite_motion_model(x[:, k-1].unsqueeze(0), dt).squeeze(0) + fault_bias + w[:, k] * fault_process_noise_multiplier
    else:
        # Normal dynamics
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


# Simulation loop
for k in range(1, len(t)):
    # Check for fault condition
    fault_time_index = len(t) // 1
    is_fault_active = k >= fault_time_index
    
    # ----- ADL-DKF -----
    # First update individual node estimates
    for node in range(num_nodes):
        sensor_pos = sensor_positions_torch[k][node]
        z_node = z[:, k, node]
        
        # Update measurement history
        z_node_histories[node] = torch.roll(z_node_histories[node], -1, dims=1)
        z_node_histories[node][:, -1] = z_node

        # NN parameter updates (every 10 steps)
        if k % 10 == 0:  # and k > window_size:
            optimizer = optimizers[node]
            nn_model = nn_models[node]
            criterion = criterions[node]

            # Run standard training epochs for all nodes
            training_epochs = 10
            
            for epoch in range(training_epochs):
                optimizer.zero_grad()

                # Process measurement history
                z_input_tensor = z_node_histories[node].T  # Shape: [window_size, m]
                z_input_tensor_normalized = (z_input_tensor - torch.mean(z_input_tensor, dim=0)) / (torch.std(z_input_tensor, dim=0) + 1e-8)
                z_input_tensor_normalized = z_input_tensor_normalized.flatten().unsqueeze(0)  # Flatten for NN input

                # Add Gaussian noise for variability
                noise = torch.randn_like(z_input_tensor_normalized) * 0.01
                z_input_tensor_normalized = z_input_tensor_normalized + noise  # Avoid in-place operation

                # Forward pass
                nn_output = nn_model(z_input_tensor_normalized)

                # Extract Q, R and delta parameters from NN output
                q_values = nn_output[0, :n]  # First n elements for process noise diagonals
                r_values = nn_output[0, n:n+m]  # Next m elements for measurement noise diagonals
                delta_values = nn_output[0, n+m:]  # Last m elements for delta values

                # Apply positive constraints
                q_values = torch.exp(q_values)  # Ensure positive values for covariance
                r_values = torch.exp(r_values)  # Ensure positive values for covariance
                delta_values = torch.abs(delta_values)  # Ensure positive values for delta
                
                # Update the adaptive filter parameters
                Q_opt = torch.diag(q_values)
                R_opt = torch.diag(r_values)
                delta_opt = delta_values
                
                # Run SSIF with adaptive features and get loss
                x_updated, P_updated, z_pred, loss = ssif_adaptive(
                    x_kf[:, k-1, node], z_node, P_kf[node], 
                    Q_opt, R_opt, delta_opt, 
                    satellite_motion_model, measurement_model, sensor_pos,
                    criterion
                )
                
                print(f"Node {node}, Epoch {epoch}, Loss: {loss.item()}")

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)

                optimizer.step()

                # Update filter parameters with the optimized values
                with torch.no_grad():
                    Q_opts[node] = Q_opt.detach().clone()
                    R_opts[node] = R_opt.detach().clone()
                    delta_opts[node] = delta_opt.detach().clone()

        # Regular filter update using current optimal parameters
        x_kf[:, k, node], P_kf[node], _, _ = ssif_adaptive(
            x_kf[:, k-1, node], z_node, P_kf[node], 
            Q_opts[node], R_opts[node], delta_opts[node],
            satellite_motion_model, measurement_model, sensor_pos,
            criterions[node]
        )

    # Consensus step for ADL-DKF using averaged model weights
    if k % 10 == 0:  # Perform weight averaging every 10 steps
        # Average NN model weights across all nodes
        with torch.no_grad():
            for param_name in nn_models[0].state_dict().keys():
                # Compute the average of the parameter across all models
                avg_param = torch.mean(
                    torch.stack([nn_models[node].state_dict()[param_name] for node in range(num_nodes)]), dim=0
                )
                # Update each model with the averaged parameter
                for node in range(num_nodes):
                    nn_models[node].state_dict()[param_name].copy_(avg_param)

    # Compute the mean state across all nodes
    mean_state = torch.mean(x_kf[:, k, :], dim=1)  # Mean across all nodes
    mean_P = torch.mean(torch.stack([P_kf[i] for i in range(num_nodes)]), dim=0)

    # Update all nodes with consensus state
    for node in range(num_nodes):
        x_kf[:, k, node] = mean_state
        P_kf[node] = mean_P
        squared_error_kf[:, k, node] = (x[:, k] - mean_state)**2


# Compute RMSE per state over time (using consensus states)
rmse_adldkf_state = torch.sqrt(torch.mean(squared_error_kf, dim=(0, 2)))  # Mean across states and nodes

# Compute total RMSE over time (using consensus states)
rmse_adldkf_total = torch.sqrt(torch.sum(torch.mean(squared_error_kf, dim=2), dim=0))

# Convert to numpy for plotting (use mean states across nodes)
x_np = x.cpu().detach().numpy()
x_kf_mean = torch.mean(x_kf, dim=2).cpu().detach().numpy()  # Mean across nodes

# Create figure with 3 rows and 2 columns
plt.figure(figsize=(15, 12))

# First row: State trajectories (x and y positions)
plt.subplot(3, 2, 1)
plt.plot(x_np[0, 1:], x_np[2, 1:], 'k-', label='True', linewidth=2)
plt.plot(x_kf_mean[0, 1:], x_kf_mean[2, 1:], 'r--', label='ADL-DKF', linewidth=1.5)
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Satellite Trajectory')
plt.legend()
plt.grid(True)

# Second subplot: X position error over time
plt.subplot(3, 2, 2)
plt.plot(t[1:], x_np[0, 1:] - x_kf_mean[0, 1:], 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('X Position Error (km)')
plt.title('X Position Error')
plt.legend()
plt.grid(True)

# Third subplot: Y position error over time
plt.subplot(3, 2, 3)
plt.plot(t[1:], x_np[2, 1:] - x_kf_mean[2, 1:], 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Y Position Error (km)')
plt.title('Y Position Error')
plt.legend()
plt.grid(True)

# Fourth subplot: Velocity estimation
plt.subplot(3, 2, 4)
true_vel = np.sqrt(x_np[1, :]**2 + x_np[3, :]**2)
plt.plot(t, true_vel, 'k-', label='True Velocity', linewidth=2)
adldkf_vel = np.sqrt(x_kf_mean[1, :]**2 + x_kf_mean[3, :]**2)
plt.plot(t, adldkf_vel, 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (km/s)')
plt.title('Velocity Estimation')
plt.legend()
plt.grid(True)

# Fifth subplot: Total RMSE over time
plt.subplot(3, 2, 5)
plt.plot(t[1:], rmse_adldkf_total[1:].cpu().detach().numpy(), 'r-', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Total RMSE')
plt.title('Total RMSE Over Time')
plt.legend()
plt.grid(True)

# Sixth subplot: Mean RMSE over time
plt.subplot(3, 2, 6)
plt.plot(t[1:], rmse_adldkf_state[1:].cpu().detach().numpy(), 'r-', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE Over Time')
plt.legend()
plt.grid(True)

# Add vertical line showing fault time and annotation
fault_time = t[len(t) // 2]
for i in range(2, 7):  # For all 6 subplots
    plt.subplot(3, 2, i)
    plt.axvline(x=fault_time, color='r', linestyle='--', alpha=0.5)
    plt.text(fault_time+5, plt.ylim()[1]*0.9, 'Fault Occurs', color='r', alpha=0.7)

plt.tight_layout()
plt.savefig('satellite_tracking_results.png', dpi=300)
plt.show()

# 3D Visualization of object movement, static and moving sensors, and estimates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot true object trajectory
ax.plot(x_np[0, :], x_np[2, :], target_height, 'k-', label='True Object Trajectory', linewidth=2)

# Plot mobile sensor trajectory
ax.plot(mobile_sensor_trajectory[:, 0], mobile_sensor_trajectory[:, 1], mobile_sensor_trajectory[:, 2], 
        'b--', label='Mobile Sensor Trajectory', linewidth=1.5)

# Plot static sensor positions
for sensor in static_sensors:
    ax.scatter(sensor[0], sensor[1], sensor[2], c='r', marker='o', label='Static Sensor')

# Plot ADL-DKF estimated trajectory
ax.plot(x_kf_mean[0, :], x_kf_mean[2, :], target_height, 'g-.', label='ADL-DKF Estimated Trajectory', linewidth=1.5)

# Set labels and title
ax.set_xlabel('X Position (km)')
ax.set_ylabel('Y Position (km)')
ax.set_zlabel('Height (km)')
ax.set_title('3D Visualization of Object Movement and Sensor Positions')
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('object_movement_3d.png', dpi=300)
plt.show()

# Compute overall RMSE properly considering all dimensions
rmse_adldkf = torch.sqrt(torch.mean(squared_error_kf[:, 1:, :]))  # Mean across states, time, nodes

print("\nOverall Mean RMSE:")
print(f"ADL-DKF: {rmse_adldkf.item():.6f}")

# Also print state-wise RMSE
state_rmse_adldkf = torch.sqrt(torch.mean(squared_error_kf[:, 1:, :], dim=(1, 2)))  # Per state RMSE

print("\nState-wise RMSE:")
states = ['x', 'vx', 'y', 'vy']
for i, state in enumerate(states):
    print(f"{state:<3} - ADL-DKF: {state_rmse_adldkf[i].item():.6f}")

# Print additional information about the fault
print("\nFault Information:")
print(f"Fault occurred at t = {fault_time:.1f} seconds")
print(f"Fault type: System dynamics fault - trajectory bias and increased process noise")
print(f"Fault details:")
print(f"  - Position/velocity bias: [{fault_bias[0].item():.4f}, {fault_bias[1].item():.4f}, {fault_bias[2].item():.4f}, {fault_bias[3].item():.4f}]")
print(f"  - Process noise multiplier: {fault_process_noise_multiplier}x")

print("\nRMSE Before Fault:")
pre_fault_idx = slice(1, len(t)//2)
rmse_adldkf_pre = torch.sqrt(torch.mean(squared_error_kf[:, pre_fault_idx, :]))
print(f"ADL-DKF: {rmse_adldkf_pre.item():.6f}")

print("\nRMSE After Fault:")
post_fault_idx = slice(len(t)//2, len(t))
rmse_adldkf_post = torch.sqrt(torch.mean(squared_error_kf[:, post_fault_idx, :]))
print(f"ADL-DKF: {rmse_adldkf_post.item():.6f}")

# Calculate adaptation time (how long it takes ADL-DKF to re-achieve pre-fault performance)
# This is approximate - we look for when the RMSE returns to within 20% of pre-fault levels
post_fault_rmse = rmse_adldkf_total[len(t)//2:].cpu().detach().numpy()
pre_fault_avg = torch.mean(rmse_adldkf_total[:len(t)//2]).cpu().detach().numpy()
threshold = pre_fault_avg * 1.2  # 20% higher than pre-fault average

adaptation_indices = np.where(post_fault_rmse < threshold)[0]
if len(adaptation_indices) > 0:
    adaptation_time = adaptation_indices[0] * dt
    print(f"\nADL-DKF adaptation time after fault: {adaptation_time:.2f} seconds")
else:
    print("\nADL-DKF did not fully adapt after the fault within the simulation time")