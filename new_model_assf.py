import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)

# Simulation parameters
tf = 500  # Total simulation time (seconds)
dt = 1  # Time interval (seconds)
T= dt
t = np.arange(0, tf + dt, dt)
n = 4  # State dimension
m = 3  # Measurement dimension

# Object parameters
target_height = 1000  # in meters
sensor_height = 0.2  # km
turning_speed = np.deg2rad(0.5)  # 0.5 degrees/s

# Initial object state (position in km, velocity in km/s)
x0_true = np.array([25e3, -120, 10e3, 0], dtype=np.float32)  # Initial position and velocity

# Set up sensor network
# Static sensor positions
static_sensors = [
    [2500, 1000, sensor_height],  # Sensor 1
    [3500, 1000, sensor_height],  # Sensor 2
    [4500, 1000, sensor_height],  # Sensor 3
]

# Define moving sensor trajectory
moving_sensor_start = np.array([-0.5e4, -2000])  # Starting position
moving_sensor_end = np.array([1e4, 5e3])         # Ending position
moving_sensor_velocity = (moving_sensor_end - moving_sensor_start) / tf  # Velocity

moving_sensor_positions = np.array([
    np.append(moving_sensor_start + moving_sensor_velocity * t_step, 0)  # Append z = 0
    for t_step in t
])

# Update moving sensor position at each time step
sensor_positions_dynamic = [
    [moving_sensor_positions[k].tolist()] + static_sensors
    for k in range(len(t))
]

mobile_sensor_trajectory = np.array([sensor_positions_dynamic[k][0] for k in range(len(t))])

# Process and measurement noise
Q = np.eye(n) * 1e-5  # Process noise covariance as specified: 10^-5 * I
R_base = np.diag([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2]).astype(np.float32)

# Generate measurement noise for each node's sensors
base_var = np.array([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  
node_variances = [base_var for _ in range(len(static_sensors) + 1)]
R_list = [np.diag(var) for var in node_variances]

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

# Object motion model function with slower movement and exact final destination
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
z = np.zeros((m, len(t), len(static_sensors) + 1))
w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T

for k in range(1, len(t)):
    # State propagation with process noise
    x[:, k] = object_motion_model(x[:, k-1], dt) + w[:, k]
    
    # Generate measurements for each sensor
    for node in range(len(static_sensors) + 1):
        sensor_pos = sensor_positions_dynamic[k][node]
        true_measurement = measurement_model(x[:, k], sensor_pos)
        v = np.random.multivariate_normal(mean=np.zeros(m), cov=R_list[node])
        z[:, k, node] = true_measurement + v

# Import torch for sigmoid saturation
import torch

# Define sigmoid saturation function
def sigmoid_saturation(innov, delta, alpha=1.0):
    return 1 / (1 + torch.exp(-alpha * (innov / delta)))

# Modify EKF function to include delta and sigmoid saturation
def ekf(x, z, P, Q, R, motion_model, measurement_model, sensor_pos, delta, alpha=1.0):
    n = len(x)
    m = len(z)
    x_pred = motion_model(x, dt)
    F = np.zeros((n, n))
    epsilon = 1e-4
    for j in range(n):
        x_perturbed = x.copy()
        x_perturbed[j] += epsilon
        F[:, j] = (motion_model(x_perturbed, dt) - x_pred) / epsilon
    P_pred = F @ P @ F.T + Q
    z_pred = measurement_model(x_pred, sensor_pos)
    H = np.zeros((m, n))
    for j in range(n):
        x_perturbed = x_pred.copy()
        x_perturbed[j] += epsilon
        H[:, j] = (measurement_model(x_perturbed, sensor_pos) - z_pred) / epsilon
    innov = z - z_pred
    innov[1:3] = np.arctan2(np.sin(innov[1:3]), np.cos(innov[1:3]))
    sat = sigmoid_saturation(torch.tensor(innov), torch.tensor(delta), alpha=alpha).numpy()
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.pinv(S)
    x_updated = x_pred + K @ (sat * innov)
    P_updated = (np.eye(n) - K @ H) @ P_pred
    return x_updated, P_updated, z_pred

# Initialize variables for tracking
x_kf = np.zeros((n, len(t), len(static_sensors) + 1))
P_kf = np.array([np.eye(n) for _ in range(len(static_sensors) + 1)])
squared_error_kf = np.zeros((n, len(t), len(static_sensors) + 1))

for node in range(len(static_sensors) + 1):
    x_kf[:, 0, node] = x0_true + np.random.randn(n) * 0.1

#import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Use the existing variables from user's environment
# t, n, m, num_nodes, z, z_histories, sensor_positions_dynamic, x_kf, P_kf, satellite_motion_model,
# measurement_model, ekf, x, state_names

# Define LSTM model creation with dropout
def create_lstm_model(input_shape, hidden_dim, output_dim):
    model = Sequential([
        KerasLSTM(hidden_dim, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Initialize LSTM models for each node
window_size = 50
input_shape = (window_size, m)  # 10 time steps, m features per step
hidden_dim = 32*4
output_dim = n + m + m  # Predict Q (n values), R (m values), and delta (m values)
lstm_models = [create_lstm_model(input_shape, hidden_dim, output_dim) for _ in range(len(static_sensors) + 1)]

# Initialize measurement history for each node
z_histories = [np.zeros((window_size, m), dtype=np.float32) for _ in range(len(static_sensors) + 1)]

# Initialize TensorFlow MSE loss function
mse_loss_fn = MeanSquaredError()

# Initialize a dictionary to store training losses for each node
training_losses = {node: [] for node in range(len(static_sensors) + 1)}

# Simulation loop with LSTM-based Q, R, and delta prediction
for k in range(1, len(t)):
    for node in range(len(static_sensors) + 1):
        sensor_pos = sensor_positions_dynamic[k][node]
        z_node = z[:, k, node]

        # Update measurement history
        z_histories[node][:-1] = z_histories[node][1:]
        z_histories[node][-1] = z_node

        # Prepare input for LSTM (shape: 1 x window_size x m)
        z_input = z_histories[node].reshape(1, window_size, m).astype(np.float32)
        lstm_model = lstm_models[node]

        # Train LSTM model every 10 steps with early stopping
        if k % 50 == 0:
            # Average weights from all other nodes
            other_weights = [lstm_models[other_node].get_weights() for other_node in range(len(static_sensors) + 1) if other_node != node]
            mean_weights = [np.mean([weights[layer] for weights in other_weights], axis=0) for layer in range(len(other_weights[0]))]
            lstm_model.set_weights(mean_weights)

            # Train the model
            epochs = 10  # Increase epochs for better training
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            history = lstm_model.fit(z_input, np.zeros((1, n + m + m)), epochs=epochs, callbacks=[early_stopping], verbose=1)
            # Store the training loss
            training_losses[node].extend(history.history['loss'])

        # Predict Q, R, and delta
        lstm_output = lstm_model.predict(z_input, verbose=0)
        q_diag = np.exp(lstm_output[0][:n])
        r_diag = np.exp(lstm_output[0][n:n + m])
        delta = np.exp(lstm_output[0][n + m:])
        Q_pred = np.diag(q_diag)
        R_pred = np.diag(r_diag)

        # Smooth predictions using exponential smoothing
        alpha = 0.8  # Smoothing factor
        if k > 1:
            Q_pred = alpha * Q_pred + (1 - alpha) * previous_Q_pred
            R_pred = alpha * R_pred + (1 - alpha) * previous_R_pred
            delta = alpha * delta + (1 - alpha) * previous_delta
        previous_Q_pred, previous_R_pred, previous_delta = Q_pred, R_pred, delta

        # Perform EKF update with predicted Q, R, and delta
        x_updated, P_updated, z_pred = ekf(
            x_kf[:, k-1, node], z_node, P_kf[node], Q_pred, R_pred,
            object_motion_model, measurement_model, sensor_pos, delta
        )

        # Compute loss (mean squared error between predicted and actual measurements)
        loss = mse_loss_fn(z_node, z_pred).numpy()
        print(f"Time Step: {k}, Node: {node}, Loss: {loss:.6f}")

        # Update states
        x_kf[:, k, node] = x_updated
        P_kf[node] = P_updated

    # Consensus step
    mean_state = np.mean(x_kf[:, k, :], axis=1)
    mean_P = np.mean(P_kf, axis=0)
    for node in range(len(static_sensors) + 1):
        x_kf[:, k, node] = mean_state
        P_kf[node] = mean_P
        squared_error_kf[:, k, node] = (x[:, k] - mean_state)**2  # Compute squared error during consensus

# Plot training loss for a random node
random_node = np.random.randint(0, len(static_sensors) + 1)
plt.figure(figsize=(10, 6))
plt.plot(training_losses[random_node], label=f'Node {random_node}')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.title(f'Training Loss for Node {random_node}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'training_loss_node_{random_node}.png', dpi=300)
plt.show()

# Compute RMSE
rmse_adldkf_total = np.sqrt(np.sum(np.mean(squared_error_kf, axis=2), axis=0))
rmse_adldkf_state = np.sqrt(np.mean(squared_error_kf, axis=(0, 2)))

# Correct overall RMSE calculation
overall_rmse = np.sqrt(np.mean(np.mean(squared_error_kf, axis=2), axis=1))

# Print overall RMSE as an array or scalar
print(f"\nOverall RMSE for the estimate (per state): {overall_rmse}")
print(f"Mean Overall RMSE: {np.mean(overall_rmse):.6f}")

# Calculate total distance traveled by the object
total_distance = np.sum(np.sqrt(np.diff(x[0, :])**2 + np.diff(x[2, :])**2))

# Normalize RMSE by total distance to get RMSE per kilometer
rmse_per_km = overall_rmse / total_distance

# Print RMSE per kilometer
print(f"RMSE per kilometer (per state): {rmse_per_km}")
print(f"Mean RMSE per kilometer: {np.mean(rmse_per_km):.6f}")

# Visualization of results
plt.figure(figsize=(15, 12))

# Satellite Trajectory
plt.subplot(3, 2, 1)
plt.plot(x[0, 1:], x[2, 1:], 'k-', label='True', linewidth=2)
plt.plot(np.mean(x_kf[0, 1:, :], axis=1), np.mean(x_kf[2, 1:, :], axis=1), 'r--', label='ADL-DKF', linewidth=1.5)
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('Satellite Trajectory')
plt.legend()
plt.grid(True)

# X Position Error
plt.subplot(3, 2, 2)
plt.plot(t[1:], x[0, 1:] - np.mean(x_kf[0, 1:, :], axis=1), 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('X Position Error (km)')
plt.title('X Position Error')
plt.legend()
plt.grid(True)

# Y Position Error
plt.subplot(3, 2, 3)
plt.plot(t[1:], x[2, 1:] - np.mean(x_kf[2, 1:, :], axis=1), 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Y Position Error (km)')
plt.title('Y Position Error')
plt.legend()
plt.grid(True)

# Velocity Estimation
plt.subplot(3, 2, 4)
true_vel = np.sqrt(x[1, :]**2 + x[3, :]**2)
plt.plot(t, true_vel, 'k-', label='True Velocity', linewidth=2)
adldkf_vel = np.sqrt(np.mean(x_kf[1, :, :], axis=1)**2 + np.mean(x_kf[3, :, :], axis=1)**2)
plt.plot(t, adldkf_vel, 'r--', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (km/s)')
plt.title('Velocity Estimation')
plt.legend()
plt.grid(True)

# Total RMSE Over Time
plt.subplot(3, 2, 5)
plt.plot(t[1:], rmse_adldkf_total[1:], 'r-', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Total RMSE')
plt.title('Total RMSE Over Time')
plt.legend()
plt.grid(True)

# Mean RMSE Over Time
plt.subplot(3, 2, 6)
plt.plot(t[1:], rmse_adldkf_state[1:], 'r-', label='ADL-DKF')
plt.xlabel('Time (sec)')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE Over Time')
plt.legend()
plt.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('satellite_tracking_results_numpy.png', dpi=300)
plt.show()

import csv  # Import CSV module for saving results

# Save system model and results to separate CSV files for each state
for state_idx in range(n):
    csv_file_path = f'estimation_results_state_{state_idx}.csv'
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write headers
        csv_writer.writerow(['Time (sec)', 'Node', 'True Value', 'Estimated Value', 'Squared Error', 'RMSE'])
        
        # Write data for each time step and node
        for k in range(1, len(t)):
            for node in range(len(static_sensors) + 1):
                true_value = x[state_idx, k]
                estimated_value = x_kf[state_idx, k, node]
                squared_error = squared_error_kf[state_idx, k, node]
                rmse = np.sqrt(np.mean(squared_error_kf[state_idx, :k+1, node]))
                csv_writer.writerow([t[k], node, true_value, estimated_value, squared_error, rmse])
    
    print(f"Estimation results for state {state_idx} saved to {csv_file_path}")

# Save overall RMSE to a summary CSV
summary_csv_file_path = 'overall_rmse_summary.csv'
with open(summary_csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['State', 'Overall RMSE'])
    for state_idx, rmse in enumerate(overall_rmse):
        csv_writer.writerow([f'State {state_idx}', rmse])
    csv_writer.writerow(['Mean Overall RMSE', np.mean(overall_rmse)])

print(f"Overall RMSE summary saved to {summary_csv_file_path}")

# Save true system model to a single CSV
true_csv_file_path = 'true_system_model.csv'
with open(true_csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers
    csv_writer.writerow(['Time (sec)'] + [f'State {i} (True)' for i in range(n)])
    
    # Write true values for each time step
    for k in range(len(t)):
        csv_writer.writerow([t[k]] + list(x[:, k]))
    
print(f"True system model saved to {true_csv_file_path}")

# Save estimated system model to a single CSV
estimate_csv_file_path = 'estimated_system_model.csv'
with open(estimate_csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers
    csv_writer.writerow(['Time (sec)'] + [f'State {i} (Estimate)' for i in range(n)])
    
    # Write estimated values (averaged across nodes) for each time step
    for k in range(len(t)):
        estimated_values = np.mean(x_kf[:, k, :], axis=1)  # Average across nodes
        csv_writer.writerow([t[k]] + list(estimated_values))
    
print(f"Estimated system model saved to {estimate_csv_file_path}")

# Save 3D plot data (sensor movements and locations) to a CSV
csv_3d_plot_file_path = '3d_plot_data.csv'

with open(csv_3d_plot_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers
    csv_writer.writerow(['Time (sec)', 'Sensor Type', 'Sensor Index', 'X Position', 'Y Position', 'Z Position'])
    
    # Write static sensor positions (constant over time)
    for i, sensor in enumerate(static_sensors):
        csv_writer.writerow([0, 'Static', i, sensor[0], sensor[1], sensor[2]])
    
    # Write mobile sensor trajectory over time
    for k, position in enumerate(mobile_sensor_trajectory):
        csv_writer.writerow([t[k], 'Mobile', 0, position[0], position[1], position[2]])
    
    # Write true satellite trajectory over time
    for k in range(len(t)):
        csv_writer.writerow([t[k], 'Satellite (True)', 0, x[0, k], x[2, k], target_height])
    
    # Write estimated satellite trajectory (averaged across nodes) over time
    estimated_trajectory = np.mean(x_kf, axis=2)
    for k in range(len(t)):
        csv_writer.writerow([t[k], 'Satellite (Estimate)', 0, estimated_trajectory[0, k], estimated_trajectory[2, k], target_height])

print(f"3D plot data saved to {csv_3d_plot_file_path}")

# Save mean RMSE per time step to a CSV
mean_rmse_csv_file_path = 'dest_mean_rmse_per_timestep.csv'
with open(mean_rmse_csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers
    csv_writer.writerow(['Time (sec)', 'Mean RMSE'])
    
    # Write mean RMSE for each time step
    for k in range(1, len(t)):
        mean_rmse = np.mean(rmse_adldkf_state[k])
        csv_writer.writerow([t[k], mean_rmse])

print(f"Mean RMSE per time step saved to {mean_rmse_csv_file_path}")
