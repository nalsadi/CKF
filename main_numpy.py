import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)

# Simulation parameters
tf = 200
dt = 0.7
t = np.arange(0, tf + dt, dt)
n = 4
m = 3

# Orbital parameters
center_x = 35
center_y = 30
orbital_radius = 15
orbital_period = 120
angular_velocity = 2 * np.pi / orbital_period
target_height = 0.5
sensor_height = 0.2

# Initial satellite state
initial_angle = np.pi / 4
pos_x = center_x + orbital_radius * np.cos(initial_angle)
pos_y = center_y + orbital_radius * np.sin(initial_angle)
vel_x = -orbital_radius * angular_velocity * np.sin(initial_angle)
vel_y = orbital_radius * angular_velocity * np.cos(initial_angle)
x0_true = np.array([pos_x, vel_x, pos_y, vel_y], dtype=np.float32)

# Set up sensor network
num_static_sensors = 8
sensor_radius = 20
static_sensors = [
    [
        center_x + sensor_radius * np.cos(2 * np.pi * i / num_static_sensors),
        center_y + sensor_radius * np.sin(2 * np.pi * i / num_static_sensors),
        sensor_height
    ]
    for i in range(num_static_sensors)
]
mobile_sensor = [15, 10, sensor_height]
mobile_sensor_vel = [0.05, 0.05, 0]
sensor_positions = [static_sensors + [mobile_sensor]]
num_nodes = len(static_sensors) + 1

for k in range(1, len(t)):
    current_positions = []
    for i, sensor in enumerate(sensor_positions[-1]):
        if i == num_nodes - 1:
            new_pos = [
                sensor[0] + mobile_sensor_vel[0] * dt,
                sensor[1] + mobile_sensor_vel[1] * dt,
                sensor[2] + mobile_sensor_vel[2] * dt
            ]
            current_positions.append(new_pos)
        else:
            current_positions.append(sensor.copy())
    sensor_positions.append(current_positions)

mobile_sensor_trajectory = np.array([sensor_positions[k][num_nodes - 1] for k in range(len(t))])

# Process and measurement noise
Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6]).astype(np.float32)
R_base = np.diag([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2]).astype(np.float32)

# Generate variant measurement noise for each node's sensors
base_var = np.array([0.1**2, np.deg2rad(1)**2, np.deg2rad(1)**2])
noise_variation = 0.5
node_variances = [
    base_var * (1 + np.random.uniform(-noise_variation, noise_variation, size=m))
    for _ in range(num_nodes)
]
R_list = [np.diag(var) for var in node_variances]
worst_variances = np.max(node_variances, axis=0)
R_worst = np.diag(worst_variances)

# Satellite motion model function
def satellite_motion_model(x, dt):
    pos_x, vel_x, pos_y, vel_y = x
    rx = pos_x - center_x
    ry = pos_y - center_y
    r = np.sqrt(rx**2 + ry**2)
    angular_velocity = 2 * np.pi / orbital_period
    ax = -rx * angular_velocity**2
    ay = -ry * angular_velocity**2
    new_pos_x = pos_x + vel_x * dt + 0.5 * ax * dt**2
    new_vel_x = vel_x + ax * dt
    new_pos_y = pos_y + vel_y * dt + 0.5 * ay * dt**2
    new_vel_y = vel_y + ay * dt
    return np.array([new_pos_x, new_vel_x, new_pos_y, new_vel_y])

# Measurement model
def measurement_model(x, sensor_pos):
    dx = x[0] - sensor_pos[0]
    dy = x[2] - sensor_pos[1]
    dz = target_height - sensor_pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    psi = np.arctan2(dy, dx)
    theta = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    return np.array([r, psi, theta])

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

# Generate true satellite trajectory
x = np.zeros((n, len(t)))
x[:, 0] = x0_true
z = np.zeros((m, len(t), num_nodes))
w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T

for k in range(1, len(t)):
    fault_time_index = len(t) // 1
    is_fault_active = k >= fault_time_index
    if is_fault_active:
        fault_bias = np.array([0.05, 0.003, -0.04, 0.002], dtype=np.float32)
        fault_process_noise_multiplier = 5.0
        x[:, k] = satellite_motion_model(x[:, k-1], dt) + fault_bias + w[:, k] * fault_process_noise_multiplier
    else:
        x[:, k] = satellite_motion_model(x[:, k-1], dt) + w[:, k]
    for node in range(num_nodes):
        sensor_pos = sensor_positions[k][node]
        true_measurement = measurement_model(x[:, k], sensor_pos)
        v = np.random.multivariate_normal(mean=np.zeros(m), cov=np.diag(node_variances[node]))
        z[:, k, node] = true_measurement + v

# Initialize variables for tracking
x_kf = np.zeros((n, len(t), num_nodes))
P_kf = np.array([np.eye(n) for _ in range(num_nodes)])
squared_error_kf = np.zeros((n, len(t), num_nodes))

for node in range(num_nodes):
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
# t, n, m, num_nodes, z, z_histories, sensor_positions, x_kf, P_kf, satellite_motion_model,
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
lstm_models = [create_lstm_model(input_shape, hidden_dim, output_dim) for _ in range(num_nodes)]

# Initialize measurement history for each node
z_histories = [np.zeros((window_size, m), dtype=np.float32) for _ in range(num_nodes)]

# Initialize TensorFlow MSE loss function
mse_loss_fn = MeanSquaredError()

# Initialize a dictionary to store training losses for each node
training_losses = {node: [] for node in range(num_nodes)}

# Simulation loop with LSTM-based Q, R, and delta prediction
for k in range(1, len(t)):
    for node in range(num_nodes):
        sensor_pos = sensor_positions[k][node]
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
            other_weights = [lstm_models[other_node].get_weights() for other_node in range(num_nodes) if other_node != node]
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
            satellite_motion_model, measurement_model, sensor_pos, delta
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
    for node in range(num_nodes):
        x_kf[:, k, node] = mean_state
        P_kf[node] = mean_P
        squared_error_kf[:, k, node] = (x[:, k] - mean_state)**2  # Compute squared error during consensus

# Plot training loss for a random node
random_node = np.random.randint(0, num_nodes)
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
