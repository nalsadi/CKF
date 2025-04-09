import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 


np.random.seed(42)


tf = 500
dt = 1  # Time interval (seconds)
T = dt  # Ensure consistency
t = np.arange(0, tf + dt, dt)  # Update time array
n = 4
m = 3



class QREstimatorLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_ih = np.random.randn(hidden_size, input_size) * 0.01  # Input-to-hidden weights
        self.weights_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden-to-hidden weights
        self.bias_h = np.zeros(hidden_size)  # Hidden bias
        self.weights_fc = np.random.randn(output_size, hidden_size) * 0.01  # Fully connected layer weights
        self.bias_fc = np.zeros(output_size)  # Fully connected layer bias

    def forward(self, x):
        h = np.zeros(self.hidden_size)  # Initialize hidden state
        for t in range(x.shape[1]):
            h = np.tanh(np.dot(self.weights_ih, x[:, t]) + np.dot(self.weights_hh, h) + self.bias_h)
        out = np.dot(self.weights_fc, h) + self.bias_fc
        return np.log(1 + np.exp(out))  # Softplus activation


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




def sigmoid_saturation(innov, delta, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * (np.abs(innov) - delta)))

def ekf(k,x, z, P, Q, R, motion_model, measurement_model, sensor_pos, delta, alpha=1.0):
    n = len(x)
    m = len(z)
    x_pred = motion_model(x, T,k)
    F = np.zeros((n, n))
    epsilon = 1e-4
    for j in range(n):
        x_perturbed = x.copy()
        x_perturbed[j] += epsilon
        F[:, j] = (motion_model(x_perturbed, T,k) - x_pred) / epsilon
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
    print(delta)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.pinv(S)
    x_updated = x_pred + K @ (sat * innov)
    P_updated = (np.eye(n) - K @ H) @ P_pred
    return x_updated, P_updated, z_pred


L1 = 0.16
Q = L1 * np.array([
    [T**3 / 3, T**2 / 2, 0, 0],
    [T**2 / 2, T, 0, 0],
    [0, 0, T**3 / 3, T**2 / 2],
    [0, 0, T**2 / 2, T]
])
R = np.diag([500**2, np.deg2rad(1)**2, np.deg2rad(1)**2])  # Noise for [r, psi, theta]

w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T
v = np.random.multivariate_normal(mean=np.zeros(m), cov=R, size=len(t)).T

x = np.zeros((n, len(t)))
z = np.zeros((m, len(t)))
u = np.zeros(len(t))


x_kf = np.zeros((n, len(t)))
x_true = np.array([[25e3], [-120], [10e3], [0]])
x[:, 0] = x_true.flatten()

P_kf = np.zeros((n, n, len(t)))
P_kf[:, :, 0] = 10 * Q
squared_error_assf = np.zeros((n, len(t)))

delta_opt = np.zeros(m)
Q_opt = np.diag([0, 0, 0]).astype(np.float32)
R_opt = np.diag([0, 0, 0]).astype(np.float32)

# Initialize LSTM
input_size = m  # Number of measurements as input
hidden_size = 32  # Hidden state size of LSTM
output_size = n + m + n  # Output is diagonal of Q, R, and delta
lstm_model = QREstimatorLSTM(input_size, hidden_size, output_size)



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

# --- Simulat
for k in range(len(t) - 1):  # Adjust loop range to avoid index out of bounds
    x[:, k+1] = object_motion_model(x[:, k], dt, k+1) + w[:, k]
    
    z[:, k+1] = measurement_model(x[:, k + 1], [0,0,0]) + v[:, k]
    
    
    
    # Every 100 steps, use LSTM to estimate Q and R
    if k % 100 == 0:
        for epoch in range(2):
            z_input = z[:, :k+1].T
            z_input_normalized = (z_input - np.mean(z_input, axis=0)) / (np.std(z_input, axis=0) + 1e-8)
            lstm_input = z_input_normalized.T

            # Forward pass
            lstm_output = lstm_model.forward(lstm_input)

            q_diag = lstm_output[:n]
            r_diag = lstm_output[n:2*n]
            delta_out = lstm_output[2*n:]

            q_diag = np.log(1 + np.exp(q_diag))
            r_diag = np.log(1 + np.exp(r_diag))
            delta_out = np.log(1 + np.exp(delta_out))

            Q_new = np.diag(q_diag)
            R_new = np.diag(r_diag)
            delta_opt = delta_out

            # ASSF
            x_kf_new, P_kf_new, z_pred_new = ekf(k,x_kf[:, k-1], z[:, k+1], P_kf[:, :, k-1], Q_new, R_new,object_motion_model,measurement_model,[0,0,0], delta_opt)
            
            # x_updated, P_updated, z_pred = ekf(
          #  x_kf[:, k-1], z_node, P_kf[node], Q_pred, R_pred,object_motion_model, measurement_model, sensor_pos, delta
        
            # Compute loss
            '''loss = np.mean((z_pred_new - z[:, k+1])**2)
            
            # Compute gradients with correct shapes
            output_grad = 2 * (z_pred_new - z[:, k+1])  # Shape: (3,)
            
            # Expand output gradient to match output size
            full_grad = np.zeros(output_size)  # Shape: (9,)
            full_grad[:m] = output_grad  # First m elements for Q
            full_grad[m:2*m] = output_grad  # Next m elements for R
            full_grad[2*m:] = output_grad  # Last m elements for delta
            
            # Backpropagate through fully connected layer
            h_last = np.tanh(np.dot(lstm_model.weights_ih, lstm_input[:, -1]) + 
                           np.dot(lstm_model.weights_hh, np.zeros(lstm_model.hidden_size)) + 
                           lstm_model.bias_h)
            
            # Compute gradients with correct shapes
            lr = 1e-6
            fc_grad = np.outer(full_grad, h_last)  # Shape: (9, 32)
            lstm_model.weights_fc -= lr * fc_grad
            lstm_model.bias_fc -= lr * full_grad

        # Update filter parameters
        x_kf[:, k] = x_kf_new
        P_kf[:, :, k] = P_kf_new
        Q_opt = Q_new
        R_opt = R_new
        delta_opt = delta_out

    x_kf[:, k], P_kf[:, :, k], _ = ssif(x_kf[:, k-1], z[:, k+1], u[k], P_kf[:, :, k-1], A, B, C, Q_opt, R_opt, delta_opt)
    squared_error_assf[:, k+1] = (x[:, k+1] - x_kf[:, k])**2
'''
plt.figure(figsize=(10, 6))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('True Trajectory')
plt.legend()
plt.grid(True)
plt.show()
print(x[0, :])
exit()
plt.plot(t[:-1], x_kf[0, :-1], label='Kalman Filter Position')
plt.xlabel('Time (sec)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()

rmse_assf = np.sqrt(np.mean(squared_error_assf[:, :-1], axis=1))
states = ['Position', 'Velocity', 'Acceleration']
results_rmse = pd.DataFrame({
    'State': states,
    'ASSF RMSE': rmse_assf,
})
print(results_rmse)
