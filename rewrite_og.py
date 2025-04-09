import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def kf_filter(x, z, u, P, A, B, C, Q, R):
    x_pred = A @ x + B.flatten() * u
    P_pred = A @ P @ A.T + Q
    innov = z - C @ x_pred
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.pinv(S)
    x_updated = x_pred + K @ innov
    P_updated = (np.eye(A.shape[0]) - K @ C) @ P_pred
    return x_updated, P_updated

def sigmoid_saturation(innov, delta, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * (np.abs(innov) - delta)))

def ssif(x, z, u, P, A, B, C, Q, R, delta):
    n = x.shape[0]
    m = z.shape[0]
    sat = np.zeros(m)
    x_pred = A @ x + B.flatten() * u
    P_pred = A @ P @ A.T + Q
    innov = z - C @ x_pred
    sat = sigmoid_saturation(innov, delta, alpha=1.0)
    K = np.linalg.pinv(C) @ np.diag(sat)
    x_updated = x_pred + K @ innov
    P_updated = (np.eye(n) - K @ C) @ P_pred @ (np.eye(n) - K @ C).T + K @ R @ K.T
    z_pred = C @ x_updated
    return x_updated.flatten(), P_updated, z_pred

tf = 2
T = 1e-3
t = np.arange(0, tf + T, T)
n = 3
m = 3

A = np.array([[1, T, 0],
              [0, 1, T],
              [-557.02, -28.616, 0.9418]], dtype=np.float32)
B = np.array([[0], [0], [557.02]], dtype=np.float32)
C = np.eye(m, dtype=np.float32)
Q = np.diag([1e-5, 1e-3, 1e-1]).astype(np.float32)
R = np.diag([1e-4, 1e-2, 1]).astype(np.float32)

np.random.seed(0)
w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T
v = np.random.multivariate_normal(mean=np.zeros(m), cov=R, size=len(t)).T

x = np.zeros((n, len(t)))
z = np.zeros((m, len(t)))
u = np.zeros(len(t))
bias = np.array([0.1, -0.05, 0.2], dtype=np.float32)

step_duration = len(t) // 4
u[:step_duration] = 0.5
u[step_duration:2*step_duration] = -0.5
u[2*step_duration:3*step_duration] = 0.5
u[3*step_duration:] = -0.5

x_kf = np.zeros((n, len(t)))
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

for k in range(1, len(t) - 1):
    x[:, k+1] = A @ x[:, k] + (B.flatten() * u[k]) + w[:, k]
    z[:, k+1] = C @ x[:, k+1] + v[:, k+1]
    
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
            x_kf_new, P_kf_new, z_pred_new = ssif(x_kf[:, k-1], z[:, k+1], u[k], P_kf[:, :, k-1], A, B, C, Q_new, R_new, delta_opt)

            # Compute loss
            loss = np.mean((z_pred_new - z[:, k+1])**2)
            
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

plt.figure(figsize=(10, 6))
plt.plot(t, x[0, :], label='True Position')
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
