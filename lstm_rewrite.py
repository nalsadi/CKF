from system_model_write import object_motion_model, measurement_model, T, n, m,t
import numpy as np
import torch


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
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.pinv(S)
    x_updated = x_pred + K @ (sat * innov)
    P_updated = (np.eye(n) - K @ H) @ P_pred
    return x_updated, P_updated, z_pred

