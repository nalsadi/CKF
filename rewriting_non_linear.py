import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 
from filterpy.kalman import ExtendedKalmanFilter
from system_model_write import object_motion_model, T, n, m,t,dt, target_height, Q, R,measurement_model
from ekf_rewrite import filterpy_motion_model, filterpy_measurement_model, filterpy_jacobian_motion, filterpy_jacobian_measurement
from lstm_rewrite import QREstimatorLSTM, sigmoid_saturation, ekf


np.random.seed(42)





w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T
v = np.random.multivariate_normal(mean=np.zeros(m), cov=R, size=len(t)).T

x = np.zeros((n, len(t)))
z = np.zeros((m, len(t)))
u = np.zeros(len(t))


x_kf = np.zeros((n, len(t)))
x_true = np.array([[25e3], [-120], [10e3], [0]])
x[:, 0] = x_true.flatten()
x_kf[:, 0] = x_true.flatten()
P_kf = np.zeros((n, n, len(t)))
P_kf[:, :, 0] = 10 * Q
squared_error_assf = np.zeros((n, len(t)))

delta_opt = np.zeros(m)
Q_opt = np.diag([0, 0, 0]).astype(np.float32)
R_opt = np.diag([0, 0, 0]).astype(np.float32)

# Initialize LSTM
input_size = m  # Number of measurements as input
hidden_size = 32  # Hidden state size of LSTM
output_size = n + m + m  # Output is diagonal of Q, R, and delta
lstm_model = QREstimatorLSTM(input_size, hidden_size, output_size)

# Initialize filterpy EKF
filterpy_ekf = ExtendedKalmanFilter(dim_x=n, dim_z=m)
filterpy_ekf.x = x_true.flatten()
filterpy_ekf.P = 10 * Q
filterpy_ekf.Q = Q
filterpy_ekf.R = R

# Simulate filterpy EKF
x_filterpy = np.zeros((n, len(t)))
x_filterpy[:, 0] = x_true.flatten()

# --- Simulate True Trajectory and Measurements ---
for k in range(len(t) - 1):  # Adjust loop range to avoid index out of bounds
    x[:, k+1] = object_motion_model(x[:, k], dt, k+1) + w[:, k]
    
    z[:, k+1] = measurement_model(x[:, k + 1], [0,0,0]) + v[:, k]
    
    # Update filterpy EKF ----------------------------------------------------------------------------------------------------------
    filterpy_ekf.F = filterpy_jacobian_motion(filterpy_ekf.x, T)
    filterpy_ekf.predict()
    filterpy_ekf.update(
        z[:, k+1],
        HJacobian=filterpy_jacobian_measurement,
        Hx=filterpy_measurement_model
    )
    x_filterpy[:, k+1] = filterpy_ekf.x
    #--------------------------------------------------------------------------------------------------
    # Every 100 steps, use LSTM to estimate Q and R
    if k % 10 == 0:
        for epoch in range(100):
            z_input = z[:, :k+1].T
            z_input_normalized = (z_input - np.mean(z_input, axis=0)) / (np.std(z_input, axis=0) + 1e-8)
            lstm_input = z_input_normalized.T

            # Forward pass
            lstm_output = lstm_model.forward(lstm_input)

            q_diag = lstm_output[:n]  # First 4 elements for Q (state dimension)
            r_diag = lstm_output[n:n+m]  # Next 3 elements for R (measurement dimension)
            delta_out = lstm_output[n+m:]  # Last 3 elements for delta

            q_diag = np.log(1 + np.exp(q_diag))
            r_diag = np.log(1 + np.exp(r_diag))
            delta_out = np.log(1 + np.exp(delta_out))

            Q_new = np.diag(q_diag)
            R_new = np.diag(r_diag)
            delta_opt = delta_out

            
            # Perform EKF prediction and update
            # ASSF
            x_kf_new, P_kf_new, z_pred_new = ekf(k,x_kf[:, k-1], z[:, k+1], P_kf[:, :, k-1], Q_new, R_new,object_motion_model,measurement_model,[0,0,0], delta_opt)
            
            # Compute loss
            loss = np.mean((z_pred_new - z[:, k+1])**2)
            print(f"Time step {k}: Loss = {loss}")
            
            # Compute gradients with correct shapes
            output_grad = 2 * (z_pred_new - z[:, k+1])  # Shape: (3,)

            # Expand output gradient to match output size (n + m + m = 10)
            full_grad = np.zeros(output_size)  # Shape: (10,)
            full_grad[:n] = np.tile(output_grad.mean(), n)  # First n elements for Q
            full_grad[n:n+m] = output_grad  # Next m elements for R
            full_grad[n+m:] = output_grad  # Last m elements for delta
            
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
       # exit()
    # Replace the ssif line with ekf
    x_kf[:, k], P_kf[:, :, k], _ = ekf(k, x_kf[:, k-1], z[:, k+1], P_kf[:, :, k-1], Q_opt, R_opt, 
                                      object_motion_model, measurement_model, [0,0,0], delta_opt)
    squared_error_assf[:, k+1] = (x[:, k+1] - x_kf[:, k])**2




plt.figure(figsize=(10, 6))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')
plt.plot(x_kf[0, :], x_kf[2, :], 'b--', label='Kalman Filter Position')  # Fixed indexing
plt.plot(x_filterpy[0, :], x_filterpy[2, :], 'g-.', label='FilterPy EKF Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE for position and velocity only (since we have 4 state dimensions)
rmse_assf = np.sqrt(np.mean(squared_error_assf[:, :-1], axis=1))
rmse_filterpy = np.sqrt(np.mean((x[:, :-1] - x_filterpy[:, :-1])**2, axis=1))

states = ['X Position', 'X Velocity', 'Y Position', 'Y Velocity']  # Match state dimensions
results_rmse = pd.DataFrame({
    'State': states,
    'ASSF RMSE': rmse_assf,
    'FilterPy EKF RMSE': rmse_filterpy
})
print(results_rmse)
