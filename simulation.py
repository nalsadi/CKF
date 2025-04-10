import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 
from filterpy.kalman import ExtendedKalmanFilter
from system_model_write import object_motion_model, T, n, m,t,dt, target_height, Q, R,measurement_model
from ekf_rewrite import filterpy_motion_model, filterpy_measurement_model, filterpy_jacobian_motion, filterpy_jacobian_measurement
from lstm_rewrite import QREstimatorLSTM, sigmoid_saturation, ekf, delta_opt, Q_opt, R_opt, lstm_model, input_size, hidden_size, output_size,P_kf, delta_opt, Q_opt, R_opt, lstm_model, input_size, hidden_size, output_size,P_kf 
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

w = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=len(t)).T
v = np.random.multivariate_normal(mean=np.zeros(m), cov=R, size=len(t)).T
x = np.zeros((n, len(t)))
z = np.zeros((m, len(t)))
u = np.zeros(len(t))


x_kf = np.zeros((n, len(t)))
x_filterpy = np.zeros((n, len(t)))

x_true = np.array([[25e3], [-120], [10e3], [0]])
x[:, 0] = x_true.flatten()
x_kf[:, 0] = x_true.flatten()
x_filterpy[:, 0] = x_true.flatten()


P_kf[:, :, 0] = 10 * Q


# Initialize filterpy EKF
filterpy_ekf = ExtendedKalmanFilter(dim_x=n, dim_z=m)
filterpy_ekf.x = x_true.flatten()
filterpy_ekf.P = 10 * Q
filterpy_ekf.Q = Q
filterpy_ekf.R = R
filterpy_ekf.F = filterpy_jacobian_motion(filterpy_ekf.x, T)

# Initialize filterpy ASSF
filterpy_assf = ExtendedKalmanFilter(dim_x=n, dim_z=m)
filterpy_assf.x = x_true.flatten()
filterpy_assf.P = 10 * Q
filterpy_assf.Q = Q
filterpy_assf.R = R
filterpy_assf.F = filterpy_jacobian_motion(filterpy_assf.x, T)

class QREstimatorLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.lstm = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_dim, input_shape=(None, input_dim)),
            tf.keras.layers.Dense(output_dim, activation='softplus')  # Ensure positive values
        ])
        
# Initialize the LSTM model
window_size = 10  # Number of past measurements to consider
hidden_dim = 32*4
output_dim = n + m  # Dimensions for Q (n×n) and R (m×m) diagonals
lstm_model = QREstimatorLSTM(m, hidden_dim, output_dim)
# Use learning rate scheduling
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# --- Simulate True Trajectory and Measurements ---
for k in range(len(t) - 1):  # Adjust loop range to avoid index out of bounds
    x[:, k+1] = object_motion_model(x[:, k], dt, k+1) + w[:, k]
    
    z[:, k+1] = measurement_model(x[:, k + 1], [0,0,0]) + v[:, k]
    

    # Update filterpy EKF ----------------------------------------------------------------------------------------------------------
    filterpy_ekf.predict()
    filterpy_ekf.update(
        z[:, k+1],
        HJacobian=filterpy_jacobian_measurement,
        Hx=filterpy_measurement_model
    )
    x_filterpy[:, k+1] = filterpy_ekf.x
    #--------------------------------------------------------------------------------------------------
    

    # Update filterpy ASSF ----------------------------------------------------------------------------------------------------------    
    if k % 20 == 0:
        print("Before Prediction:", np.diag(filterpy_assf.Q))
        z_input = z[:, -window_size:]
        z_input = z_input.reshape(1, window_size, m)

        for epoch in range(10):
            with tf.GradientTape() as tape:
                lstm_output = lstm_model.lstm(z_input, training=True)
                q_diag = lstm_output[0, :n]
                r_diag = lstm_output[0, n:]

                # Stay in TensorFlow
                Q_tf = tf.linalg.diag(q_diag)
                R_tf = tf.linalg.diag(r_diag)

                filterpy_assf.Q = np.diag(q_diag)
                filterpy_assf.R = np.diag(r_diag)   

                # Assume filterpy_assf.x and filterpy_assf.P are numpy arrays
                H = tf.convert_to_tensor(filterpy_jacobian_measurement(filterpy_assf.x), dtype=tf.float32)
                x_tf = tf.convert_to_tensor(filterpy_assf.x, dtype=tf.float32)
                P_tf = tf.convert_to_tensor(filterpy_assf.P, dtype=tf.float32)

                z_pred = tf.convert_to_tensor(filterpy_measurement_model(filterpy_assf.x), dtype=tf.float32)
                innovation = tf.convert_to_tensor(z[:, k+1], dtype=tf.float32) - z_pred

                S = H @ P_tf @ tf.transpose(H) + R_tf

                S_inv = tf.linalg.inv(S)
                nis = tf.matmul(tf.matmul(tf.expand_dims(innovation, axis=0), S_inv), tf.expand_dims(innovation, axis=1))

                loss = nis #nis[0, 0] + 0.1 * (tf.reduce_sum(q_diag) + tf.reduce_sum(r_diag))

            grads = tape.gradient(loss, lstm_model.lstm.trainable_variables)
            optimizer.apply_gradients(zip(grads, lstm_model.lstm.trainable_variables))

            print("Loss:", loss.numpy())
 
    print("After Training:", np.diag(filterpy_assf.Q))
        
    filterpy_assf.predict()
    filterpy_assf.update(
        z[:, k+1],
        HJacobian=filterpy_jacobian_measurement,
        Hx=filterpy_measurement_model
    )
    x_kf[:, k+1] = filterpy_assf.x
    
    #--------------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(x[0, :], x[2, :], 'r-', linewidth=2, label='True Trajectory')
plt.plot(x_kf[0, :], x_kf[2, :], 'b--', label='ASSF Position')  # Fixed indexing
plt.plot(x_filterpy[0, :], x_filterpy[2, :], 'g-.', label='FilterPy EKF Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE for position and velocity only (since we have 4 state dimensions)
rmse_assf = np.sqrt(np.mean((x[:, :-1] - x_kf[:, :-1])**2, axis=1))
rmse_filterpy = np.sqrt(np.mean((x[:, :-1] - x_filterpy[:, :-1])**2, axis=1))

states = ['X Position', 'X Velocity', 'Y Position', 'Y Velocity']  # Match state dimensions
results_rmse = pd.DataFrame({
    'State': states,
    'ASSF RMSE': rmse_assf,
    'FilterPy EKF RMSE': rmse_filterpy
})
print(results_rmse)
