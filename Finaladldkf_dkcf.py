import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simulation parameters
tf = 2  # Final time in simulation (seconds)
T = 1e-3  # Sample rate (seconds)
t = np.arange(0, tf + T, T)  # Time vector
n = 3  # Number of states
m = 3  # Number of measurements
num_nodes = 3  # Number of nodes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# System matrices as PyTorch tensors
A_np = np.array([[1, T, 0],
                 [0, 1, T],
                 [-557.02, -28.616, 0.9418]], dtype=np.float32)
A = torch.tensor(A_np, dtype=torch.float32, device=device)
Ahat = A.clone()
B_np = np.array([[0], [0], [557.02]], dtype=np.float32)
B = torch.tensor(B_np, dtype=torch.float32, device=device)
C_np = np.eye(m, dtype=np.float32)
C = torch.tensor(C_np, dtype=torch.float32, device=device)
Q_np = np.diag([1e-5, 1e-3, 1e-1]).astype(np.float32)
Q = torch.tensor(Q_np, dtype=torch.float32, device=device)

# Define unique measurement noise covariance matrices for each node
R_base = np.diag([1e-4, 1e-2, 1]).astype(np.float32)
R_nodes_np = [R_base * (1 + 0.5 * i) for i in range(num_nodes)]
R_nodes = [torch.tensor(R_node_np, dtype=torch.float32, device=device) for R_node_np in R_nodes_np]

# Input signal
u = torch.zeros(len(t), dtype=torch.float32, device=device)
step_duration = len(t) // 4
u[:step_duration] = 0.5  # First step
u[step_duration:2*step_duration] = -0.5  # Second step
u[2*step_duration:3*step_duration] = 0.5  # Third step
u[3*step_duration:] = -0.5  # Fourth step

# Define LSTM for parameter estimation
class LSTMEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMEstimator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Initialize LSTM models, optimizers, and loss functions for each node
lstm_models = []
optimizers = []
criterions = []

for _ in range(num_nodes):
    lstm_model = LSTMEstimator(input_dim=m, hidden_dim=32, output_dim=2 * n + 1).to(device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    lstm_models.append(lstm_model)
    optimizers.append(optimizer)
    criterions.append(criterion)

# Define nonlinear system dynamics and measurement functions that preserve gradients
def f_nonlinear(x, u):
    # Example nonlinear dynamics that preserves gradients
    x_next = torch.zeros_like(x)
    x_next[0] = x[0] + T * x[1]
    x_next[1] = x[1] + T * (-x[0] + torch.sin(x[2]))
    x_next[2] = x[2] + T * u
    return x_next

def h_nonlinear(x):
    # Example nonlinear measurement function that preserves gradients
    z = torch.zeros(3, device=device)
    z[0] = x[0]
    z[1] = torch.sin(x[1])
    z[2] = x[2]**2
    return z

# Jacobians of the nonlinear functions
def jacobian_f(x, u):
    # Jacobian of f_nonlinear with respect to x
    return torch.tensor([
        [1, T, 0],
        [-T, 1, T * torch.cos(x[2])],
        [0, 0, 1]
    ], device=device)

def jacobian_h(x):
    # Jacobian of h_nonlinear with respect to x
    return torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x[1]), 0],
        [0, 0, 2 * x[2]]
    ], device=device)

# Modified Extended Kalman Filter (EKF) function
def ekf_filter(x, z, u, P, Q, R):
    # Create a differentiable version of x
    x_tensor = x.clone().detach().requires_grad_(True)
    
    # Predict step
    F = jacobian_f(x_tensor, u)
    x_pred = f_nonlinear(x_tensor, u)
    P_pred = F @ P @ F.T + Q

    # Update step
    H = jacobian_h(x_pred)
    z_pred = h_nonlinear(x_pred)
    innov = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ torch.linalg.pinv(S)
    x_updated = x_pred + K @ innov
    P_updated = (torch.eye(n, device=device) - K @ H) @ P_pred

    return x_updated, P_updated

# Initialize variables
x = torch.zeros((n, len(t)), dtype=torch.float32, device=device)
w = torch.tensor(np.random.multivariate_normal(mean=np.zeros(n), cov=Q_np, size=len(t)).T, dtype=torch.float32, device=device)

# State and covariance tracking for each node
x_kf = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device)
P_kf = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)
squared_error_kf = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device)

# Regular KF for comparison for each node
x_kf_reg = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device)
P_kf_reg = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)
squared_error_kf_reg = torch.zeros((n, len(t), num_nodes), dtype=torch.float32, device=device)

# LSTM parameters for each node
Q_opts = [Q.clone() for _ in range(num_nodes)]
R_opts = [R_nodes[i].clone() for i in range(num_nodes)]
delta_opts = [torch.tensor(1.0, device=device) for _ in range(num_nodes)]

# Define the sliding window size
window_size = 10

# Initialize measurement histories for each node
z_node_histories = [torch.zeros((m, len(t)), dtype=torch.float32, device=device) for _ in range(num_nodes)]

# Simulation loop
for k in range(1, len(t) - 1):
    # Fault injection
    if np.isclose(t[k], tf/2, atol=T/2):
        # Example fault: modify dynamics
        def f_nonlinear(x, u):
            return torch.tensor([
                x[0] + T * x[1],
                x[1] + T * (-0.5 * x[0] + torch.sin(x[2])),
                x[2] + T * u
            ], device=device)

    # True system dynamics
    x[:, k+1] = f_nonlinear(x[:, k], u[k]) + w[:, k]

    # Generate unique measurements for each node
    z_nodes = []
    for i in range(num_nodes):
        R_node_np = R_nodes_np[i]
        v_node = torch.tensor(np.random.multivariate_normal(mean=np.zeros(m), cov=R_node_np, size=1).T, dtype=torch.float32, device=device).squeeze()
        z_node = h_nonlinear(x[:, k+1]) + v_node
        z_nodes.append(z_node)
        z_node_histories[i][:, k+1] = z_node

    # Run EKF for each node and calculate error
    for node in range(num_nodes):
        z_node = z_nodes[node]
        R_node = R_nodes[node]

        # EKF
        x_kf[:, k, node], P_kf[node] = ekf_filter(
            x_kf[:, k-1, node], z_node, u[k], P_kf[node], Q_opts[node], R_opts[node]
        )
        squared_error_kf[:, k, node] = (x[:, k+1] - x_kf[:, k, node])**2

        # Regular EKF (comparison)
        x_kf_reg[:, k, node], P_kf_reg[node] = ekf_filter(
            x_kf_reg[:, k-1, node], z_node, u[k], P_kf_reg[node], Q, R_node
        )
        squared_error_kf_reg[:, k, node] = (x[:, k+1] - x_kf_reg[:, k, node])**2

        # Every 10 steps, update parameters
        if k % 10 == 0:
            Q_opts[node] = Q.clone()
            R_opts[node] = R_nodes[node].clone()
            delta_opts[node] = torch.tensor(1.0, device=device)

        # Every 100 steps, LSTM-based parameter update
        if k % 50 == 0:
            optimizer = optimizers[node]
            lstm_model = lstm_models[node]
            criterion = criterions[node]

            for epoch in range(10):
                optimizer.zero_grad()

                # Collect past measurements for the node
                z_input_tensor = z_node_histories[node][:, max(0, k + 1 - window_size):k + 1].T  # Shape: [window_size, m]

                # Normalize
                z_input_tensor_normalized = (z_input_tensor - torch.mean(z_input_tensor, dim=0)) / (torch.std(z_input_tensor, dim=0) + 1e-8)
                z_input_tensor_normalized = z_input_tensor_normalized.unsqueeze(0)  # Shape: [1, window_size, m]

                # LSTM model forward
                lstm_output = lstm_model(z_input_tensor_normalized)

                # Extract and apply LSTM parameters
                q_diag = torch.nn.functional.softplus(lstm_output[:, :n])
                r_diag = torch.nn.functional.softplus(lstm_output[:, n:2*n])
                delta_out = torch.nn.functional.softplus(lstm_output[:, 2*n:])

                Q_opt = torch.diag_embed(q_diag).squeeze(0)
                R_opt = torch.diag_embed(r_diag).squeeze(0)
                delta_opt = delta_out.squeeze()

                # Create a dummy prediction model to enable backpropagation
                x_prev = x_kf[:, k-1, node].clone().detach().requires_grad_(True)
                z_target = z_node.clone().detach()  # Target measurement
                
                # Forward pass through the filter
                x_kf_new, _ = ekf_filter(x_prev, z_target, u[k], P_kf[node], Q_opt, R_opt)
                
                # Generate predicted measurement
                z_pred = h_nonlinear(x_kf_new)
                
                # Compute loss
                loss = criterion(z_pred, z_target)
                loss.backward()
                optimizer.step()

                # Update parameters
                with torch.no_grad():
                    x_kf_eval, P_kf_eval = ekf_filter(
                        x_kf[:, k-1, node], z_node, u[k], P_kf[node],
                        Q_opt.detach(), R_opt.detach()
                    )
                    x_kf[:, k, node] = x_kf_eval
                    P_kf[node] = P_kf_eval
                    Q_opts[node] = Q_opt.detach()
                    R_opts[node] = R_opt.detach()
                    delta_opts[node] = delta_opt.detach()

    # Optional progress print
    if k % 100 == 0:
        print(f"Completed time step {k}")

# Plotting the state 0 (position) of each node for both methods
plt.figure(figsize=(12, 6))

# Plot the true state 0
plt.plot(t, x[0, :].cpu().detach().numpy(), label='True State 0', linewidth=2, color='black')

# Plot state 0 of each node using EKF
for node in range(num_nodes):
    plt.plot(t, x_kf[0, :, node].cpu().detach().numpy(), label=f'Node {node+1} EKF', linestyle='--')

# Plot state 0 of each node using regular EKF
for node in range(num_nodes):
    plt.plot(t, x_kf_reg[0, :, node].cpu().detach().numpy(), label=f'Node {node+1} Regular EKF', linestyle=':')

plt.xlabel('Time (sec)')
plt.ylabel('State 0')
plt.title('Comparison of State 0 for Each Node')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE at each time step for each method
rmse_lstm_per_t = torch.sqrt(torch.mean((x - torch.mean(x_kf, dim=2))**2, dim=0))  # Mean over states, average across nodes
rmse_regular_per_t = torch.sqrt(torch.mean((x - torch.mean(x_kf_reg, dim=2))**2, dim=0))  # Mean over states, average across nodes

# Plot RMSE over time for each method
plt.figure(figsize=(12, 6))
plt.plot(t, rmse_lstm_per_t.cpu().detach().numpy(), label='LSTM-EKF RMSE', linestyle='--', linewidth=2)
plt.plot(t, rmse_regular_per_t.cpu().detach().numpy(), label='Regular EKF RMSE', linestyle=':', linewidth=2)
plt.xlabel('Time (sec)')
plt.ylabel('RMSE')
plt.title('RMSE Over Time for Each Method')
plt.legend()
plt.grid(True)
plt.show()

# Print the final RMSE difference between the two methods
final_rmse_lstm = torch.mean(rmse_lstm_per_t).item()
final_rmse_regular = torch.mean(rmse_regular_per_t).item()
print(f"Final RMSE (LSTM-EKF): {final_rmse_lstm:.6f}")
print(f"Final RMSE (Regular EKF): {final_rmse_regular:.6f}")
print(f"Difference in RMSE: {abs(final_rmse_lstm - final_rmse_regular):.6f}")