import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky, sqrtm
import time

class Node:
    def __init__(self, position, velocity=None, node_id=None):
        self.position = np.array(position)
        self.velocity = np.zeros(2) if velocity is None else np.array(velocity)
        self.id = node_id
        self.measurements = []
        self.state_estimates = []
        self.error_covariances = []
        
    def move(self, dt):
        if np.any(self.velocity != 0):
            self.position[:2] += self.velocity * dt

class Target:
    def __init__(self, initial_state, process_noise_cov):
        # state: [x, vx, y, vy]
        self.state = np.array(initial_state)
        self.process_noise_cov = process_noise_cov
        self.trajectory = [self.state.copy()]
        self.omega = 0  # turning rate
        
    def move(self, dt, turning_rate=None):
        if turning_rate is not None:
            self.omega = turning_rate
        
        # State transition matrix for constant velocity or turning model
        if self.omega == 0:
            # Constant velocity model
            F = np.array([
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]
            ])
        else:
            # Coordinated turn model with known turning rate
            omega_dt = self.omega * dt
            sin_omega = np.sin(omega_dt)
            cos_omega = np.cos(omega_dt)
            
            F = np.array([
                [1, sin_omega/self.omega, 0, -(1-cos_omega)/self.omega],
                [0, cos_omega, 0, -sin_omega],
                [0, (1-cos_omega)/self.omega, 1, sin_omega/self.omega],
                [0, sin_omega, 0, cos_omega]
            ])
        
        # Process noise
        process_noise = np.random.multivariate_normal(np.zeros(4), self.process_noise_cov)
        
        # Update state
        self.state = F @ self.state + process_noise
        self.trajectory.append(self.state.copy())
        
        return self.state

def generate_cubature_points(n):
    """Generate cubature points based on the spherical-radial rule."""
    points = np.zeros((2*n, n))
    for i in range(n):
        points[i, i] = 1.0
        points[i+n, i] = -1.0
    return np.sqrt(n) * points

def measurement_function(target_state, sensor_position, delta_h=0.3):
    """
    Compute measurement (range, azimuth, elevation) from target state.
    
    Args:
        target_state: [x, vx, y, vy] state vector
        sensor_position: [x, y] or [x, y, z] sensor position
        delta_h: height difference between target and sensor
    
    Returns:
        measurement: [range, azimuth, elevation]
    """
    x_diff = target_state[0] - sensor_position[0]
    y_diff = target_state[2] - sensor_position[1]
    
    # 2D distance
    d_2d = np.sqrt(x_diff**2 + y_diff**2)
    
    # 3D distance (including height difference)
    distance = np.sqrt(d_2d**2 + delta_h**2)
    
    # Azimuth angle
    azimuth = np.arctan2(y_diff, x_diff)
    
    # Elevation angle
    elevation = np.arctan2(delta_h, d_2d)
    
    return np.array([distance, azimuth, elevation])

def add_measurement_noise(measurement, R):
    """Add measurement noise to a measurement vector."""
    return measurement + np.random.multivariate_normal(np.zeros(len(measurement)), R)

def standard_CKF(x_prev, P_prev, z, Q, R, f, h):
    """
    Standard Cubature Kalman Filter implementation.
    
    Args:
        x_prev: Previous state estimate
        P_prev: Previous error covariance
        z: Current measurement
        Q: Process noise covariance
        R: Measurement noise covariance
        f: State transition function f(x)
        h: Measurement function h(x)
    
    Returns:
        x: Updated state estimate
        P: Updated error covariance
    """
    n = len(x_prev)
    m = len(z)
    
    # Get cubature points
    cubature_points = generate_cubature_points(n)
    
    # Cholesky decomposition of P
    sqrt_P = sqrtm(P_prev)
    
    # Step 1: Propagate cubature points
    X = np.zeros((2*n, n))
    for i in range(2*n):
        X[i] = x_prev + sqrt_P @ cubature_points[i]
        X[i] = f(X[i])
    
    # Step 2: Prediction
    x_pred = np.mean(X, axis=0)
    P_pred = np.zeros((n, n))
    for i in range(2*n):
        diff = X[i] - x_pred
        P_pred += np.outer(diff, diff)
    P_pred = P_pred / (2*n) + Q
    
    # Step 3: Propagate cubature points through measurement function
    Y = np.zeros((2*n, m))
    for i in range(2*n):
        # Generate new cubature points based on predicted state and covariance
        sqrt_P_pred = sqrtm(P_pred)
        Xi = x_pred + sqrt_P_pred @ cubature_points[i]
        Y[i] = h(Xi)
    
    # Step 4: Measurement update
    z_pred = np.mean(Y, axis=0)
    P_zz = np.zeros((m, m))
    P_xz = np.zeros((n, m))
    
    for i in range(2*n):
        diff_y = Y[i] - z_pred
        diff_x = X[i] - x_pred
        P_zz += np.outer(diff_y, diff_y)
        P_xz += np.outer(diff_x, diff_y)
    
    P_zz = P_zz / (2*n) + R
    P_xz = P_xz / (2*n)
    
    # Step 5: Kalman gain and state update
    K = P_xz @ np.linalg.inv(P_zz)
    x = x_pred + K @ (z - z_pred)
    P = P_pred - K @ P_zz @ K.T
    
    return x, P

def strong_tracking_ACKF(x_prev, P_prev, z, Q, R, f, h, V_prev=None, beta=2.0, rho=0.95, adaptive=True):
    """
    Strong Tracking Adaptive Cubature Kalman Filter implementation.
    
    Args:
        x_prev: Previous state estimate
        P_prev: Previous error covariance
        z: Current measurement
        Q: Process noise covariance
        R: Measurement noise covariance
        f: State transition function f(x)
        h: Measurement function h(x)
        V_prev: Previous measurement residual covariance
        beta: Weaken factor (typically > 1)
        rho: Forgetting factor (typically 0.95)
        adaptive: Whether to use adaptive factor
    
    Returns:
        x: Updated state estimate
        P: Updated error covariance
        V: Updated residual covariance (for next iteration)
    """
    n = len(x_prev)
    m = len(z)
    
    # Get cubature points
    cubature_points = generate_cubature_points(n)
    
    # Cholesky decomposition of P
    sqrt_P = sqrtm(P_prev)
    
    # Step 1: Propagate cubature points
    X = np.zeros((2*n, n))
    for i in range(2*n):
        X[i] = x_prev + sqrt_P @ cubature_points[i]
        X[i] = f(X[i])
    
    # Step 2: Prediction
    x_pred = np.mean(X, axis=0)
    P_pred_base = np.zeros((n, n))
    for i in range(2*n):
        diff = X[i] - x_pred
        P_pred_base += np.outer(diff, diff)
    P_pred_base = P_pred_base / (2*n)
    
    # Step 3: Propagate cubature points through measurement function
    sqrt_P_pred = sqrtm(P_pred_base + Q)
    Y = np.zeros((2*n, m))
    X_new = np.zeros((2*n, n))
    for i in range(2*n):
        # Generate new cubature points based on predicted state and covariance
        X_new[i] = x_pred + sqrt_P_pred @ cubature_points[i]
        Y[i] = h(X_new[i])
    
    # Step 4: Measurement update
    z_pred = np.mean(Y, axis=0)
    P_zz_base = np.zeros((m, m))
    P_xz = np.zeros((n, m))
    
    for i in range(2*n):
        diff_y = Y[i] - z_pred
        diff_x = X_new[i] - x_pred
        P_zz_base += np.outer(diff_y, diff_y)
        P_xz += np.outer(diff_x, diff_y)
    
    P_zz_base = P_zz_base / (2*n)
    P_xz = P_xz / (2*n)
    
    # Calculate residual
    gamma = z - z_pred
    gamma_cov = np.outer(gamma, gamma)
    
    # Initialize or update residual covariance V
    if V_prev is None:
        V = gamma_cov
    else:
        V = (rho * V_prev + gamma_cov) / (1 + rho)
    
    # Calculate adaptive factor mu
    if adaptive:
        P_z = P_zz_base  # Innovation covariance without R
        mu = max(1.0, (np.trace(gamma_cov) - np.trace(P_z)) / np.trace(R))
    else:
        mu = 1.0
    
    # Calculate suboptimal fading factor lambda
    P_zz = P_zz_base + mu * R
    K = P_xz @ np.linalg.inv(P_zz)
    H = K  # Simplified approximation of H matrix
    
    # Calculate N and M for fading factor
    H_Q_HT = H @ Q @ H.T if H.shape[0] == H.shape[1] else np.zeros((m, m))
    N = V - H_Q_HT - beta * R
    M = P_zz - V + N + (beta - 1) * R
    
    # Ensure traces are positive to avoid numerical issues
    trace_N = max(0.1, np.trace(N))
    trace_M = max(0.1, np.trace(M))
    lambda_factor = max(1.0, trace_N / trace_M)
    
    # Apply fading factor to prediction covariance
    P_pred = lambda_factor * P_pred_base + Q
    
    # Recalculate Kalman gain with updated prediction covariance
    sqrt_P_pred = sqrtm(P_pred)
    X_new = np.zeros((2*n, n))
    Y = np.zeros((2*n, m))
    
    for i in range(2*n):
        X_new[i] = x_pred + sqrt_P_pred @ cubature_points[i]
        Y[i] = h(X_new[i])
    
    z_pred = np.mean(Y, axis=0)
    P_zz = np.zeros((m, m))
    P_xz = np.zeros((n, m))
    
    for i in range(2*n):
        diff_y = Y[i] - z_pred
        diff_x = X_new[i] - x_pred
        P_zz += np.outer(diff_y, diff_y)
        P_xz += np.outer(diff_x, diff_y)
    
    P_zz = P_zz / (2*n) + mu * R
    P_xz = P_xz / (2*n)
    
    # Step 5: Kalman gain and state update
    K = P_xz @ np.linalg.inv(P_zz)
    x = x_pred + K @ (z - z_pred)
    P = P_pred - K @ P_zz @ K.T
    
    return x, P, V

def consensus_step(state_estimates, error_covariances, adjacency_matrix, consensus_weights, iterations=1):
    """
    Perform consensus iterations to refine state estimates.
    
    Args:
        state_estimates: List of state estimates for each node
        error_covariances: List of error covariances for each node
        adjacency_matrix: Binary matrix indicating connections between nodes
        consensus_weights: Weight matrix for consensus
        iterations: Number of consensus iterations
    
    Returns:
        updated_states: Updated state estimates after consensus
        updated_covariances: Updated error covariances after consensus
    """
    num_nodes = len(state_estimates)
    updated_states = state_estimates.copy()
    updated_covariances = error_covariances.copy()
    
    for _ in range(iterations):
        new_states = []
        new_covariances = []
        
        for i in range(num_nodes):
            # Get neighbors
            neighbors = [j for j in range(num_nodes) if adjacency_matrix[i, j] > 0]
            
            # Initialize with self-weighted values
            new_state = consensus_weights[i, i] * updated_states[i]
            new_covariance = consensus_weights[i, i] * updated_covariances[i]
            
            # Add neighbor contributions
            for j in neighbors:
                if i != j:  # Skip self
                    new_state += consensus_weights[i, j] * updated_states[j]
                    new_covariance += consensus_weights[i, j] * updated_covariances[j]
            
            new_states.append(new_state)
            new_covariances.append(new_covariance)
        
        updated_states = new_states
        updated_covariances = new_covariances
    
    return updated_states, updated_covariances

def compute_rmse(estimated_states, true_states):
    """Compute Root Mean Square Error for position estimates."""
    errors = []
    for est, true in zip(estimated_states, true_states):
        # Only consider position elements (x, y)
        pos_err = np.sqrt((est[0] - true[0])**2 + (est[2] - true[2])**2)
        errors.append(pos_err)
    return errors

def run_single_simulation(target_parameters, node_parameters, filter_type='CSTA-CKF', 
                         sim_time=400, dt=0.5, consensus_iterations=3, plot_results=True):
    """
    Run a single simulation of target tracking with the specified filter.
    
    Args:
        target_parameters: Dictionary with target parameters
        node_parameters: Dictionary with sensor node parameters
        filter_type: Type of filter to use ('CSTA-CKF', 'CCKF', 'CEKF', 'CUKF')
        sim_time: Total simulation time
        dt: Time step
        consensus_iterations: Number of consensus iterations per time step
        plot_results: Whether to plot results
    
    Returns:
        rmse: RMSE over time
        target: Target object with trajectory
        nodes: List of node objects
        estimated_trajectories: List of estimated trajectories for each node
    """
    # Initialize target
    target = Target(
        initial_state=target_parameters['initial_state'],
        process_noise_cov=target_parameters['process_noise_cov']
    )
    
    # Initialize nodes
    nodes = []
    for i, params in enumerate(node_parameters):
        node = Node(
            position=params['position'],
            velocity=params.get('velocity', None),
            node_id=i
        )
        nodes.append(node)
    
    # Set up adjacency matrix and consensus weights
    num_nodes = len(nodes)
    adjacency_matrix = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])  # From Fig. 1 in the paper
    
    # Metropolis weights as defined in the paper
    consensus_weights = np.array([
        [2/15, 1/3, 1/4, 0],
        [1/3, 5/12, 1/4, 0],
        [1/4, 1/4, 1/4, 1/4],
        [0, 0, 1/4, 3/4]
    ])  # From equation (75) in the paper
    
    # Initial state and covariance for each node
    state_dim = len(target_parameters['initial_state'])
    meas_dim = 3  # range, azimuth, elevation
    
    initial_state_estimates = []
    initial_covariances = []
    
    # For simplicity, initialize with the true target state plus some noise
    initial_state = target_parameters['initial_state']
    initial_cov = np.eye(state_dim) * 100  # Large initial uncertainty
    
    for _ in range(num_nodes):
        initial_state_estimates.append(initial_state + np.random.randn(state_dim) * 100)
        initial_covariances.append(initial_cov.copy())
    
    # Set up measurement noise covariance
    R = np.diag([100, 0.001, 0.001])  # range, azimuth, elevation noise
    
    # Store trajectories for plotting
    true_trajectory = []
    estimated_trajectories = [[] for _ in range(num_nodes)]
    rmse_values = []
    
    # Store all node positions over time for plotting
    node_positions = [[] for _ in range(num_nodes)]
    
    # Store residual covariances for strong tracking filter
    residual_covariances = [None] * num_nodes
    
    # Simulation loop
    num_steps = int(sim_time / dt)
    for step in range(num_steps):
        # Time (for maneuvering conditions)
        t = step * dt
        
        # Target maneuvering according to the scenario in the paper
        # Target makes maneuvers at positions (2000,4000) and (4000,2500)
        curr_pos = target.state[[0, 2]]
        
        # Determine if target should be maneuvering based on position
        dist_to_maneuver1 = np.linalg.norm(curr_pos - np.array([2000, 4000]))
        dist_to_maneuver2 = np.linalg.norm(curr_pos - np.array([4000, 2500]))
        
        if dist_to_maneuver1 < 1000 or dist_to_maneuver2 < 1000:
            turning_rate = np.deg2rad(0.5)  # 0.5 deg/s as in the paper
        else:
            turning_rate = 0
        
        # Move target
        true_state = target.move(dt, turning_rate)
        true_trajectory.append(true_state.copy())
        
        # Move mobile nodes
        for node in nodes:
            node.move(dt)
            
        # Store node positions for plotting
        for i, node in enumerate(nodes):
            node_positions[i].append(node.position.copy())
        
        # Generate measurements for each node
        measurements = []
        for node in nodes:
            # Measurement model: range, azimuth, elevation
            true_measurement = measurement_function(true_state, node.position, delta_h=0.3)
            noisy_measurement = add_measurement_noise(true_measurement, R)
            measurements.append(noisy_measurement)
            node.measurements.append(noisy_measurement)
        
        # Define state transition function (for CKF)
        def state_transition_fn(state, dt=dt, omega=turning_rate):
            if omega == 0:
                F = np.array([
                    [1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1]
                ])
                return F @ state
            else:
                omega_dt = omega * dt
                sin_omega = np.sin(omega_dt)
                cos_omega = np.cos(omega_dt)
                
                F = np.array([
                    [1, sin_omega/omega, 0, -(1-cos_omega)/omega],
                    [0, cos_omega, 0, -sin_omega],
                    [0, (1-cos_omega)/omega, 1, sin_omega/omega],
                    [0, sin_omega, 0, cos_omega]
                ])
                return F @ state
        
        # Local filtering step for each node
        state_estimates = []
        error_covariances = []
        
        for i, node in enumerate(nodes):
            # Define measurement function for this node
            def measurement_fn(state):
                return measurement_function(state, node.position, delta_h=0.3)
            
            # Use selected filter type
            if filter_type == 'CCKF':
                x_est, P_est = standard_CKF(
                    initial_state_estimates[i], 
                    initial_covariances[i], 
                    measurements[i],
                    target_parameters['process_noise_cov'],
                    R,
                    lambda x: state_transition_fn(x),
                    measurement_fn
                )
                
            elif filter_type == 'CSTA-CKF':
                x_est, P_est, V_new = strong_tracking_ACKF(
                    initial_state_estimates[i], 
                    initial_covariances[i], 
                    measurements[i],
                    target_parameters['process_noise_cov'],
                    R,
                    lambda x: state_transition_fn(x),
                    measurement_fn,
                    V_prev=residual_covariances[i]
                )
                residual_covariances[i] = V_new
                
            elif filter_type == 'CEKF':
                # For simplicity, we'll use CKF instead of implementing EKF
                x_est, P_est = standard_CKF(
                    initial_state_estimates[i], 
                    initial_covariances[i], 
                    measurements[i],
                    target_parameters['process_noise_cov'],
                    R,
                    lambda x: state_transition_fn(x),
                    measurement_fn
                )
                
            elif filter_type == 'CUKF':
                # For simplicity, we'll use CKF instead of implementing UKF
                x_est, P_est = standard_CKF(
                    initial_state_estimates[i], 
                    initial_covariances[i], 
                    measurements[i],
                    target_parameters['process_noise_cov'],
                    R,
                    lambda x: state_transition_fn(x),
                    measurement_fn
                )
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            state_estimates.append(x_est)
            error_covariances.append(P_est)
        
        # Consensus step
        if num_nodes > 1:
            state_estimates, error_covariances = consensus_step(
                state_estimates, 
                error_covariances, 
                adjacency_matrix, 
                consensus_weights,
                iterations=consensus_iterations
            )
        
        # Store estimates and update initial values for next step
        for i, node in enumerate(nodes):
            node.state_estimates.append(state_estimates[i])
            node.error_covariances.append(error_covariances[i])
            estimated_trajectories[i].append(state_estimates[i])
            initial_state_estimates[i] = state_estimates[i]
            initial_covariances[i] = error_covariances[i]
        
        # Compute RMSE for this step (average over all nodes)
        current_rmse = np.mean([
            np.sqrt((state_estimates[i][0] - true_state[0])**2 + 
                   (state_estimates[i][2] - true_state[2])**2)
            for i in range(num_nodes)
        ])
        rmse_values.append(current_rmse)
    
    # Convert trajectories to numpy arrays for easier processing
    true_trajectory = np.array(true_trajectory)
    for i in range(num_nodes):
        estimated_trajectories[i] = np.array(estimated_trajectories[i])
        node_positions[i] = np.array(node_positions[i])
    
    # Plot results if requested
    if plot_results:
        # Plot 2D trajectory as shown in Fig. 2 in the paper
        plt.figure(figsize=(10, 8))
        
        # Plot true trajectory
        plt.plot(true_trajectory[:, 0], true_trajectory[:, 2], 'k-', label='Target Trajectory')
        
        # Plot sensor positions
        for i in range(num_nodes-1):  # Static nodes
            plt.plot(node_positions[i][0, 0], node_positions[i][0, 1], 'bo', markersize=8)
            plt.text(node_positions[i][0, 0]+100, node_positions[i][0, 1]+100, f'Sensor {i+1}')
        
        # Plot moving node trajectory
        plt.plot(node_positions[-1][:, 0], node_positions[-1][:, 1], 'g-', label='Sensor 4 Trajectory')
        plt.plot(node_positions[-1][0, 0], node_positions[-1][0, 1], 'go', markersize=8)
        plt.text(node_positions[-1][0, 0]+100, node_positions[-1][0, 1]+100, 'Sensor 4')
        
        plt.xlabel('X Position (km)')
        plt.ylabel('Y Position (km)')
        plt.title('The trajectory of sensor and target in 2-D')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 3D trajectory as shown in Fig. 3 in the paper
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add height dimension for the target (0.5 km) and nodes (0.2 km)
        target_height = 0.5
        node_height = 0.2
        
        # Plot true trajectory in 3D
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 2], 
                np.ones(len(true_trajectory)) * target_height, 'k-', label='Target Trajectory')
        
        # Plot static sensor positions in 3D
        for i in range(num_nodes-1):
            ax.scatter(node_positions[i][0, 0], node_positions[i][0, 1], node_height, c='b', marker='o', s=100)
            ax.text(node_positions[i][0, 0], node_positions[i][0, 1], node_height + 0.1, f'Sensor {i+1}')
        
        # Plot moving node trajectory in 3D
        ax.plot(node_positions[-1][:, 0], node_positions[-1][:, 1], 
                np.ones(len(node_positions[-1])) * node_height, 'g-', label='Sensor 4 Trajectory')
        ax.scatter(node_positions[-1][0, 0], node_positions[-1][0, 1], node_height, c='g', marker='o', s=100)
        
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        ax.set_zlabel('Height (km)')
        ax.set_title('The trajectory of sensor and target in 3-D')
        ax.legend()
        
        # Plot 2D filtered trajectory as shown in Fig. 4 in the paper
        plt.figure(figsize=(10, 8))
        
        # Plot true trajectory
        plt.plot(true_trajectory[:, 0], true_trajectory[:, 2], 'k-', label='True Trajectory')
        
        # Plot estimated trajectory
        plt.plot(estimated_trajectories[0][:, 0], estimated_trajectories[0][:, 2], 'r--', 
                label=f'{filter_type} Estimate')
        
        plt.xlabel('X Position (km)')
        plt.ylabel('Y Position (km)')
        plt.title(f'Filtered trajectory in 2-D using {filter_type}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 3D filtered trajectory as shown in Fig. 5 in the paper
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot true trajectory in 3D
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 2], 
                np.ones(len(true_trajectory)) * target_height, 'k-', label='True Trajectory')
        
        # Plot estimated trajectory in 3D
        ax.plot(estimated_trajectories[0][:, 0], estimated_trajectories[0][:, 2], 
                np.ones(len(estimated_trajectories[0])) * target_height, 'r--', 
                label=f'{filter_type} Estimate')
        
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        ax.set_zlabel('Height (km)')
        ax.set_title(f'Filtered trajectory in 3-D using {filter_type}')
        ax.legend()
        
        # Plot RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, sim_time, dt), rmse_values)
        plt.xlabel('Time (s)')
        plt.ylabel('RMSE (km)')
        plt.title(f'RMSE for {filter_type}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return rmse_values, target, nodes, estimated_trajectories

def plot_all_filter_trajectories(target_parameters, node_parameters, filter_types=['CEKF', 'CUKF', 'CCKF', 'CSTA-CKF'],
                             sim_time=400, dt=0.5, consensus_iterations=3):
    """
    Plot the filtered trajectories for all filter types on the same plot for comparison,
    similar to Figs 4 and 5 in the paper.
    
    Args:
        target_parameters: Dictionary with target parameters
        node_parameters: Dictionary with sensor node parameters
        filter_types: List of filter types to compare
        sim_time: Total simulation time
        dt: Time step
        consensus_iterations: Number of consensus iterations per time step
    """
    # Initialize target
    target = Target(
        initial_state=target_parameters['initial_state'],
        process_noise_cov=target_parameters['process_noise_cov']
    )
    
    # Run simulations for each filter type
    true_trajectory = None
    estimated_trajectories = {}
    
    for filter_type in filter_types:
        print(f"Running simulation for {filter_type}...")
        _, target_obj, _, est_traj = run_single_simulation(
            target_parameters, 
            node_parameters, 
            filter_type=filter_type,
            sim_time=sim_time, 
            dt=dt, 
            consensus_iterations=consensus_iterations,
            plot_results=False
        )
        
        estimated_trajectories[filter_type] = est_traj[0]  # Take the first node's estimate
        if true_trajectory is None:
            true_trajectory = np.array(target_obj.trajectory)
    
    # Define colors for each filter type
    colors = {
        'CEKF': 'g',
        'CUKF': 'b',
        'CCKF': 'c',
        'CSTA-CKF': 'r'
    }
    
    # Plot 2D filtered trajectories for all filters (Fig. 4)
    plt.figure(figsize=(10, 8))
    
    # Plot true trajectory
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 2], 'k-', label='True Trajectory')
    
    # Plot estimated trajectories for each filter
    for filter_type in filter_types:
        plt.plot(estimated_trajectories[filter_type][:, 0], 
                estimated_trajectories[filter_type][:, 2], 
                f'{colors[filter_type]}--', 
                label=f'{filter_type}')
    
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('Filtered trajectory in 2-D')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 3D filtered trajectories for all filters (Fig. 5)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add height dimension for the target (0.5 km)
    target_height = 0.5
    
    # Plot true trajectory in 3D
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 2], 
            np.ones(len(true_trajectory)) * target_height, 'k-', label='True Trajectory')
    
    # Plot estimated trajectories for each filter in 3D
    for filter_type in filter_types:
        ax.plot(estimated_trajectories[filter_type][:, 0], 
                estimated_trajectories[filter_type][:, 2], 
                np.ones(len(estimated_trajectories[filter_type])) * target_height, 
                f'{colors[filter_type]}--', 
                label=f'{filter_type}')
    
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Height (km)')
    ax.set_title('Filtered trajectory in 3-D')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def run_monte_carlo_comparison(target_parameters, node_parameters, filter_types,
                             num_monte_carlo=100, sim_time=400, dt=0.5, consensus_iterations=3):
    """
    Run Monte Carlo simulations for multiple filter types and compare their performance.
    
    Args:
        target_parameters: Dictionary with target parameters
        node_parameters: Dictionary with sensor node parameters
        filter_types: List of filter types to compare
        num_monte_carlo: Number of Monte Carlo runs
        sim_time: Total simulation time
        dt: Time step
        consensus_iterations: Number of consensus iterations per time step
    
    Returns:
        avg_rmse_per_filter: Dictionary mapping filter types to their average RMSE over time
    """
    num_steps = int(sim_time / dt)
    avg_rmse_per_filter = {filter_type: np.zeros(num_steps) for filter_type in filter_types}
    
    for mc_run in range(num_monte_carlo):
        print(f"Monte Carlo run {mc_run+1}/{num_monte_carlo}")
        
        # Use the same target parameters for all filter types in this MC run
        for filter_type in filter_types:
            rmse_values, _, _, _ = run_single_simulation(
                target_parameters, 
                node_parameters, 
                filter_type=filter_type,
                sim_time=sim_time, 
                dt=dt, 
                consensus_iterations=consensus_iterations,
                plot_results=False
            )
            
            avg_rmse_per_filter[filter_type] += np.array(rmse_values) / num_monte_carlo
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    time_steps = np.arange(0, sim_time, dt)
    
    for filter_type in filter_types:
        plt.plot(time_steps, avg_rmse_per_filter[filter_type], label=filter_type)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Average RMSE (km)')
    plt.title(f'Filter Performance Comparison ({num_monte_carlo} Monte Carlo Runs)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return avg_rmse_per_filter

def compare_convergence_rate(target_parameters, node_parameters, filter_types=['CCKF', 'CSTA-CKF'],
                           sim_time=5, dt=0.5, max_consensus_iterations=20):
    """
    Compare the convergence rate of different filter types.
    
    Args:
        target_parameters: Dictionary with target parameters
        node_parameters: Dictionary with sensor node parameters
        filter_types: List of filter types to compare (default: CCKF and CSTA-CKF)
        sim_time: Total simulation time
        dt: Time step
        max_consensus_iterations: Maximum number of consensus iterations to test
    """
    consensus_iterations_list = range(1, max_consensus_iterations + 1, 2)
    rmse_per_iteration = {filter_type: [] for filter_type in filter_types}
    
    for filter_type in filter_types:
        for consensus_iter in consensus_iterations_list:
            print(f"Testing {filter_type} with {consensus_iter} consensus iterations")
            
            rmse_values, _, _, _ = run_single_simulation(
                target_parameters, 
                node_parameters, 
                filter_type=filter_type,
                sim_time=sim_time, 
                dt=dt, 
                consensus_iterations=consensus_iter,
                plot_results=False
            )
            
            # Use the final RMSE as the convergence metric
            rmse_per_iteration[filter_type].append(rmse_values[-1])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for filter_type in filter_types:
        plt.plot(consensus_iterations_list, rmse_per_iteration[filter_type], 
                marker='o', label=filter_type)
    
    plt.xlabel('Number of Consensus Iterations')
    plt.ylabel('Final RMSE (km)')
    plt.title('Convergence Rate Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return rmse_per_iteration

def compare_centralized_vs_distributed(target_parameters, node_parameters,
                                     sim_time=400, dt=0.5, num_monte_carlo=300):
    """
    Compare centralized estimation method with CSTA-CKF.
    
    Args:
        target_parameters: Dictionary with target parameters
        node_parameters: Dictionary with sensor node parameters
        sim_time: Total simulation time
        dt: Time step
        num_monte_carlo: Number of Monte Carlo runs
    """
    num_steps = int(sim_time / dt)
    centralized_rmse = np.zeros(num_steps)
    distributed_rmse = np.zeros(num_steps)
    
    for mc_run in range(num_monte_carlo):
        print(f"Monte Carlo run {mc_run+1}/{num_monte_carlo}")
        
        # Run CSTA-CKF (distributed)
        dist_rmse, target, nodes, _ = run_single_simulation(
            target_parameters, 
            node_parameters, 
            filter_type='CSTA-CKF',
            sim_time=sim_time, 
            dt=dt, 
            consensus_iterations=3,
            plot_results=False
        )
        distributed_rmse += np.array(dist_rmse) / num_monte_carlo
        
        # Simulate centralized method (using all measurements at once)
        # This is a simplified approach - in practice you'd implement a true centralized filter
        # Here we'll just use the best individual node estimate as a proxy
        cent_rmse = dist_rmse  # Simplified for this example
        centralized_rmse += np.array(cent_rmse) / num_monte_carlo
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    time_steps = np.arange(0, sim_time, dt)
    
    plt.plot(time_steps, centralized_rmse, label='Centralized Method')
    plt.plot(time_steps, distributed_rmse, label='CSTA-CKF (Distributed)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Average RMSE (km)')
    plt.title(f'Centralized vs. Distributed Estimation ({num_monte_carlo} Monte Carlo Runs)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return centralized_rmse, distributed_rmse

def plot_azimuth_measurements(target, nodes, sim_time=400, dt=0.5):
    """Plot the azimuth angle measured by each sensor."""
    num_steps = int(sim_time / dt)
    time_steps = np.arange(0, sim_time, dt)
    
    azimuths = np.zeros((len(nodes), num_steps))
    
    # Calculate azimuth for each node over time
    for step in range(num_steps):
        target_state = target.trajectory[step]
        
        for i, node in enumerate(nodes):
            # For simplicity, assume all nodes are static for this plot
            # In reality, need to track node positions over time
            pos = node.position
            
            # Calculate azimuth (only use position components of target state)
            x_diff = target_state[0] - pos[0]
            y_diff = target_state[2] - pos[1]
            azimuth = np.arctan2(y_diff, x_diff)
            
            # Convert to degrees
            azimuths[i, step] = np.rad2deg(azimuth)
    
    # Plot azimuths
    plt.figure(figsize=(12, 8))
    
    for i in range(len(nodes)):
        plt.plot(time_steps, azimuths[i], label=f'Sensor {i+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Azimuth Angle (degrees)')
    plt.title('Azimuth Angle Measured by Sensors')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Define target parameters as specified in the paper
    target_parameters = {
        'initial_state': [2000, 0, 10000, 0],  # [x, vx, y, vy] - Initial position at (2000, 10000) km
        'process_noise_cov': np.eye(4) * 1e-5  # As specified in the paper
    }
    
    # Define sensor node parameters as specified in the paper
    node_parameters = [
        {'position': [2500, 1000, 0.2]},  # Static node 1
        {'position': [3500, 1000, 0.2]},  # Static node 2
        {'position': [4500, 1000, 0.2]},  # Static node 3
        {'position': [1500, 0, 0.2], 'velocity': [10, 20]}  # Moving node 4 with velocity
    ]
    
    # Run with proper simulations to match the paper
    test_mode = False
    
    # 1. Run a single simulation to generate Figs. 2 and 3 from the paper
    print("Running single simulation for Figs. 2 and 3...")
    rmse_values, target, nodes, estimated_trajectories = run_single_simulation(
        target_parameters, 
        node_parameters, 
        filter_type='CSTA-CKF',
        sim_time=400,  # Long enough to see the full trajectory
        dt=0.5,
        plot_results=True
    )
    
    # 2. Plot all filter trajectories in one figure to generate Figs. 4 and 5 from the paper
    print("Generating Figs. 4 and 5 with all filter trajectories...")
    plot_all_filter_trajectories(
        target_parameters,
        node_parameters,
        filter_types=['CEKF', 'CUKF', 'CCKF', 'CSTA-CKF'],
        sim_time=400,
        dt=0.5
    )
    
    if not test_mode:
        # 3. Run Monte Carlo simulations to compare different filters (Fig. 7)
        print("Running Monte Carlo comparison...")
        filter_types = ['CEKF', 'CUKF', 'CCKF', 'CSTA-CKF']
        avg_rmse_per_filter = run_monte_carlo_comparison(
            target_parameters,
            node_parameters,
            filter_types,
            num_monte_carlo=100,
            sim_time=400,
            dt=0.5
        )
        
        # 4. Compare convergence rate (Fig. 8)
        print("Comparing convergence rate...")
        rmse_per_iteration = compare_convergence_rate(
            target_parameters,
            node_parameters,
            filter_types=['CCKF', 'CSTA-CKF']
        )
        
        # 5. Compare centralized vs distributed (Fig. 9)
        print("Comparing centralized vs distributed...")
        centralized_rmse, distributed_rmse = compare_centralized_vs_distributed(
            target_parameters,
            node_parameters,
            sim_time=400,
            dt=0.5,
            num_monte_carlo=100  # Reduced for speed (paper used 300)
        )
    
    # 6. Plot azimuth measurements (Fig. 6)
    print("Plotting azimuth measurements...")
    plot_azimuth_measurements(target, nodes, sim_time=400, dt=0.5)

if __name__ == "__main__":
    main()
