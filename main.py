import numpy as np
from math import atan2, sqrt
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, position: np.ndarray, velocity: np.ndarray = None, is_mobile: bool = False):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity if velocity is not None else np.zeros(2), dtype=np.float64)
        self.is_mobile = is_mobile
        
    def update_position(self, dt: float):
        if self.is_mobile:
            self.position += self.velocity * dt

class Target:
    def __init__(self, state: np.ndarray):
        # state = [x, vx, y, vy]
        self.state = np.array(state, dtype=np.float64)
        self.time = 0
        # Phase velocities
        self.velocities = {
            'descent': np.array([0, -60]),  # Straight down at 60 units/s
            'turn1': np.array([20, -15]),   # Right turn with slight descent
            'turn2': np.array([0, -25])    # Final maneuver
        }
        self.phase = 'descent'
        
    def update(self, dt: float):
        self.time += dt
        
        # Update phase based on position/time
        if self.state[2] <= 4000 and self.phase == 'descent':
            self.phase = 'turn1'
        elif self.state[0] >= 4000 and self.phase == 'turn1':
            self.phase = 'turn2'
            
        # Apply velocity based on current phase
        vel = self.velocities[self.phase]
        self.state[0] += vel[0] * dt  # x position
        self.state[2] += vel[1] * dt  # y position
        self.state[1] = vel[0]        # x velocity
        self.state[3] = vel[1]        # y velocity

class CSTACKF:
    def __init__(self, nodes: List[Node], consensus_weights: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.nodes = nodes
        self.weights = consensus_weights
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.n_nodes = len(nodes)
        self.state_dim = 4
        self.rho = 0.95  # Forgetting factor
        self.beta = 2.0  # Increased for better maneuver tracking
        self.consensus_iterations = 3
        self.measurement_noise = 0.01  # Added measurement noise
        self.V_prev = None  # Previous measurement residual covariance
        
    def get_measurement(self, target: Target, node: Node) -> np.ndarray:
        dx = target.state[0] - node.position[0]
        dy = target.state[2] - node.position[1]
        dh = 0.3  # Height difference between target and sensor
        
        # Range
        r = sqrt(dx**2 + dy**2)
        # Azimuth
        psi = atan2(dy, dx)
        # Elevation
        theta = atan2(dh, sqrt(dx**2 + dy**2 + dh**2))
        
        z = np.array([r, psi, theta])
        noise = np.random.normal(0, self.measurement_noise, size=3)
        noise[0] *= 100  # Range noise larger than angle noise
        return z + noise

    def generate_cubature_points(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        n = x.shape[0]
        sqrt_P = np.linalg.cholesky(P)
        
        # Generate cubature points
        xi_points = []
        unit_points = np.sqrt(n) * np.vstack((np.eye(n), -np.eye(n)))
        
        for i in range(2*n):
            xi_points.append(sqrt_P @ unit_points[i] + x)
            
        return np.array(xi_points)

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dt = 0.5
        # Proper state transition matrix for CV model with explicit float64 type
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Get cubature points
        cubature_points = self.generate_cubature_points(x, P)
        
        # Propagate through state model
        propagated_points = []
        for point in cubature_points:
            prop_point = F @ point
            propagated_points.append(prop_point)
            
        propagated_points = np.array(propagated_points, dtype=np.float64)
        
        # Calculate predicted state and covariance
        x_pred = np.mean(propagated_points, axis=0)
        P_pred = np.zeros_like(P, dtype=np.float64)  # Explicit float64 type
        
        for point in propagated_points:
            diff = point - x_pred
            P_pred += np.outer(diff, diff)
            
        P_pred = P_pred / len(propagated_points) + self.Q
        return x_pred, P_pred

    def measurement_function(self, state: np.ndarray, node_position: np.ndarray = np.array([0, 0])) -> np.ndarray:
        """Nonlinear measurement function h(x) with respect to sensor position"""
        x, vx, y, vy = state
        dx = x - node_position[0]
        dy = y - node_position[1]
        dh = 0.3  # Height difference between target and sensor
        
        # Range
        r = np.sqrt(dx**2 + dy**2)
        # Azimuth
        psi = np.arctan2(dy, dx)
        # Elevation
        theta = np.arctan2(dh, np.sqrt(dx**2 + dy**2 + dh**2))
        
        return np.array([r, psi, theta])

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray, node_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get cubature points
        cubature_points = self.generate_cubature_points(x, P)
        
        # Propagate points through measurement model
        z_points = []
        for point in cubature_points:
            z_points.append(self.measurement_function(point, self.nodes[node_idx].position))
        z_points = np.array(z_points)
        
        # Calculate predicted measurement
        z_pred = np.mean(z_points, axis=0)
        
        # Calculate innovation covariance
        Pzz = np.zeros((3, 3))  # 3x3 for [r, psi, theta]
        Pxz = np.zeros((4, 3))  # 4x3 for state vs measurement
        
        for i, (z_point, x_point) in enumerate(zip(z_points, cubature_points)):
            diff_z = z_point - z_pred
            diff_x = x_point - x
            Pzz += np.outer(diff_z, diff_z)
            Pxz += np.outer(diff_x, diff_z)
            
        Pzz = Pzz / len(z_points)
        Pxz = Pxz / len(z_points)
        
        # Improved innovation handling
        innovation = z - z_pred
        # Wrap angle innovations to [-pi, pi]
        innovation[1:] = np.arctan2(np.sin(innovation[1:]), np.cos(innovation[1:]))
        
        innovation_cov = np.outer(innovation, innovation)
        
        if self.V_prev is None:
            V = innovation_cov
        else:
            V = (self.rho * self.V_prev + innovation_cov) / (1 + self.rho)
        
        # More stable matrix calculations
        H = Pxz.T @ np.linalg.pinv(P + 1e-6 * np.eye(P.shape[0]))
        N = V - H @ self.Q @ H.T - self.beta * self.R
        M = Pzz - V + N + (self.beta - 1) * self.R
        
        # Improved fading factor
        lambda_k = max(1, np.clip(np.trace(N) / (np.trace(M) + 1e-6), 1, 3))
        
        # Better adaptive factor
        S = innovation_cov - Pzz
        mu_k = max(1, np.clip(np.trace(S) / (np.trace(self.R) + 1e-6), 1, 5))
        
        # Stabilized update
        R_adapted = mu_k * self.R
        Pzz_stable = Pzz + R_adapted + 1e-6 * np.eye(3)
        K = Pxz @ np.linalg.pinv(Pzz_stable)
        
        x_updated = x + K @ innovation
        P_updated = lambda_k * (P - K @ Pzz_stable @ K.T)
        
        # Ensure symmetric and positive definite
        P_updated = (P_updated + P_updated.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(P_updated)))
        if min_eig < 1e-6:
            P_updated += (1e-6 - min_eig) * np.eye(P_updated.shape[0])
            
        self.V_prev = V
        return x_updated, P_updated

    def consensus_step(self, states: List[np.ndarray], covs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        new_states = states.copy()
        new_covs = covs.copy()
        
        for _ in range(self.consensus_iterations):
            updated_states = []
            updated_covs = []
            
            for i in range(self.n_nodes):
                state_i = np.zeros_like(states[0])
                cov_i = np.zeros_like(covs[0])
                
                for j in range(self.n_nodes):
                    state_i += self.weights[i,j] * new_states[j]
                    cov_i += self.weights[i,j] * new_covs[j]
                    
                updated_states.append(state_i)
                updated_covs.append(cov_i)
            
            new_states = updated_states
            new_covs = updated_covs
            
        return new_states, new_covs

def run_simulation():
    # Initialize simulation parameters
    dt = 0.5  # Time step
    sim_time = 300  # Reduced simulation time to match maneuver pattern
    n_steps = int(sim_time/dt)
    
    # Initialize nodes
    nodes = [
        Node(np.array([2500, 1000])),  # Static node 1
        Node(np.array([3500, 1000])),  # Static node 2
        Node(np.array([4500, 1000])),  # Static node 3
        Node(np.array([1500, 0]), np.array([10, 20]), True)  # Mobile node
    ]
    
    # Initialize target at starting position with zero velocity
    target = Target(np.array([2000, 0, 10000, 0]))
    
    # Initialize CSTACKF
    consensus_weights = np.array([
        [1/2, 1/3, 1/4, 0],
        [1/3, 5/12, 1/4, 0],
        [1/4, 1/4, 1/4, 1/4],
        [0, 0, 1/4, 3/4]
    ])
    
    # Initialize with explicit float64 type
    Q = 1e-3 * np.diag([0.1, 1, 0.1, 1]).astype(np.float64)
    R = np.diag([100, 0.01, 0.01]).astype(np.float64)
    
    filter = CSTACKF(nodes, consensus_weights, Q, R)
    
    # Initialize state estimates and covariances for each node
    x_estimates = [np.array([2000, 0, 10000, 0], dtype=np.float64) for _ in range(len(nodes))]
    P_estimates = [np.diag([1000, 10, 1000, 10]).astype(np.float64) for _ in range(len(nodes))]
    
    # Storage for trajectories and errors
    target_trajectory = []
    node_trajectories = [[] for _ in range(4)]
    estimated_trajectories = [[] for _ in range(4)]
    rmse_history = []
    consensus_trajectory = []
    
    # Main simulation loop
    for k in range(n_steps):
        # Update target and node positions
        target.update(dt)
        target_trajectory.append(np.array([target.state[0], target.state[2]]))
        
        for i, node in enumerate(nodes):
            node.update_position(dt)
            node_trajectories[i].append(node.position.copy())
            
            # Get measurement
            z = filter.get_measurement(target, node)
            
            # Local filter update
            x_pred, P_pred = filter.predict(x_estimates[i], P_estimates[i])
            x_estimates[i], P_estimates[i] = filter.update(x_pred, P_pred, z, i)  # Added node index
        
        # Consensus step
        x_estimates, P_estimates = filter.consensus_step(x_estimates, P_estimates)
        
        # Store estimates
        for i in range(len(nodes)):
            estimated_trajectories[i].append(np.array([x_estimates[i][0], x_estimates[i][2]]))
        
        # Calculate and store consensus estimate
        consensus_state = np.mean([x[0:4:2] for x in x_estimates], axis=0)
        consensus_trajectory.append(consensus_state)
        
        # Calculate RMSE for this timestep
        rmse = np.sqrt(np.mean([(x_estimates[i][0] - target.state[0])**2 + 
                               (x_estimates[i][2] - target.state[2])**2 
                               for i in range(len(nodes))]))
        rmse_history.append(rmse)
    
    # Convert trajectories to numpy arrays
    target_trajectory = np.array(target_trajectory)
    node_trajectories = [np.array(traj) for traj in node_trajectories]
    estimated_trajectories = [np.array(traj) for traj in estimated_trajectories]
    consensus_trajectory = np.array(consensus_trajectory)
    rmse_history = np.array(rmse_history)
    
    # Plot 2D trajectories with estimates and consensus
    plt.figure(figsize=(12, 8))
    plt.plot(target_trajectory[:,0], target_trajectory[:,1], 'r-', label='Target')
    plt.plot(consensus_trajectory[:,0], consensus_trajectory[:,1], 'k--', linewidth=2, label='Consensus Estimate')
    
    colors = ['b', 'g', 'c', 'm']
    for i, traj in enumerate(node_trajectories):
        plt.plot(traj[:,0], traj[:,1], f'{colors[i]}.-', label=f'Sensor {i+1}')
        plt.plot(estimated_trajectories[i][:,0], estimated_trajectories[i][:,1], 
                f'{colors[i]}:', alpha=0.5, label=f'Local Estimate {i+1}')
    
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('2D Trajectories with Local and Consensus Estimates')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(n_steps) * dt, rmse_history)
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (km)')
    plt.title('Root Mean Square Error')
    plt.grid(True)
    plt.show()

    # Plot 3D trajectories
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Target trajectory
    ax.plot(target_trajectory[:,0], target_trajectory[:,1], 
            np.ones_like(target_trajectory[:,0])*0.5, 'r-', label='Target')
    
    # Sensor trajectories
    for i, traj in enumerate(node_trajectories):
        ax.plot(traj[:,0], traj[:,1], 
                np.ones_like(traj[:,0])*0.2, f'{colors[i]}.-', label=f'Sensor {i+1}')
    
    # Consensus trajectory
    ax.plot(consensus_trajectory[:,0], consensus_trajectory[:,1], 
            np.ones_like(consensus_trajectory[:,0])*0.5, 'k--', linewidth=2, label='Consensus Estimate')
    
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Height (km)')
    ax.set_title('3D Trajectories with Consensus Estimate')
    plt.legend()
    plt.show()

def run_monte_carlo_simulation(n_runs: int = 100):
    # Storage for Monte Carlo results
    rmse_histories = []
    
    for run in range(n_runs):
        # Initialize simulation parameters
        dt = 0.5  # Time step (0.5s as per paper)
        sim_time = 500  # Total simulation time
        n_steps = int(sim_time/dt)
        
        # Initialize nodes with exact positions from paper
        nodes = [
            Node(np.array([2500, 1000])),  # Static node 1
            Node(np.array([3500, 1000])),  # Static node 2
            Node(np.array([4500, 1000])),  # Static node 3
            Node(np.array([1500, 0]), np.array([10, 20]), True)  # Mobile node with 10km/s X, 20km/s Y
        ]
        
        # Initialize target with exact initial state
        target = Target(np.array([2000, 0, 10000, 0]))
        
        # Initialize CSTACKF with parameters from paper
        consensus_weights = np.array([
            [1/2, 1/3, 1/4, 0],
            [1/3, 5/12, 1/4, 0],
            [1/4, 1/4, 1/4, 1/4],
            [0, 0, 1/4, 3/4]
        ])
        
        Q = 1e-3 * np.diag([0.1, 1, 0.1, 1])  # More process noise on velocities
        R = np.diag([100, 0.01, 0.01])  # Realistic measurement noise
        
        filter = CSTACKF(nodes, consensus_weights, Q, R)
        
        # Initialize state estimates and covariances
        x_estimates = [np.array([2000, 0, 10000, 0], dtype=np.float64) for _ in range(len(nodes))]
        P_estimates = [np.diag([1000, 10, 1000, 10]).astype(np.float64) for _ in range(len(nodes))]
        
        # Storage for this run
        rmse_history = []
        
        # Simulation loop
        for k in range(n_steps):
            # Update target and node positions
            target.update(dt)
            
            for i, node in enumerate(nodes):
                node.update_position(dt)
                
                # Get measurement
                z = filter.get_measurement(target, node)
                
                # Local filter update
                x_pred, P_pred = filter.predict(x_estimates[i], P_estimates[i])
                x_estimates[i], P_estimates[i] = filter.update(x_pred, P_pred, z, i)
            
            # Consensus step
            x_estimates, P_estimates = filter.consensus_step(x_estimates, P_estimates)
            
            # Calculate RMSE for this timestep
            rmse = np.sqrt(np.mean([(x_estimates[i][0] - target.state[0])**2 + 
                                   (x_estimates[i][2] - target.state[2])**2 
                                   for i in range(len(nodes))]))
            rmse_history.append(rmse)
        
        rmse_histories.append(rmse_history)
    
    # Calculate average RMSE across all runs
    avg_rmse = np.mean(rmse_histories, axis=0)
    std_rmse = np.std(rmse_histories, axis=0)
    
    # Plot averaged results
    plt.figure(figsize=(10, 6))
    time = np.arange(n_steps) * dt
    plt.plot(time, avg_rmse, 'b-', label='CSTA-CKF')
    plt.fill_between(time, avg_rmse - std_rmse, avg_rmse + std_rmse, alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (km)')
    plt.title('Average RMSE over {} Monte Carlo Runs'.format(n_runs))
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Also plot azimuth angles as in paper Fig. 6
    plt.figure(figsize=(10, 6))
    for i in range(4):
        azimuth_history = []
        node = nodes[i]
        for k in range(n_steps):
            dx = target.state[0] - node.position[0]
            dy = target.state[2] - node.position[1]
            azimuth = np.degrees(np.arctan2(dy, dx))
            azimuth_history.append(azimuth)
        plt.plot(time, azimuth_history, label=f'Sensor {i+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Azimuth Angle (degrees)')
    plt.title('Azimuth Angles Measured by Sensors')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run both single simulation and Monte Carlo analysis
    run_simulation()
    run_monte_carlo_simulation(100)  # 100 runs as per paper
