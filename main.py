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
            'turn2': np.array([10, -25])    # Final maneuver
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
        self.beta = 1.1  # Weakening factor
        self.V_prev = None  # Previous measurement residual covariance
        
    def get_measurement(self, target: Target, node: Node) -> Tuple[np.ndarray, np.ndarray]:
        dx = target.state[0] - node.position[0]
        dy = target.state[2] - node.position[1]
        dh = 0.3  # Height difference between target and sensor
        
        # Range
        r = sqrt(dx**2 + dy**2)
        # Azimuth
        psi = atan2(dy, dx)
        # Elevation
        theta = atan2(dh, sqrt(dx**2 + dy**2 + dh**2))
        
        return np.array([r, psi, theta])

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
        # Get cubature points
        cubature_points = self.generate_cubature_points(x, P)
        
        # Propagate points through state model (constant velocity model)
        propagated_points = []
        for point in cubature_points:
            propagated_points.append(point)  # Apply state transition
            
        propagated_points = np.array(propagated_points)
        
        # Calculate predicted state and covariance
        x_pred = np.mean(propagated_points, axis=0)
        P_pred = np.zeros_like(P)
        
        for point in propagated_points:
            P_pred += np.outer(point - x_pred, point - x_pred)
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
        
        # Calculate suboptimal fading factor
        gamma = z - z_pred
        if self.V_prev is None:
            V = np.outer(gamma, gamma)
        else:
            V = (self.rho * self.V_prev + np.outer(gamma, gamma)) / (1 + self.rho)
            
        H = Pxz.T @ np.linalg.inv(P)  # Changed from Pzz to match dimensions
        N = V - H @ self.Q @ H.T - self.beta * self.R
        M = Pzz - V + N + (self.beta - 1) * self.R
        
        lambda_k = max(1, np.trace(N) / np.trace(M))
        
        # Calculate adaptive factor
        mu_k = (np.trace(np.outer(gamma, gamma)) - np.trace(Pzz)) / np.trace(self.R)
        mu_k = max(1, mu_k)
        
        # Update state estimate
        Pzz += mu_k * self.R  # Add adaptive measurement noise
        K = Pxz @ np.linalg.inv(Pzz)
        
        x_updated = x + K @ (z - z_pred)
        P_updated = lambda_k * (P - K @ Pzz @ K.T)
        
        self.V_prev = V
        
        return x_updated, P_updated

    def consensus_step(self, states: List[np.ndarray], covs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        new_states = []
        new_covs = []
        
        for i in range(self.n_nodes):
            state_i = np.zeros_like(states[0])
            cov_i = np.zeros_like(covs[0])
            
            for j in range(self.n_nodes):
                state_i += self.weights[i,j] * states[j]
                cov_i += self.weights[i,j] * covs[j]
                
            new_states.append(state_i)
            new_covs.append(cov_i)
            
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
    
    Q = 1e-5 * np.eye(4)  # Process noise covariance
    R = np.eye(3)  # Measurement noise covariance
    
    filter = CSTACKF(nodes, consensus_weights, Q, R)
    
    # Initialize state estimates and covariances for each node
    x_estimates = [np.array([2000, 0, 10000, 0]) for _ in range(len(nodes))]
    P_estimates = [100 * np.eye(4) for _ in range(len(nodes))]
    
    # Storage for trajectories and errors
    target_trajectory = []
    node_trajectories = [[] for _ in range(4)]
    estimated_trajectories = [[] for _ in range(4)]
    rmse_history = []
    
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
            
        # Calculate RMSE for this timestep
        rmse = np.sqrt(np.mean([(x_estimates[i][0] - target.state[0])**2 + 
                               (x_estimates[i][2] - target.state[2])**2 
                               for i in range(len(nodes))]))
        rmse_history.append(rmse)
    
    # Convert trajectories to numpy arrays
    target_trajectory = np.array(target_trajectory)
    node_trajectories = [np.array(traj) for traj in node_trajectories]
    estimated_trajectories = [np.array(traj) for traj in estimated_trajectories]
    rmse_history = np.array(rmse_history)
    
    # Plot 2D trajectories with estimates
    plt.figure(figsize=(12, 8))
    plt.plot(target_trajectory[:,0], target_trajectory[:,1], 'r-', label='Target')
    
    colors = ['b', 'g', 'c', 'm']
    for i, traj in enumerate(node_trajectories):
        plt.plot(traj[:,0], traj[:,1], f'{colors[i]}.-', label=f'Sensor {i+1}')
        plt.plot(estimated_trajectories[i][:,0], estimated_trajectories[i][:,1], 
                f'{colors[i]}--', label=f'Estimate {i+1}')
    
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('2D Trajectories with Estimates')
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
    
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Height (km)')
    ax.set_title('3D Trajectories')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()
