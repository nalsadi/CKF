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

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Implement cubature Kalman filter prediction step
        # ...existing code...
        return x_pred, P_pred

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Implement strong tracking adaptive cubature Kalman filter update step
        # ...existing code...
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
    
    # Storage for trajectories
    target_trajectory = []
    node_trajectories = [[] for _ in range(4)]
    
    # Main simulation loop
    for k in range(n_steps):
        # Update target and node positions
        target.update(dt)
        target_trajectory.append(np.array([target.state[0], target.state[2]]))
        
        for i, node in enumerate(nodes):
            node.update_position(dt)
            node_trajectories[i].append(node.position.copy())
    
    # Convert trajectories to numpy arrays
    target_trajectory = np.array(target_trajectory)
    node_trajectories = [np.array(traj) for traj in node_trajectories]
    
    # Plot 2D trajectories
    plt.figure(figsize=(12, 8))
    plt.plot(target_trajectory[:,0], target_trajectory[:,1], 'r-', label='Target')
    
    colors = ['b', 'g', 'c', 'm']
    for i, traj in enumerate(node_trajectories):
        plt.plot(traj[:,0], traj[:,1], f'{colors[i]}.-', label=f'Sensor {i+1}')
    
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('2D Trajectories')
    plt.legend()
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
