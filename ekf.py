import numpy as np

class EKF:
    def __init__(self, Q, R, dt):
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.dt = dt
        self.n = 4  # State dimension
        self.P = np.eye(4) * 100  # Initial state covariance
        self.x = np.array([2000, 0, 10000, -40], dtype=np.float32)  # Initial state estimate
    
    def F_matrix(self):
        return np.array([
            [1, self.dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.dt],
            [0, 0, 0, 1]
        ])
    
    def H_jacobian(self, x, sensor_pos):
        dx = x[0] - sensor_pos[0]
        dy = x[2] - sensor_pos[1]
        dz = 0.5 - sensor_pos[2]  # target_height - sensor_height
        
        d_squared = dx**2 + dy**2
        d = np.sqrt(d_squared)
        r = np.sqrt(d_squared + dz**2)
        
        # Partial derivatives for range
        dr_dx = dx/r
        dr_dvx = 0
        dr_dy = dy/r
        dr_dvy = 0
        
        # Partial derivatives for azimuth
        dpsi_dx = -dy/(d_squared)
        dpsi_dvx = 0
        dpsi_dy = dx/(d_squared)
        dpsi_dvy = 0
        
        # Partial derivatives for elevation
        dtheta_dx = -dx*dz/(r**2*d)
        dtheta_dvx = 0
        dtheta_dy = -dy*dz/(r**2*d)
        dtheta_dvy = 0
        
        H = np.array([
            [dr_dx, dr_dvx, dr_dy, dr_dvy],
            [dpsi_dx, dpsi_dvx, dpsi_dy, dpsi_dvy],
            [dtheta_dx, dtheta_dvx, dtheta_dy, dtheta_dvy]
        ])
        
        return H
    
    def predict(self):
        F = self.F_matrix()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x
    
    def update(self, z, sensor_pos):
        predicted_measurement = self.measurement_model(self.x, sensor_pos)
        H = self.H_jacobian(self.x, sensor_pos)
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        innovation = z - predicted_measurement
        # Normalize angular measurements to [-π, π]
        innovation[1:] = (innovation[1:] + np.pi) % (2 * np.pi) - np.pi
        
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.n) - K @ H) @ self.P
        
        return self.x
    
    def measurement_model(self, x, sensor_pos):
        dx = x[0] - sensor_pos[0]
        dy = x[2] - sensor_pos[1]
        h = 0.5 - sensor_pos[2]  # target_height - sensor_height
        
        d_squared = dx**2 + dy**2
        r = np.sqrt(d_squared + h**2)
        psi = np.arctan2(dy, dx)
        theta = np.arctan2(h, np.sqrt(d_squared))
        
        return np.array([r, psi, theta])
    
    def consensus_update(self, neighbors_info, consensus_weights_row):
        """
        Perform consensus update using information from neighbors.
        neighbors_info: List of tuples [(x_neighbor, P_neighbor), ...]
        consensus_weights_row: Row of the consensus weight matrix for the current node.
        """
        x_consensus = consensus_weights_row[0] * self.x
        P_consensus = consensus_weights_row[0] * self.P

        for i, (x_neighbor, P_neighbor) in enumerate(neighbors_info):
            x_consensus += consensus_weights_row[i + 1] * x_neighbor
            P_consensus += consensus_weights_row[i + 1] * P_neighbor

        self.x = x_consensus
        self.P = P_consensus
