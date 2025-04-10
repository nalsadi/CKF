from system_model_write import object_motion_model, measurement_model, T, n, m,t
import numpy as np

def filterpy_motion_model(x, dt):
    return object_motion_model(x, dt, 0)  # Use existing motion model

def filterpy_measurement_model(x):
    return measurement_model(x, [0, 0, 0])  # Use existing measurement model

def filterpy_jacobian_motion(x, dt):
    F = np.zeros((n, n))
    epsilon = 1e-4
    for j in range(n):
        x_perturbed = x.copy()
        x_perturbed[j] += epsilon
        F[:, j] = (filterpy_motion_model(x_perturbed, dt) - filterpy_motion_model(x, dt)) / epsilon
    return F

def filterpy_jacobian_measurement(x):
    H = np.zeros((m, n))
    epsilon = 1e-4
    z_pred = filterpy_measurement_model(x)
    for j in range(n):
        x_perturbed = x.copy()
        x_perturbed[j] += epsilon
        H[:, j] = (filterpy_measurement_model(x_perturbed) - z_pred) / epsilon
    return H
