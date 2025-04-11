"""
Real‑time CKF with on‑line LSTM tuning of Q and R.
-------------------------------------------------
• First CKF = baseline (fixed Q, R)
• Second CKF = Q,R scale factors learned on‑line
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
# ‑‑‑ 1. PARAMETERS (exactly your originals) ‑‑‑ #
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


dt, T, tfinal = 1.0, 1.0, 500
t = np.arange(0, tfinal + dt, dt)
n, m = 4, 2

L1, L2 = 0.16, 0.01
Q_base = L1 * np.array([[dt**3/3, dt**2/2, 0, 0],
                        [dt**2/2, dt,       0, 0],
                        [0,       0,  dt**3/3, dt**2/2],
                        [0,       0,  dt**2/2, dt]]) + L2*np.diag([0,0,0,1])
R_base = 2500 * np.eye(2)          # 50 m σ per axis

# ‑‑‑ 2. MOTION MODELS ‑‑‑ #
UM = np.array([[1, T, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, T],
               [0, 0, 0, 1]])

def CT(w):
    if abs(w) < 1e-6:
        return UM
    s, c = np.sin(w*T), np.cos(w*T)
    return np.array([[1,  s/w, 0, -(1-c)/w],
                     [0,   c , 0,  -s   ],
                     [0, (1-c)/w, 1,  s/w],
                     [0,   s , 0,   c   ]])

def motion_model(x, k):
    if t[k] <= 125:
        A = UM
    elif 125 < t[k] <= 215:
        A = CT(np.deg2rad(1))
    elif 215 < t[k] <= 340:
        A = UM
    elif 340 < t[k] <= 370:
        A = CT(np.deg2rad(-3))
    else:
        A = UM
    return A @ x

# ‑‑‑ 3. MEASUREMENT MODEL ‑‑‑ #
H = np.array([[1,0,0,0],
              [0,0,1,0]])
def measurement_model(x):
    return H @ x

# ‑‑‑ 4. NUMERICAL HELPERS ‑‑‑ #
def sym(P):      return 0.5*(P+P.T)
def chol(P):
    P = sym(P)
    try:                return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        d,V = np.linalg.eigh(P)
        return np.linalg.cholesky(V @ np.diag(np.clip(d,1e-9,None)) @ V.T)

# Cubature points
def cubature_points(dim):
    Xi = np.sqrt(dim)*np.vstack((np.eye(dim), -np.eye(dim)))
    W  = np.full(2*dim, 1/(2*dim))
    return Xi, W

# ‑‑‑ 5. TRUE TRAJECTORY & MEASUREMENTS ‑‑‑ #
x0_true = np.array([25e3, -120, 10e3, 0.0])
x_true  = np.zeros((n, len(t)));  x_true[:,0] = x0_true
z_meas  = np.zeros((m, len(t)))

for k in range(1,len(t)):
    w = np.random.multivariate_normal(np.zeros(n), Q_base)
    v = np.random.multivariate_normal(np.zeros(m), R_base)
    x_true[:,k] = motion_model(x_true[:,k-1], k) + w
    z_meas[:,k] = measurement_model(x_true[:,k]) + v

# ‑‑‑ 6. FILTER INITIALISATION ‑‑‑ #
x0_est = np.array([24800, -100, 10200, -20])
P0     = np.diag([200**2, 20**2, 200**2, 20**2])

x_ckf      = np.zeros_like(x_true);     x_ckf[:,0]      = x0_est
x_ckf_lstm = np.zeros_like(x_true);     x_ckf_lstm[:,0] = x0_est
P_ckf, P_ckf_lstm = P0.copy(), P0.copy()

Q_est, R_est = Q_base.copy(), R_base.copy()

# ‑‑‑ 7. LSTM MODEL FOR Q/R SCALE FACTORS ‑‑‑ #
WINDOW = 20                # number of past innovations
LR     = 1e-4
class QRNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm  = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(2, activation='softplus') # >0
    def call(self,x):        # x.shape = (None, WINDOW, m)
        return self.dense(self.lstm(x)) + 1e-6

qr_net   = QRNet()
optimizer = tf.keras.optimizers.Adam(LR)

# Buffer to store the last WINDOW innovations
innov_hist = []

# Gaussian NLL loss
def nll_loss(innov, r_scale):
    """innov: (batch,m), r_scale: (batch,)"""
    R_pred = r_scale[:,None,None]*R_base[None,:,:]  # shape (batch,2,2)
    # log|R| + νᵀ R⁻¹ ν
    inv   = tf.linalg.inv(R_pred)
    quad  = tf.reduce_sum(tf.expand_dims(innov,1)*tf.matmul(inv,
                    tf.expand_dims(innov,2)), axis=[1,2])
    logdet= tf.math.log(tf.linalg.det(R_pred))
    return 0.5*tf.reduce_mean(logdet + quad)

# ‑‑‑ 8. MAIN LOOP ‑‑‑ #
Xi, W = cubature_points(n)

for k in range(1, len(t)):
    z = z_meas[:,k]

    # ---------- 8.1  BASE CKF (fixed Q,R) ----------
    S = chol(P_ckf) @ Xi.T
    X = x_ckf[:,k-1][:,None] + S
    Xpred = np.array([motion_model(X[:,i],k) for i in range(2*n)]).T
    xpred = Xpred @ W
    Ppred = Q_base + (Xpred-xpred[:,None]) @ np.diag(W) @ (Xpred-xpred[:,None]).T
    Z     = (H @ Xpred)
    zpred = Z @ W
    Smat  = R_base + (Z-zpred[:,None]) @ np.diag(W) @ (Z-zpred[:,None]).T
    Pxz   = (Xpred-xpred[:,None]) @ np.diag(W) @ (Z-zpred[:,None]).T
    K     = Pxz @ np.linalg.inv(Smat)
    x_ckf[:,k] = xpred + K @ (z - zpred)
    P_ckf      = sym(Ppred - K @ Smat @ K.T)

    # ---------- 8.2  CKF + LSTM‑tuned Q,R ----------
    S = chol(P_ckf_lstm) @ Xi.T
    X = x_ckf_lstm[:,k-1][:,None] + S
    Xpred = np.array([motion_model(X[:,i],k) for i in range(2*n)]).T
    xpred = Xpred @ W

    # Use current Q_est in time update
    Ppred = Q_est + (Xpred-xpred[:,None]) @ np.diag(W) @ (Xpred-xpred[:,None]).T
    Z     = (H @ Xpred)
    zpred = Z @ W

    # Innovation
    innov = z - zpred
    innov_hist.append(innov)
    if len(innov_hist) > WINDOW:
        innov_hist.pop(0)

    # Scale factors prediction & online training
    if len(innov_hist) == WINDOW:
        with tf.GradientTape() as tape:
            inp   = tf.convert_to_tensor(np.array(innov_hist)[None,:,:], dtype=tf.float32)
            q_r   = qr_net(inp)[0]             # [q_scale, r_scale]
            q_s, r_s = q_r[0], q_r[1]
            loss = nll_loss(inp[:,-1,:], r_s[None])
            print("Loss:", loss.numpy())
        grads = tape.gradient(loss, qr_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, qr_net.trainable_variables))

        # Update Q,R estimates (diagonal scaling)
        Q_est = q_s.numpy()*Q_base
        R_est = r_s.numpy()*R_base

    # Measurement update with latest R_est
    Smat  = R_est + (Z-zpred[:,None]) @ np.diag(W) @ (Z-zpred[:,None]).T
    Pxz   = (Xpred-xpred[:,None]) @ np.diag(W) @ (Z-zpred[:,None]).T
    K     = Pxz @ np.linalg.inv(Smat)
    x_ckf_lstm[:,k] = xpred + K @ innov
    P_ckf_lstm      = sym(Ppred - K @ Smat @ K.T)

# ‑‑‑ 9. EVALUATION ‑‑‑ #
def rmse(est, true):  return np.sqrt(np.mean((est-true)**2, axis=1))

print("Position RMSE [x, y] (m)")
print("  CKF fixed  :", rmse(x_ckf,      x_true)[[0,2]])
print("  CKF + LSTM :", rmse(x_ckf_lstm, x_true)[[0,2]])

plt.figure(figsize=(12,5))
plt.plot(t, x_true[0],         'k',  label='True x')
plt.plot(t, x_ckf[0],          'b-.',label='CKF fixed')
plt.plot(t, x_ckf_lstm[0],     'r--',label='CKF + LSTM')
plt.xlabel('Time (s)'); plt.ylabel('x position (m)')
plt.title('Tracking performance (x)')
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()


# =============================================================================
# 10. Plot 2D Trajectories (x vs y)
# =============================================================================
plt.figure(figsize=(10,8))
plt.plot(x_true[0], x_true[2], 'k', label="True Trajectory", linewidth=2)
plt.plot(x_ckf[0], x_ckf[2],          'b-.',label='CKF fixed')
plt.plot(x_ckf_lstm[0], x_ckf_lstm[2],     'r--',label='CKF + LSTM')
plt.xlabel('x position [m]')
plt.ylabel('y position [m]')
plt.title('2D Trajectory: x vs y (Air Traffic Control Scenario)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio to properly show motion
plt.tight_layout()
plt.show()

# =============================================================================
# 11. Plot RMSE Over Time
# =============================================================================
rmse_ckf = np.sqrt(np.mean((x_ckf - x_true)**2, axis=0))
rmse_ckf_lstm = np.sqrt(np.mean((x_ckf_lstm - x_true)**2, axis=0))

plt.figure(figsize=(12,5))
plt.plot(t, rmse_ckf, 'b-.', label='CKF fixed')
plt.plot(t, rmse_ckf_lstm, 'r--', label='CKF + LSTM')
plt.xlabel('Time (s)')
plt.ylabel('RMSE (m)')
plt.title('RMSE Over Time')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
