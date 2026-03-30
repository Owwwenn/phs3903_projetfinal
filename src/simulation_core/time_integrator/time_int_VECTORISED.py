import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from scipy.spatial.transform import Rotation
from simulation_core.potential_force.potentiel_force import build_potential_vector_force_torque_matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
#from simulation_core.potential_force.norms import build_inv_norm_matrix

# =============================================================================
# CONSTANTS
# =============================================================================
mmass = 18.015          # g/mol
kB    = 0.831446        # (g/mol)(Å/ps)²/K
T_init = 273   # K
q_o = -0.8476   
q_h = 0.4238
N  = 15
L  = 30
ex = np.array([1.0,0.0,0.0])
ey = np.array([0.0,1.0,0.0])
ez = np.array([0.0,0.0,1.0])


# SPC/E geometry in body frame
OH_BOND   =      1   # Å
HOH_ANGLE = 109.47      # degrees
O_body    = np.array([0.0, 0.0, 0.0])
angle_rad = np.radians(HOH_ANGLE / 2)
H1_body   = np.array([ np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND
H2_body   = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND

# Principal moments of inertia for SPC/E water (g/mol * Å²)
I_body = np.array([1.3743, 1.9144, 0.6001])
i1, i2, i3 = I_body


# =============================================================================
# SYSTEM CLASS
# =============================================================================
class MDSystem:
    def __init__(self, N):
        self.N       = N
        self.cm_pos  = np.zeros((N, 3))   # center of mass positions
        self.cm_vel  = np.zeros((N, 3))   # center of mass velocities
        self.force   = np.zeros((N, 3))   # forces on COM
        self.L       = np.zeros((N, 3))   # angular momentum in body frame
        self.T       = np.zeros((N, 3))   # torque in lab frame
        self.quat    = np.zeros((N, 4))   # quaternion (w, x, y, z)
        self.quat[:, 0] = 1.0             # identity quaternion for all
        self.r_last = np.zeros((N, 3))
        self.neighbor_list = None
        self.neighbor_count = None
        self.size = np.zeros(3)
        

# =============================================================================
# INITIALIZATION
# =============================================================================
def initialize_system(N, L):
    sys = MDSystem(N)

    # positions on cubic grid
    n_side = int(np.ceil(N**(1/3)))
    d = L / n_side
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                positions.append([i*d, j*d, k*d])
    sys.cm_pos = np.array(positions[:N])

    # random velocities from Maxwell-Boltzmann
    std_dev = np.sqrt(kB * T_init / mmass)
    sys.cm_vel = np.random.normal(0, std_dev, size=(N, 3))

    # remove center of mass drift
    sys.cm_vel -= sys.cm_vel.mean(axis=0)

    # Sample angular momentum in BODY FRAME
    Ix = i1
    Iy = i2
    Iz = i3

    # Sample angular momentum in BODY FRAME
    std_Lx = np.sqrt(Ix * kB * T_init)
    std_Ly = np.sqrt(Iy * kB * T_init)
    std_Lz = np.sqrt(Iz * kB * T_init)

    Lx = np.random.normal(0, std_Lx, size=N)
    Ly = np.random.normal(0, std_Ly, size=N)
    Lz = np.random.normal(0, std_Lz, size=N)

    sys.L[:, 0] = Lx
    sys.L[:, 1] = Ly
    sys.L[:, 2] = Lz

    # -----------------------
    # Remove total angular momentum (optional but recommended)
    # -----------------------
    L_total = sys.L.sum(axis=0)
    sys.L -= L_total / N


    return sys


# =============================================================================
# GEOMETRY
# =============================================================================
def get_atom_positions(sys):
    O_lab  = np.zeros((sys.N, 3))
    H1_lab = np.zeros((sys.N, 3))
    H2_lab = np.zeros((sys.N, 3))

    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    O_lab  = sys.cm_pos + R.apply(O_body)
    H1_lab = sys.cm_pos + R.apply(H1_body)
    H2_lab = sys.cm_pos + R.apply(H2_body)

    return O_lab, H1_lab, H2_lab

def mic(dr, L):
       return dr - L * np.round(dr / L)

def wrap_positions(pos, L):
    return pos - L * np.floor(pos / L)
    


# =============================================================================
# FORCE AND TORQUE
# =============================================================================
def compute_forces_and_torques(sys):
    v = np.concatenate([
        sys.cm_pos.flatten(),
        sys.cm_vel.flatten(),
        sys.quat.flatten(),
        sys.L.flatten()
    ])
    nbr_list = np.ones((sys.N, sys.N)) - np.eye(sys.N)
    U, F = build_potential_vector_force_torque_matrix(
        sys.N, v, L, L, L, nbr_list,
        np.radians(HOH_ANGLE), OH_BOND, q_o, q_h
    )
    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    sys.force = R.apply(F[:, :3])   # body → lab
    sys.T     = R.apply(F[:, 3:6])  # body → lab
 
# =============================================================================
# TRANSLATIONAL INTEGRATOR
# =============================================================================
def half_step_velocity(sys, dt):
    sys.cm_vel += 0.5 * (sys.force / mmass) * dt

def full_step_position(sys, dt):
    sys.cm_pos += sys.cm_vel * dt
    

def half_step_velocity_final(sys, dt):
    sys.cm_vel += 0.5 * (sys.force / mmass) * dt

# =============================================================================
# ROTATIONAL INTEGRATOR
# =============================================================================
def half_step_L(sys, dt):
    # --- 1. Lab → body frame torque (vectorized) ---
    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    T_body = R.inv().apply(sys.T)

    # --- 2. Half-step torque update ---
    sys.L += 0.5 * dt * T_body

    # --- 3. Ry rotation (around y-axis) ---
    Lx, Ly, Lz = sys.L.T

    alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
    c = np.cos(alpha)
    s = np.sin(alpha)

    Lx_new =  c * Lx + s * Lz
    Ly_new =  Ly
    Lz_new = -s * Lx + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    # --- 4. Rx rotation (around x-axis) ---
    Lx, Ly, Lz = sys.L.T

    beta = (dt / 2) * (1/i3 - 1/i1) * Lx
    c = np.cos(beta)
    s = np.sin(beta)

    Lx_new = Lx
    Ly_new =  c * Ly - s * Lz
    Lz_new =  s * Ly + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)


def quat_mul(q, r):
    # q, r : (N, 4) avec (w, x, y, z)
    w1, x1, y1, z1 = q.T
    w2, x2, y2, z2 = r.T

    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=1)


    
def axis_angle_to_quat(axis, angle):
    # axis: (3,), angle: (N,)
    half = 0.5 * angle
    s = np.sin(half)
    c = np.cos(half)

    return np.stack([
        c,
        axis[0]*s,
        axis[1]*s,
        axis[2]*s
    ], axis=1)

def get_quat(sys, dt):
    q = sys.quat  # (N, 4)
    omega = sys.L / I_body  # (N, 3)

  
    # rotations élémentaires (vectorisées)
    qy1 = axis_angle_to_quat(ey, omega[:,1] * dt/2)
    qx1 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qz  = axis_angle_to_quat(ez, omega[:,2] * dt)
    qx2 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qy2 = axis_angle_to_quat(ey, omega[:,1] * dt/2)

    # application (ordre IMPORTANT)
    q = quat_mul(qy1, q)
    q = quat_mul(qx1, q)
    q = quat_mul(qz,  q)
    q = quat_mul(qx2, q)
    q = quat_mul(qy2, q)

    # # normalisation (vectorisée)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    sys.quat = q
            

def half_step_L_final(sys, dt):
    # --- 1. Lab → body frame torque (vectorized) ---
    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    T_body = R.inv().apply(sys.T)

    # --- 2. Rx rotation FIRST (reversed order) ---
    Lx, Ly, Lz = sys.L.T

    beta = (dt / 2) * (1/i3 - 1/i1) * Lx
    c = np.cos(beta)
    s = np.sin(beta)

    Lx_new = Lx
    Ly_new =  c * Ly - s * Lz
    Lz_new =  s * Ly + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    # --- 3. Ry rotation ---
    Lx, Ly, Lz = sys.L.T

    alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
    c = np.cos(alpha)
    s = np.sin(alpha)

    Lx_new =  c * Lx + s * Lz
    Ly_new =  Ly
    Lz_new = -s * Lx + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    # --- 4. Second half-step torque ---
    sys.L += 0.5 * dt * T_body

# =============================================================================
# ENERGY
# =============================================================================
def kinetic_energy(sys):
    return 0.5 * mmass * np.sum(sys.cm_vel**2)

def rotational_energy(sys):
    return 0.5 * np.sum((sys.L**2) / I_body)

# =============================================================================
# NEIGHBOUR LIST
# =============================================================================
def build_nl(sys, r_c, r_s, L):
    rt = r_c + r_s
    cm_pos = sys.cm_pos
    N = len(cm_pos)
    Nmax = int(np.ceil(1.2 * 418.9 * rt**3))
    nl = np.full((N, Nmax), -1, dtype=int)  
    counts = np.zeros(N, dtype=int)

    for i in range(N):
        for j in range(i+1, N):
            dr = cm_pos[i] - cm_pos[j]
            dr = mic(dr, L)

            if np.dot(dr, dr) < rt**2:
                
                nl[i, counts[i]] = j
                counts[i] += 1

                
                nl[j, counts[j]] = i
                counts[j] += 1

    return nl, counts


            

def def_rebuild(sys, L, skin):
    disp = sys.cm_pos - sys.r_last
    disp = mic(disp, L)
    max_disp = np.max(np.linalg.norm(disp, axis = 1))


    return max_disp > (skin / 2)
    

# =============================================================================
# MAIN
# =============================================================================
dt = 0.000025
n_steps = 100000
s = np.zeros(100000)
en = np.zeros(100000)
sys = initialize_system(N, L)
sys.cm_pos = wrap_positions(sys.cm_pos, L)
sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)

pos_init   = sys.cm_pos.copy()
E_init     = kinetic_energy(sys) + rotational_energy(sys)
L_init     = sys.L.sum(axis=0).copy()

compute_forces_and_torques(sys)

for step in range(n_steps):
    half_step_velocity(sys, dt)
    half_step_L(sys, dt)
    get_quat(sys, dt)
    full_step_position(sys, dt)
    sys.cm_pos = wrap_positions(sys.cm_pos, L)
    compute_forces_and_torques(sys)
    half_step_velocity_final(sys, dt)
    half_step_L_final(sys, dt)


    # === DIAGNOSTICS ===
    E     = kinetic_energy(sys) + rotational_energy(sys)
    T     = 2 * kinetic_energy(sys) / (3 * N * kB)
    L_tot = sys.L.sum(axis=0)
    qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
    s[step] = step
    en[step] = E
    print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
          f"T={T:.1f}K | |L_drift|={np.linalg.norm(L_tot-L_init):.2e} | "
          f"qnorm_err={qnorm:.2e}")

# dt = 0.001
# ani = animate_simulation(n_steps=500, dt=0.001, interval=50)

print(max(abs(en - E_init*np.ones(len(en)))/E_init*100))
plt.plot(s, en)
plt.show()



