import numpy as np
from scipy.spatial.transform import Rotation
from md_sim.core.system import SPCE
from md_sim.core.potential_force.coul_LJ_opt import compute_forces_and_torques
from md_sim.core.system import kB

# ─────────────────────────────────────────────
#  Quaternion helpers
# ─────────────────────────────────────────────
def quat_mul(q, r):
    """Hamilton product, both (N,4) [w,x,y,z]."""
    w1,x1,y1,z1 = q.T
    w2,x2,y2,z2 = r.T
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)

def axis_angle_to_quat(axis, angles):
    """axis (3,), angles (N,) → (N,4) [w,x,y,z]."""
    h = 0.5 * angles
    s = np.sin(h)
    c = np.cos(h)
    return np.stack([c, axis[0]*s, axis[1]*s, axis[2]*s], axis=1)

def nh_update(xi_t, xi_r, s_t, s_r, cm_vel, L_body, n_mol, mass, T, dt):
    I1, I2, I3 = SPCE['I_body']
    q_t = (3*n_mol - 3) * kB * T / (50**2)
    q_r = 3*n_mol * kB * T / (50**2)
    
    KE_t = 0.5 * np.sum(mass * np.sum(cm_vel**2, axis=1))
    KE_r = 0.5 * np.sum(L_body**2 / np.array([I1, I2, I3]))
    
    g_t = (2*KE_t - (3*n_mol - 3) * kB*T) / q_t
    g_r = (2*KE_r - 3*n_mol * kB*T) / q_r
    
    xi_t += 0.5 * dt * g_t
    xi_r += 0.5 * dt * g_r
    s_t += 0.5 * dt * xi_t
    s_r += 0.5 * dt * xi_r
    
    return xi_t, xi_r, s_t, s_r


# ─────────────────────────────────────────────
#  Rotational integrator  (symplectic, fixed order)
# ─────────────────────────────────────────────
def half_step_L(L_ang, tau_body, dt, xi_r):
    """Step 1: kick → free rotor → NH rescaling"""
    L_ang = L_ang + 0.5 * dt * tau_body    
    L_ang = _free_rotor_half(L_ang, dt)      
    L_ang = L_ang * np.exp(-0.5 * dt * xi_r) 
    return L_ang

def half_step_L_final(L_ang, tau_body, dt, xi_r):
    """Step 2: NH rescaling → free rotor inv → kick"""
    L_ang = L_ang * np.exp(-0.5 * dt * xi_r) 
    L_ang = _free_rotor_half_inv(L_ang, dt)   
    L_ang = L_ang + 0.5 * dt * tau_body      
    return L_ang

def _free_rotor_half(L, dt, I1=None, I2=None, I3=None):
    """Ry(dt/2) then Rx(dt/2) free rotation of angular momentum."""
    if I1 is None: I1, I2, I3 = SPCE['I_body']
    Lx, Ly, Lz = L.T
    # Ry
    a  = (dt/2) * (1/I3 - 1/I2) * Ly
    ca, sa = np.cos(a), np.sin(a)
    Lx, Lz = ca*Lx + sa*Lz, -sa*Lx + ca*Lz
    # Rx
    b  = (dt/2) * (1/I3 - 1/I1) * Lx
    cb, sb = np.cos(b), np.sin(b)
    Ly, Lz = cb*Ly - sb*Lz, sb*Ly + cb*Lz
    return np.stack([Lx, Ly, Lz], axis=1)

def _free_rotor_half_inv(L, dt, I1=None, I2=None, I3=None):
    """Inverse of _free_rotor_half: Rx then Ry (reversed order)."""
    if I1 is None: I1, I2, I3 = SPCE['I_body']
    Lx, Ly, Lz = L.T
    # Rx inverse
    b  = (dt/2) * (1/I3 - 1/I1) * Lx
    cb, sb = np.cos(b), np.sin(b)
    Ly, Lz = cb*Ly + sb*Lz, -sb*Ly + cb*Lz
    # Ry inverse
    a  = (dt/2) * (1/I3 - 1/I2) * Ly
    ca, sa = np.cos(a), np.sin(a)
    Lx, Lz = ca*Lx - sa*Lz, sa*Lx + ca*Lz
    return np.stack([Lx, Ly, Lz], axis=1)

def full_step_quat(quats, L_body, I_body, dt):
    """
    Advance quaternions by dt using body-frame angular velocity ω = L/I.
    Symmetric Euler: Ry(dt/2) Rx(dt/2) Rz(dt) Rx(dt/2) Ry(dt/2)
    """
    ex = np.array([1.,0.,0.])
    ey = np.array([0.,1.,0.])
    ez = np.array([0.,0.,1.])
    omega = L_body / I_body   # (N,3)

    qy1 = axis_angle_to_quat(ey, omega[:,1] * dt/2)
    qx1 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qz  = axis_angle_to_quat(ez, omega[:,2] * dt)
    qx2 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qy2 = axis_angle_to_quat(ey, omega[:,1] * dt/2)

    q = quats
    q = quat_mul(q, qy1)
    q = quat_mul(q, qx1)
    q = quat_mul(q, qz)
    q = quat_mul(q, qx2)
    q = quat_mul(q, qy2)
    q /= np.linalg.norm(q, axis=1, keepdims=True)   # renormalise
    return q

def world_to_body(quats, v_world):
    """Rotate world-frame vectors to body frame using inverse quaternion."""
    R = Rotation.from_quat(quats[:, [1,2,3,0]])
    return R.inv().apply(v_world)

# ─────────────────────────────────────────────
#  Full Velocity Verlet step (translation + rotation)
# ─────────────────────────────────────────────
def velocity_verlet_step(cm_pos, cm_vel, quats, L_body, forces, tau, mass,
                          I_body, L_box, p, dt, nbr_list, xi_t = 0, xi_r = 0, s_t = 0, s_r = 0, n_mol = 0, T = 0):

    # --- 1. UPDATENHCP premier demi-pas ---
    xi_t, xi_r, s_t, s_r = nh_update(xi_t, xi_r, s_t, s_r, cm_vel, L_body, n_mol, mass, T, dt)

    # --- 2. Half-step translations ---
    cm_vel = np.exp(-0.5 * dt * xi_t) * cm_vel
    cm_vel = cm_vel + 0.5 * dt * forces / mass

    # --- 3. Half-step rotation ---
    tau_body = world_to_body(quats, tau)
    L_body = half_step_L(L_body, tau_body, dt, xi_r)

    # --- 4. Full-step positions ---
    cm_pos = (cm_pos + dt * cm_vel) % L_box

    # --- 5. Full-step quaternions ---
    quats = full_step_quat(quats, L_body, I_body, dt)

    # --- 6. Recompute forces and torques ---
    forces, tau, pe = compute_forces_and_torques(len(cm_pos), cm_pos, quats, L_box, nbr_list, p)

    # --- 7. Second half-step rotation ---
    tau_body = world_to_body(quats, tau)
    L_body = half_step_L_final(L_body, tau_body, dt, xi_r)

    # --- 8. Second half-step translations ---
    cm_vel = cm_vel * np.exp(-0.5 * dt * xi_t)
    cm_vel += 0.5 * dt * forces / mass

    # --- 9. UPDATENHCP deuxième demi-pas ---
    xi_t, xi_r, s_t, s_r = nh_update(xi_t, xi_r, s_t, s_r,
                                       cm_vel, L_body, n_mol, mass, T, dt)

    return cm_pos, cm_vel, quats, L_body, forces, tau, pe, xi_t, xi_r, s_t, s_r