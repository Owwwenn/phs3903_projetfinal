import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────
#  SPC/E parameters
# ─────────────────────────────────────────────
SPCE = dict(
    epsilon  = 0.1553,       # kcal/mol  O-O LJ
    sigma    = 3.1656,       # Å         O-O LJ
    mass     = 18.015,       # amu
    rc_LJ    = 9.0,          # Å  LJ cutoff
    rc_coul  = 10,          # Å  Coulomb cutoff
    q_o      = -0.8476,      # e
    q_h      =  0.4238,      # e
    r_oh     =  1.0,         # Å  O-H bond
    theta    =  np.radians(109.47),   # H-O-H angle
    # SPC/E moments of inertia [amu·Å²]
    I_body   = np.array([1.3743, 1.9144, 0.6001]),
    # Coulomb prefactor in kcal/mol units: k_e * e² in kcal·Å/mol
    k_coul   = 332.0637,
)

kB        = 0.001987   # kcal/(mol·K)
T_UNIT_PS = 0.04888    # 1 t* in ps

# ─────────────────────────────────────────────
#  Molecule geometry helpers
# ─────────────────────────────────────────────
def get_site_offsets(theta, r_oh):
    """
    H site positions in body frame, relative to COM (oxygen at origin here).
    Returns r_h1, r_h2 shape (3,)
    """
    s = np.sin(theta / 2)
    c = np.cos(theta / 2)
    r_h1 = r_oh * np.array([0.0,  s, c])
    r_h2 = r_oh * np.array([0.0, -s, c])
    return r_h1, r_h2

def rotate_vectors_batch(quats, vecs):
    """
    Rotate a fixed vector `vecs` (3,) by each quaternion in `quats` (N,4) [w,x,y,z].
    Returns (N, 3).
    """
    # Use scipy Rotation — expects (x,y,z,w) ordering
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    return R.apply(vecs)

def get_atom_positions(cm_pos, quats, r_h1, r_h2):
    """
    Returns O, H1, H2 world-frame positions, each (N, 3).
    Oxygen coincides with COM (no z_cm offset — avoids Bug #3 from diagnosis).
    """
    O  = cm_pos.copy()
    H1 = cm_pos + rotate_vectors_batch(quats, r_h1)
    H2 = cm_pos + rotate_vectors_batch(quats, r_h2)
    return O, H1, H2

# ─────────────────────────────────────────────
#  Initial state
# ─────────────────────────────────────────────
def make_initial_state(N, rho, T_K, p, seed=42, phase='liquid'):
    np.random.seed(seed)
    L = np.full(3, (N / rho) ** (1/3))

    # ── Positions ──────────────────────────────────────────
    if phase == 'liquid':
        # Cubic lattice — good for dense liquid
        n_side  = int(np.ceil(N ** (1/3)))
        spacing = L[0] / n_side
        idx = np.array([[i, j, k]
                        for i in range(n_side)
                        for j in range(n_side)
                        for k in range(n_side)])[:N]
        cm_pos = idx * spacing

    elif phase == 'gas':
        # Random placement with minimum distance — no overlap
        r_min   = p.get('r_min_gas', 2.5)   # Å, tweak if needed
        cm_pos  = np.zeros((N, 3))
        placed  = 0
        max_try = 100_000
        tries   = 0
        while placed < N and tries < max_try:
            candidate = np.random.uniform(0, L[0], size=3)
            if placed == 0:
                cm_pos[0] = candidate
                placed += 1
            else:
                # Minimum image distances to all placed molecules
                diffs = candidate - cm_pos[:placed]
                diffs -= L[0] * np.round(diffs / L[0])   # PBC
                dists = np.linalg.norm(diffs, axis=1)
                if dists.min() >= r_min:
                    cm_pos[placed] = candidate
                    placed += 1
            tries += 1
        if placed < N:
            raise RuntimeError(
                f"Could only place {placed}/{N} molecules without overlap. "
                f"Try decreasing r_min_gas (currently {r_min} Å) or rho."
            )

    # ── Translational velocities — Maxwell-Boltzmann ───────
    cm_vel  = np.random.randn(N, 3)
    cm_vel -= cm_vel.mean(axis=0)
    ke_t    = 0.5 * p['mass'] * np.sum(cm_vel**2)
    cm_vel *= np.sqrt(1.5 * N * kB * T_K / ke_t)

    # ── Quaternions — identity ─────────────────────────────
    quats = np.zeros((N, 4))
    quats[:, 0] = 1.0

    # ── Angular momenta — Maxwell-Boltzmann ────────────────
    I_body = p['I_body']
    L_body = np.random.randn(N, 3)
    for k in range(3):
        ke_r_k = 0.5 * np.sum(L_body[:, k]**2) / I_body[k]
        target = 0.5 * N * kB * T_K
        L_body[:, k] *= np.sqrt(target / ke_r_k)

    return cm_pos, cm_vel, quats, L_body, L