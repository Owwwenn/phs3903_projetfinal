import numpy as np
from system import get_atom_positions
# ─────────────────────────────────────────────
#  LJ forces  (O-O only, same as before)
# ─────────────────────────────────────────────
def build_lj_forces(n, pos, L, nbr_list, epsilon, sigma, rc):
    i_idx, j_idx = nbr_list
    dr = pos[j_idx] - pos[i_idx]
    dr -= L * np.round(dr / L)
    r2 = np.einsum('ij,ij->i', dr, dr)
    mask = r2 < rc**2
    i_idx, j_idx, dr, r2 = i_idx[mask], j_idx[mask], dr[mask], r2[mask]

    rc6     = (sigma**2 / rc**2)**3
    E_shift = 4.0 * epsilon * (rc6**2 - rc6)
    r2i  = sigma**2 / r2
    r6i  = r2i**3
    r12i = r6i**2

    pe     = np.sum(4.0 * epsilon * (r12i - r6i) - E_shift)
    fmag_r = 48.0 * epsilon * r2i * (r12i - 0.5 * r6i)
    f_vec  = fmag_r[:, None] * dr

    forces = np.zeros((n, 3))
    np.add.at(forces, i_idx, -f_vec)
    np.add.at(forces, j_idx,  f_vec)
    return forces, pe

# ─────────────────────────────────────────────
#  Coulomb forces + torques
#  9 site-site pairs per molecule pair: OO, OH1, OH2, H1O, H1H1, H1H2, H2O, H2H1, H2H2
# ─────────────────────────────────────────────
def build_coulomb_forces_torques(n, O, H1, H2, L, nbr_list, q_o, q_h, k_coul, rc_coul):
    """
    Returns:
        F_coul  : (n, 3) forces on COM
        tau     : (n, 3) torques in world frame
        pe_coul : scalar potential energy
    """
    i_idx, j_idx = nbr_list

    # site positions for each pair
    # molecule i sites
    Oi  = O[i_idx];  H1i = H1[i_idx]; H2i = H2[i_idx]
    # molecule j sites
    Oj  = O[j_idx];  H1j = H1[j_idx]; H2j = H2[j_idx]

    # 9 displacement vectors r_ab = site_b_j - site_a_i, with MIC
    pairs = [
        (Oi,  Oj,  q_o, q_o),
        (Oi,  H1j, q_o, q_h),
        (Oi,  H2j, q_o, q_h),
        (H1i, Oj,  q_h, q_o),
        (H1i, H1j, q_h, q_h),
        (H1i, H2j, q_h, q_h),
        (H2i, Oj,  q_h, q_o),
        (H2i, H1j, q_h, q_h),
        (H2i, H2j, q_h, q_h),
    ]

    F_coul  = np.zeros((n, 3))
    tau     = np.zeros((n, 3))
    pe_coul = 0.0

    # lever arms from COM to each site on molecule i (world frame)
    lever_i = [
        O[i_idx]  - O[i_idx],   # O lever = 0 (O is at COM)
        H1[i_idx] - O[i_idx],   # H1 lever
        H1[i_idx] - O[i_idx],   # H1 lever (repeated for H1-H1, H1-H2)
        H2[i_idx] - O[i_idx],   # H2 lever
        H2[i_idx] - O[i_idx],
        H2[i_idx] - O[i_idx],
    ]
    # map each of the 9 pairs to which site on molecule i it belongs
    #        OO  OH1 OH2 H1O H1H1 H1H2 H2O H2H1 H2H2
    i_site = [ 0,  0,  0,  1,   1,   1,   2,   2,   2]
    levers_i_map = [
        O[i_idx]  - O[i_idx],   # site 0: O (zero lever)
        H1[i_idx] - O[i_idx],   # site 1: H1
        H2[i_idx] - O[i_idx],   # site 2: H2
    ]

    # similarly for molecule j
    #        OO  OH1 OH2 H1O H1H1 H1H2 H2O H2H1 H2H2
    j_site = [ 0,  1,  2,  0,   1,   2,   0,   1,   2]
    levers_j_map = [
        O[j_idx]  - O[j_idx],   # site 0: O
        H1[j_idx] - O[j_idx],   # site 1: H1
        H2[j_idx] - O[j_idx],   # site 2: H2
    ]

    for k, (si, sj, qi, qj) in enumerate(pairs):
        dr = sj - si
        dr -= L * np.round(dr / L)           # MIC — applied on actual site-site vector (fixes Bug #2)

        r2   = np.einsum('ij,ij->i', dr, dr)
        mask = r2 < rc_coul**2
        if not mask.any():
            continue

        r2m   = r2[mask]
        drm   = dr[mask]
        i_m   = i_idx[mask]
        j_m   = j_idx[mask]

        r1    = np.sqrt(r2m)
        r3    = r2m * r1

        qq    = k_coul * qi * qj
        pe_coul += np.sum(qq / r1)

        fmag  = -qq / r3          # scalar, sign convention: F_i gets +fmag*dr
        f_vec = fmag[:, None] * drm

        np.add.at(F_coul, i_m,  f_vec)
        np.add.at(F_coul, j_m, -f_vec)

        # torques: τ = lever × F
        lev_i = levers_i_map[i_site[k]][mask]
        lev_j = levers_j_map[j_site[k]][mask]

        np.add.at(tau, i_m,  np.cross(lev_i,  f_vec))
        np.add.at(tau, j_m,  np.cross(lev_j, -f_vec))

    return F_coul, tau, pe_coul

# ─────────────────────────────────────────────
#  Combined force/torque
# ─────────────────────────────────────────────
def compute_forces_and_torques(n, cm_pos, quats, L, nbr_list, p):
    O, H1, H2 = get_atom_positions(cm_pos, quats, p['r_h1'], p['r_h2'])

    F_lj,    pe_lj    = build_lj_forces(n, cm_pos, L, nbr_list,
                                         p['epsilon'], p['sigma'], p['rc_LJ'])
    F_coul, tau, pe_coul = build_coulomb_forces_torques(n, O, H1, H2, L, nbr_list,
                                                         p['q_o'], p['q_h'],
                                                         p['k_coul'], p['rc_coul'])
    return F_lj + F_coul, tau, pe_lj + pe_coul