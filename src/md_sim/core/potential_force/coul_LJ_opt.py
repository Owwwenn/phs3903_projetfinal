import numpy as np
from numba import njit
from scipy.special import erfc
from md_sim.core.system import get_atom_positions
from md_sim.core.potential_force.ewald import build_coulomb_forces_torques_ewald

# ─────────────────────────────────────────────
#  LJ forces  (O-O only)
# ─────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _lj_kernel(pos, i_idx, j_idx, L, epsilon, sigma, rc):
    n_pairs = len(i_idx)
    n_atoms = pos.shape[0]

    forces    = np.zeros((n_atoms, 3))
    pe        = 0.0
    virial    = 0.0
    virial_zz = 0.0

    rc2  = rc * rc
    sig2 = sigma * sigma

    # energy shift
    rc6    = (sig2 / rc2) ** 3
    Eshift = 4.0 * epsilon * (rc6 * rc6 - rc6)

    for k in range(n_pairs):
        i = i_idx[k]
        j = j_idx[k]

        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]

        # MIC
        dx -= L[0] * np.rint(dx / L[0])
        dy -= L[1] * np.rint(dy / L[1])
        dz -= L[2] * np.rint(dz / L[2])

        r2 = dx*dx + dy*dy + dz*dz
        if r2 >= rc2:
            continue

        r2i  = sig2 / r2
        r6i  = r2i * r2i * r2i
        r12i = r6i * r6i

        # energy
        pe += 4.0 * epsilon * (r12i - r6i) - Eshift

        # force
        fmag = 48.0 * epsilon * r2i * (r12i - 0.5 * r6i)

        fx = fmag * dx
        fy = fmag * dy
        fz = fmag * dz

        # accumulate forces
        forces[i, 0] -= fx
        forces[i, 1] -= fy
        forces[i, 2] -= fz

        forces[j, 0] += fx
        forces[j, 1] += fy
        forces[j, 2] += fz

        # virial
        virial    += dx*fx + dy*fy + dz*fz
        virial_zz += dz*fz

    return forces, pe, virial, virial_zz

def build_lj_forces(n, pos, L, nbr_list, epsilon, sigma, rc):
    i_idx, j_idx = nbr_list
    forces, pe, virial, virial_zz = _lj_kernel(pos, i_idx, j_idx, L, epsilon, sigma, rc)
    return forces, pe, virial, virial_zz

# ─────────────────────────────────────────────
#  Coulomb forces + torques — fully vectorised
#
#  All 9 site-site pairs are stacked along a leading
#  "pair type" axis P=9, so every operation is a single
#  NumPy call over an (P, N, 3) tensor instead of a
#  Python loop over 9 iterations.
#
#  Pair order  (P index):
#    0  OO   1  OH1   2  OH2
#    3  H1O  4  H1H1  5  H1H2
#    6  H2O  7  H2H1  8  H2H2
# ─────────────────────────────────────────────

# Which site on molecule-i / molecule-j for each of the 9 pairs.
# 0 = O, 1 = H1, 2 = H2
_I_SITE = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])   # (9,)
_J_SITE = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])   # (9,)


def build_coulomb_forces_torques(n, O, H1, H2, L, nbr_list, q_o, q_h, k_coul, rc_coul):
    """
    Returns:
        F_coul  : (n, 3) forces on COM
        tau     : (n, 3) torques in world frame
        pe_coul : scalar potential energy
    """
    i_idx, j_idx = nbr_list
    rc2 = rc_coul ** 2

    # ── Site positions stacked as (3, N, 3): axis-0 indexes O/H1/H2 ──────────
    sites_i = np.stack([O[i_idx], H1[i_idx], H2[i_idx]])   # (3, N, 3)
    sites_j = np.stack([O[j_idx], H1[j_idx], H2[j_idx]])   # (3, N, 3)

    # ── Gather the 9 pair combinations → (9, N, 3) ───────────────────────────
    si = sites_i[_I_SITE]   # (9, N, 3)
    sj = sites_j[_J_SITE]   # (9, N, 3)

    # ── Displacement + minimum-image convention ───────────────────────────────
    dr = sj - si                            # (9, N, 3)
    dr -= L * np.round(dr / L)

    # ── Squared distances + cutoff mask ──────────────────────────────────────
    r2   = np.einsum('pni,pni->pn', dr, dr) # (9, N)
    mask = r2 < rc2                          # (9, N)  boolean

    # Avoid division by zero outside the cutoff (values discarded by mask)
    r2_safe = np.where(mask, r2, 1.0)

    # ── Charge products: (9,) broadcast over N ───────────────────────────────
    # qq order matches pair order above
    qq_vals = np.array([
        q_o*q_o, q_o*q_h, q_o*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
    ]) * k_coul                              # (9,)
    qq = qq_vals[:, None]                    # (9, 1) for broadcasting

    # ── Potential energy ──────────────────────────────────────────────────────
    r1      = np.sqrt(r2_safe)               # (9, N)
    pe_coul = np.sum((qq / r1) * mask)

    # ── Force magnitude / r  (fmag · dr gives the force vector) ──────────────
    r3   = r2_safe * r1                      # (9, N)
    fmag = np.where(mask, -qq / r3, 0.0)    # (9, N)  zero outside cutoff

    # ── Force vectors ─────────────────────────────────────────────────────────
    f_vec     = fmag[:, :, None] * dr        # (9, N, 3)
    virial    = np.sum(dr * f_vec)
    virial_zz = np.sum(dr[:, :, 2] * f_vec[:, :, 2])

    # ── Lever arms from COM (= O position) ───────────────────────────────────
    # levers_i[s] = site_s_position[i_idx] - O[i_idx]
    levers_i = sites_i - sites_i[0:1]        # (3, N, 3)  O lever = 0 by construction
    levers_j = sites_j - sites_j[0:1]        # (3, N, 3)

    lev_i = levers_i[_I_SITE]               # (9, N, 3)
    lev_j = levers_j[_J_SITE]               # (9, N, 3)

    # ── Torques: τ = lever × F ────────────────────────────────────────────────
    tau_i =  np.cross(lev_i,  f_vec)         # (9, N, 3)
    tau_j =  np.cross(lev_j, -f_vec)         # (9, N, 3)

    # ── Reduce over the 9 pair types before scatter-add ──────────────────────
    # Summing first cuts add.at calls from 9×4=36 down to 4.
    fi_sum = f_vec.sum(axis=0)               # (N, 3)
    fj_sum = (-f_vec).sum(axis=0)
    ti_sum = tau_i.sum(axis=0)               # (N, 3)
    tj_sum = tau_j.sum(axis=0)

    F_coul = np.zeros((n, 3))
    tau    = np.zeros((n, 3))

    np.add.at(F_coul, i_idx, fi_sum)
    np.add.at(F_coul, j_idx, fj_sum)
    np.add.at(tau,    i_idx, ti_sum)
    np.add.at(tau,    j_idx, tj_sum)

    return F_coul, tau, pe_coul, virial, virial_zz

def build_coulomb_forces_torques_wolf(n, O, H1, H2, L, nbr_list, q_o, q_h, k_coul, rc_coul, alpha):
    """
    Returns:
        F_coul  : (n, 3) forces on COM
        tau     : (n, 3) torques in world frame
        pe_coul : scalar potential energy
    """
    i_idx, j_idx = nbr_list
    rc2 = rc_coul ** 2

    sites_i = np.stack([O[i_idx], H1[i_idx], H2[i_idx]])
    sites_j = np.stack([O[j_idx], H1[j_idx], H2[j_idx]])

    si = sites_i[_I_SITE]
    sj = sites_j[_J_SITE]

    dr = sj - si
    dr -= L * np.round(dr / L)

    r2   = np.einsum('pni,pni->pn', dr, dr)
    mask = r2 < rc2
    r2_safe = np.where(mask, r2, 1.0)

    qq_vals = np.array([
        q_o*q_o, q_o*q_h, q_o*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
    ]) * k_coul
    qq = qq_vals[:, None]

    r1 = np.sqrt(r2_safe)

    # ── Wolf terms ───────────────────────────────────────────
    erfc_term = erfc(alpha * r1)
    exp_term  = np.exp(-(alpha**2) * r2_safe)

    # cutoff constants
    erfc_rc = erfc(alpha * rc_coul)
    exp_rc  = np.exp(-(alpha**2) * rc_coul**2)

    dV_rc = -(erfc_rc / rc_coul**2 + (2*alpha/np.sqrt(np.pi)) * exp_rc / rc_coul)

    # ── Potential energy (shifted-force) ─────────────────────
    pe_coul = np.sum(qq * (
        erfc_term / r1
        - erfc_rc / rc_coul
        - (r1 - rc_coul) * dV_rc
    ) * mask)

    # ── Forces ───────────────────────────────────────────────
    r3 = r2_safe * r1

    fmag = np.where(
        mask,
        -qq * (
            erfc_term / r3
            + (2*alpha/np.sqrt(np.pi)) * exp_term / r2_safe
            - dV_rc / r1
        ),
        0.0
    )

    f_vec     = fmag[:, :, None] * dr
    virial    = np.sum(dr * f_vec)
    virial_zz = np.sum(dr[:, :, 2] * f_vec[:, :, 2])

    levers_i = sites_i - sites_i[0:1]
    levers_j = sites_j - sites_j[0:1]

    lev_i = levers_i[_I_SITE]
    lev_j = levers_j[_J_SITE]

    tau_i =  np.cross(lev_i,  f_vec)
    tau_j =  np.cross(lev_j, -f_vec)

    fi_sum = f_vec.sum(axis=0)
    fj_sum = (-f_vec).sum(axis=0)
    ti_sum = tau_i.sum(axis=0)
    tj_sum = tau_j.sum(axis=0)

    F_coul = np.zeros((n, 3))
    tau    = np.zeros((n, 3))

    np.add.at(F_coul, i_idx, fi_sum)
    np.add.at(F_coul, j_idx, fj_sum)
    np.add.at(tau,    i_idx, ti_sum)
    np.add.at(tau,    j_idx, tj_sum)

    return F_coul, tau, pe_coul, virial, virial_zz

# ─────────────────────────────────────────────
#  Combined force/torque
# ─────────────────────────────────────────────

def compute_forces_and_torques(n, cm_pos, quats, L, nbr_list, p, mode='normal'):
    O, H1, H2 = get_atom_positions(cm_pos, quats, p['r_h1'], p['r_h2'])
    nbr_list_LJ, nbr_list_coul = nbr_list

    F_lj, pe_lj, virial_LJ, virial_LJ_zz = build_lj_forces(n, cm_pos, L, nbr_list_LJ,
                                        p['epsilon'], p['sigma'], p['rc_LJ'])

    if mode == 'ewald':
        F_coul, tau, pe_coul, virial_coul, virial_coul_zz = build_coulomb_forces_torques_ewald(
            n, O, H1, H2, L, nbr_list_coul,
            p['q_o'], p['q_h'], p['k_coul'], p['rc_coul'],
            p['alpha'], p.get('kmax', 6)
        )
    elif mode == 'wolf':
        F_coul, tau, pe_coul, virial_coul, virial_coul_zz = build_coulomb_forces_torques_wolf(n, O, H1, H2, L, nbr_list_coul,
                                                            p['q_o'], p['q_h'],
                                                            p['k_coul'], p['rc_coul'], p['alpha'])
    elif mode == 'normal':
        F_coul, tau, pe_coul, virial_coul, virial_coul_zz = build_coulomb_forces_torques(n, O, H1, H2, L, nbr_list_coul,
                                                            p['q_o'], p['q_h'],
                                                            p['k_coul'], p['rc_coul'])
    return F_lj + F_coul, tau, pe_lj + pe_coul, virial_LJ + virial_coul, virial_LJ_zz + virial_coul_zz