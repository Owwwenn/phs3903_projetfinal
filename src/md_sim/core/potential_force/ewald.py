"""
build_coulomb_forces_torques_ewald
===================================
Drop-in replacement de build_coulomb_forces_torques_wolf.

Même signature d'appel, mêmes outputs :
    F_coul  : (n, 3)  forces sur CM
    tau     : (n, 3)  couples en frame monde
    pe_coul : float   énergie potentielle coulombienne [kcal/mol]
    virial  : float   pour barostat Nosé-Hoover NPT

Structure interne identique à ton code Wolf :
  - nbr_list sur indices de molécules (CM)
  - sites O/H1/H2 construits avec get_atom_positions
  - 9 paires site-site vectorisées sur axe P
  - scatter-add final via np.bincount (plus rapide que np.add.at)

Ewald = 4 contributions :
  U = U_real + U_recip + U_self + U_surf

U_real  : somme dans l'espace réel avec erfc, même structure que Wolf
U_recip : somme sur vecteurs k, Numba parallel (depuis _ewald_reciprocal.py)
U_self  : correction auto-interaction (scalaire, pas de force)
U_surf  : terme dipolaire tin-foil — CRITIQUE pour slab liquide-vapeur

Paramètres dans p{} :
    p['alpha']  : Å⁻¹  — typiquement 3.5/rc_coul
    p['kmax']   : int  — vecteurs k par direction, typiquement 5-7

Unités internes : Å, e, kcal/mol  (k_coul = 332.0637)
"""

import numpy as np
from scipy.special import erfc
from md_sim.core.potential_force._ewald_reciprocal import _ewald_reciprocal

# Ordre des 9 paires site-site — identique à ton code Wolf
_I_SITE = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
_J_SITE = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])


def _make_qq(q_o, q_h, k_coul):
    """(9,) array de q_i * q_j * k_coul pour les 9 paires."""
    return np.array([
        q_o*q_o, q_o*q_h, q_o*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
        q_h*q_o, q_h*q_h, q_h*q_h,
    ]) * k_coul


# =============================================================================
# PARTIE RÉELLE — identique à Wolf mais sans terme de shift
# =============================================================================

def _ewald_real(si, sj, dr, r2_safe, r1, mask, qq, alpha, rc_coul):
    """
    U_real = Σ q_i q_j erfc(α r) / r
    F_real = -∇U_real

    Même structure tensorielle (9, N, 3) que ton Wolf.
    """
    ar      = alpha * r1                                   # (9, N)
    erfc_ar = erfc(ar)                                     # (9, N)
    exp_ar2 = np.exp(-(alpha**2) * r2_safe)                # (9, N)

    pe_real = np.sum(qq[:, None] * erfc_ar / r1 * mask)

    two_alpha_sqrtpi = 2.0 * alpha / np.sqrt(np.pi)
    r3 = r2_safe * r1

    fmag = np.where(
        mask,
        -qq[:, None] * (erfc_ar / r3 + two_alpha_sqrtpi * exp_ar2 / r2_safe),
        0.0
    )                                                      # (9, N)

    f_vec = fmag[:, :, None] * dr                          # (9, N, 3)
    return pe_real, f_vec


# =============================================================================
# CORRECTION SELF
# =============================================================================

def _ewald_self(all_charges_flat, alpha):
    """U_self = -α/√π Σ_i q_i²   (pas de force, énergie seulement)"""
    return -332.0637 * alpha / np.sqrt(np.pi) * np.sum(all_charges_flat**2)


# =============================================================================
# TERME DE SURFACE — tin-foil (ε_surf → ∞)
# =============================================================================

def _ewald_surface(all_pos_flat, all_charges_flat, Lx, Ly, Lz):
    """
    U_surf = (2π/3V) |M|²   où M = Σ_i q_i r_i

    CRITIQUE pour boîte slab liquide-vapeur.
    Forces : F_i = -(4π/3V) q_i M
    """
    V = Lx * Ly * Lz
    prefactor = 332.0637 * 2.0 * np.pi / (3.0 * V)

    Mx = np.sum(all_charges_flat * all_pos_flat[:, 0])
    My = np.sum(all_charges_flat * all_pos_flat[:, 1])
    Mz = np.sum(all_charges_flat * all_pos_flat[:, 2])
    M2 = Mx*Mx + My*My + Mz*Mz

    pe_surf = prefactor * M2

    forces = np.zeros_like(all_pos_flat)
    forces[:, 0] = -2.0 * prefactor * all_charges_flat * Mx
    forces[:, 1] = -2.0 * prefactor * all_charges_flat * My
    forces[:, 2] = -2.0 * prefactor * all_charges_flat * Mz

    return pe_surf, forces


# =============================================================================
# FONCTION PRINCIPALE — drop-in replacement de build_coulomb_forces_torques_wolf
# =============================================================================

def build_coulomb_forces_torques_ewald(
        n, O, H1, H2, L, nbr_list,
        q_o, q_h, k_coul, rc_coul, alpha, kmax=6):
    """
    Ewald complet pour SPC/E en boîte slab.

    Paramètres
    ----------
    n        : nombre de molécules
    O,H1,H2  : (n, 3) positions des sites en coordonnées absolues
    L        : (3,) dimensions de boîte [Lx, Ly, Lz]
    nbr_list : (i_idx, j_idx) sur indices de molécules (CM)
    q_o, q_h : charges SPC/E en e
    k_coul   : constante de Coulomb (332.0637 kcal·Å/mol/e²)
    rc_coul  : cutoff partie réelle [Å]
    alpha    : paramètre Ewald [Å⁻¹], typiquement 3.5/rc_coul
    kmax     : nb de vecteurs k par direction (typiquement 6)

    Retourne
    --------
    F_coul  : (n, 3)  forces sur CM   [kcal/mol/Å]
    tau     : (n, 3)  couples          [kcal/mol]
    pe_coul : float   énergie totale   [kcal/mol]
    virial  : float   pour barostat
    """
    Lx, Ly, Lz = L[0], L[1], L[2]
    i_idx, j_idx = nbr_list
    rc2 = rc_coul**2
    qq  = _make_qq(q_o, q_h, k_coul)   # (9,)

    # ── 1. Partie réelle — même tenseur (9, N, 3) que Wolf ───────────────────
    sites_i = np.stack([O[i_idx], H1[i_idx], H2[i_idx]])   # (3, N_pairs, 3)
    sites_j = np.stack([O[j_idx], H1[j_idx], H2[j_idx]])

    si = sites_i[_I_SITE]   # (9, N_pairs, 3)
    sj = sites_j[_J_SITE]

    dr   = sj - si
    dr  -= L * np.round(dr / L)                             # MIC

    r2      = np.einsum('pni,pni->pn', dr, dr)              # (9, N_pairs)
    mask    = r2 < rc2
    r2_safe = np.where(mask, r2, 1.0)
    r1      = np.sqrt(r2_safe)

    pe_real, f_vec_real = _ewald_real(
        si, sj, dr, r2_safe, r1, mask, qq, alpha, rc_coul
    )

    # ── 2. Partie réciproque — Numba parallel, sur TOUS les sites (3n) ───────
    all_pos     = np.concatenate([O, H1, H2], axis=0)         # (3n, 3)
    all_charges = np.concatenate([
        np.full(n, q_o),
        np.full(n, q_h),
        np.full(n, q_h),
    ])                                                          # (3n,)

    pe_recip, f_recip_flat = _ewald_reciprocal(
        all_pos, all_charges, Lx, Ly, Lz, alpha, kmax
    )

    # ── 3. Correction self (énergie seulement) ───────────────────────────────
    pe_self = _ewald_self(all_charges, alpha)

    # ── 4. Terme de surface — tin-foil ───────────────────────────────────────
    pe_surf, f_surf_flat = _ewald_surface(
        all_pos, all_charges, Lx, Ly, Lz
    )

    # ── 5. Énergie totale ─────────────────────────────────────────────────────
    pe_coul = pe_real + pe_recip + pe_self + pe_surf

    # ── 6. Forces réciproque + surface → scatter sur molécules ───────────────
    # f_recip_flat et f_surf_flat sont (3n, 3) avec ordre [O×n, H1×n, H2×n]
    f_ks_O  = f_recip_flat[:n]     + f_surf_flat[:n]
    f_ks_H1 = f_recip_flat[n:2*n]  + f_surf_flat[n:2*n]
    f_ks_H2 = f_recip_flat[2*n:]   + f_surf_flat[2*n:]

    lever_H1 = H1 - O   # (n, 3)
    lever_H2 = H2 - O   # (n, 3)

    F_ks  = f_ks_O + f_ks_H1 + f_ks_H2   # (n, 3)

    tau_ks = (np.cross(lever_H1, f_ks_H1)
             + np.cross(lever_H2, f_ks_H2))   # (n, 3)

    # ── 7. Forces réelles → scatter sur molécules ────────────────────────────
    levers_i = sites_i - sites_i[0:1]   # (3, N_pairs, 3)
    levers_j = sites_j - sites_j[0:1]

    lev_i = levers_i[_I_SITE]           # (9, N_pairs, 3)
    lev_j = levers_j[_J_SITE]

    tau_i_real =  np.cross(lev_i,  f_vec_real)
    tau_j_real =  np.cross(lev_j, -f_vec_real)

    fi_sum = f_vec_real.sum(axis=0)              # (N_pairs, 3)
    fj_sum = (-f_vec_real).sum(axis=0)
    ti_sum = tau_i_real.sum(axis=0)
    tj_sum = tau_j_real.sum(axis=0)

    F_real   = np.zeros((n, 3))
    tau_real = np.zeros((n, 3))
    for d in range(3):
        F_real[:, d]   += np.bincount(i_idx, weights=fi_sum[:, d], minlength=n)
        F_real[:, d]   += np.bincount(j_idx, weights=fj_sum[:, d], minlength=n)
        tau_real[:, d] += np.bincount(i_idx, weights=ti_sum[:, d], minlength=n)
        tau_real[:, d] += np.bincount(j_idx, weights=tj_sum[:, d], minlength=n)

    # ── 8. Assemblage final ───────────────────────────────────────────────────
    F_coul = F_real + F_ks
    tau    = tau_real + tau_ks

    # virial : Σ r_ij · F_ij  (partie réelle seulement, convention standard)
    virial    = np.sum(dr * f_vec_real)
    virial_zz = np.sum(dr[:, :, 2] * f_vec_real[:, :, 2])

    return F_coul, tau, pe_coul, virial, virial_zz
