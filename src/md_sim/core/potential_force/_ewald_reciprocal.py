"""
Partie réciproque Ewald — Numba parallel, optimisé slab + demi-espace.

Optimisations :
  1. Auto-détection slab (Lz > 1.5 × min(Lx, Ly)) → kmax_z réduit
  2. Demi-espace canonique (symétrie k <-> -k) → Nk divisé par ~2
  3. Combiné : 2196 → ~420 vecteurs k (-80%)
  4. f_k réduit : (422, N, 3) ~8 MB au lieu de (2196, N, 3) ~40 MB
  5. Cache cos/sin par atome conservé (une seule évaluation trig par (i,k))
"""

import numpy as np
from numba import njit, prange
from math import cos, sin, pi

# =============================================================================
# CONSTRUCTION DES VECTEURS K — demi-espace anisotrope (slab + symétrie)
# =============================================================================

def make_kvecs_half_aniso(Lx, Ly, Lz, kmax_xy, kmax_z):
    """
    Vecteurs k du demi-espace canonique avec kmax anisotrope.

    Garde le représentant canonique de chaque paire (k, -k) :
      nx > 0  OU  (nx==0, ny > 0)  OU  (nx==0, ny==0, nz > 0)
    """
    kv = []
    for nx in range(0, kmax_xy + 1):
        for ny in range(-kmax_xy, kmax_xy + 1):
            for nz in range(-kmax_z, kmax_z + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                if nx == 0 and ny < 0:
                    continue
                if nx == 0 and ny == 0 and nz < 0:
                    continue
                kv.append([2*pi/Lx*nx, 2*pi/Ly*ny, 2*pi/Lz*nz])
    return np.array(kv, dtype=np.float64)


def _make_ewald_factors(kvecs, alpha):
    """exp(-k²/4α²)/k² pour chaque vecteur k."""
    k2 = np.sum(kvecs**2, axis=1)
    return np.exp(-k2 / (4.0 * alpha**2)) / k2


# =============================================================================
# KERNEL NUMBA — une passe, cache cos/sin, parallèle sur k
# =============================================================================

@njit(parallel=True, fastmath=True)
def _recip_kernel(all_pos, all_charges, kvecs, ewald_fac, pref):
    """
    Calcule U_recip et les forces réciproques.

    Parallélisé sur les Nk vecteurs k (indépendants).
    Cache cos/sin par atome : chaque trig calculé une seule fois par (i, k).
    pref = k_coul × 2π/V × 2  (le ×2 compense le demi-espace).

    Allocation : f_k(Nk, N, 3) — avec Nk~420 et N~768, ~8 MB au lieu de ~40 MB.
    """
    N  = all_pos.shape[0]
    Nk = kvecs.shape[0]

    pe_k = np.zeros(Nk)
    f_k  = np.zeros((Nk, N, 3))

    for ik in prange(Nk):
        kx = kvecs[ik, 0]; ky = kvecs[ik, 1]; kz = kvecs[ik, 2]
        ef = ewald_fac[ik]

        S_re = 0.0; S_im = 0.0
        cos_phi = np.empty(N)
        sin_phi = np.empty(N)
        for i in range(N):
            phi = kx*all_pos[i,0] + ky*all_pos[i,1] + kz*all_pos[i,2]
            cp = cos(phi); sp = sin(phi)
            cos_phi[i] = cp; sin_phi[i] = sp
            S_re += all_charges[i] * cp
            S_im += all_charges[i] * sp

        pe_k[ik] = pref * ef * (S_re*S_re + S_im*S_im)

        fk_pref = pref * 2.0 * ef
        for i in range(N):
            fk = fk_pref * all_charges[i] * (S_re*sin_phi[i] - S_im*cos_phi[i])
            f_k[ik, i, 0] = fk * kx
            f_k[ik, i, 1] = fk * ky
            f_k[ik, i, 2] = fk * kz

    return pe_k.sum(), f_k.sum(axis=0)


# =============================================================================
# INTERFACE PUBLIQUE
# =============================================================================

_cache = {}

def _ewald_reciprocal(all_pos, all_charges, Lx, Ly, Lz, alpha, kmax):
    """
    Interface drop-in.

    Auto-détecte la géométrie slab (Lz > 1.5 × min(Lx, Ly)) et réduit
    kmax_z. Utilise le demi-espace pour diviser Nk par ~2.
    Cache invalidé si L ou alpha change (NPT) — appeler clear_cache().
    """
    Lmin   = min(Lx, Ly)
    kmax_z = max(2, int(kmax * Lmin / Lz)) if Lz > 1.5 * Lmin else kmax

    cache_key = (round(Lx, 4), round(Ly, 4), round(Lz, 4),
                 round(alpha, 8), kmax, kmax_z)

    if cache_key not in _cache:
        kvecs     = make_kvecs_half_aniso(Lx, Ly, Lz, kmax, kmax_z)
        ewald_fac = _make_ewald_factors(kvecs, alpha)
        V         = Lx * Ly * Lz
        pref      = 332.0637 * 2.0 * pi / V * 2.0   # ×2 pour demi-espace
        _cache[cache_key] = (kvecs, ewald_fac, pref)
        if len(_cache) > 4:
            del _cache[next(iter(_cache))]
    else:
        kvecs, ewald_fac, pref = _cache[cache_key]

    return _recip_kernel(all_pos, all_charges, kvecs, ewald_fac, pref)


def clear_cache():
    """Appeler après un step barostat NPT qui change L."""
    _cache.clear()


# =============================================================================
# WARM-UP — compile le kernel JIT à l'import
# =============================================================================

def _warmup():
    _p    = np.random.rand(12, 3).astype(np.float64) * 9.
    _q    = np.array([-0.8476, 0.4238, 0.4238] * 4, dtype=np.float64)
    _kv   = make_kvecs_half_aniso(10., 10., 50., 2, 1)
    _ef   = _make_ewald_factors(_kv, 0.35)
    _pref = 332.0637 * 2.0 * pi / 5000. * 2.0
    _recip_kernel(_p, _q, _kv, _ef, _pref)

_warmup()
