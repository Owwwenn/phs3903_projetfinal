import numpy as np

def semi_isotropic_barostat_z(L_box, cm_pos, cm_vel, virial_zz, N, T, P0, dt, Q_p):
    """
    Barostat semi-isotrope : seul Lz est ajusté pour contrôler Pzz.
    Compatible avec la géométrie slab (Lx, Ly fixes, vide en z).
    Cible Pzz = P0 (typiquement 0 pour coexistence liquide-vapeur).
    """
    kB  = 0.001987
    V   = np.prod(L_box)
    Pzz = (N * kB * T + virial_zz) / V
    tau_P = np.sqrt(Q_p / (N * kB * max(T, 1.0)))
    dP    = Pzz - P0
    P_ref = max(abs(Pzz), abs(P0), 1e-12)
    mu    = 1.0 + (dt / tau_P) * (dP / P_ref)
    mu    = np.clip(mu, 0.997, 1.003)
    L_box       = L_box.copy()
    L_box[2]   *= mu
    cm_pos      = cm_pos.copy()
    cm_pos[:,2]*= mu
    return L_box, cm_pos, cm_vel, Pzz


def pr_barostat_isotropic(L_box, cm_pos, cm_vel,
                          virial, N, T,
                          eta_p, Q_p, P0, dt):
    """
    Barostat de Berendsen isotrope.

    Pas d'variable eta_p accumulée — la correction est directe à chaque step :
        mu³ = 1 - (dt / tau_P) * (P_inst - P0) / P_inst
    où tau_P = sqrt(Q_p / (N * kB * T))  [en t*]

    Plus stable que Nosé-Hoover-Andersen car pas de drift de eta_p.
    Q_p contrôle la raideur : Q_p = N * kB * T * tau_P**2
    """
    kB = 0.001987
    V  = np.prod(L_box)

    P_inst = (N * kB * T + virial / 3.0) / V

    # Temps de relaxation déduit de Q_p
    tau_P  = np.sqrt(Q_p / (N * kB * max(T, 1.0)))

    # Correction Berendsen : P_inst > P0 → mu > 1 → boite grandit → pression baisse
    dP     = P_inst - P0
    P_ref  = max(abs(P_inst), abs(P0), 1e-12)
    mu3    = 1.0 + (dt / tau_P) * (dP / P_ref)
    mu3    = np.clip(mu3, 0.997**3, 1.003**3)   # max 0.3% par step
    mu     = mu3 ** (1.0 / 3.0)

    L_box  = L_box  * mu
    cm_pos = cm_pos * mu
    # eta_p non utilisé mais conservé pour compatibilité signature
    return L_box, cm_pos, cm_vel, eta_p