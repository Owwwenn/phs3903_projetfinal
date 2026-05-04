import numpy as np

# ─────────────────────────────────────────────
#  Energy diagnostics
# ─────────────────────────────────────────────
def kinetic_energy_trans(cm_vel, mass):
    return 0.5 * mass * np.sum(cm_vel**2)

def kinetic_energy_rot(L_body, I_body):
    return 0.5 * np.sum(L_body**2 / I_body)

def get_cv_fluctuations(te_arr, T_arr, step, N, kB, window=200):
    """Cv via fluctuations NVT : Var(E) / (N * kB * T_mean²).
    Retourne nan si T n'est pas stable dans la fenêtre (rampe de chauffe)."""
    if step < window:
        return np.nan
    E_win = te_arr[step - window : step]
    T_win = T_arr[step - window : step]
    T_mean = T_win.mean()
    if T_mean < 1.0:
        return np.nan
    # Rejette les fenêtres hors-équilibre (rampe de T)
    if T_win.std() > 0.05 * T_mean:
        return np.nan
    return np.var(E_win) / (N * kB * T_mean**2)