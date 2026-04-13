import numpy as np

# ─────────────────────────────────────────────
#  Energy diagnostics
# ─────────────────────────────────────────────
def kinetic_energy_trans(cm_vel, mass):
    return 0.5 * mass * np.sum(cm_vel**2)

def kinetic_energy_rot(L_body, I_body):
    return 0.5 * np.sum(L_body**2 / I_body)