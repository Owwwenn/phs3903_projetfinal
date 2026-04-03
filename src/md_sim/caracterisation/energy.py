import numpy as np

def kinetic_energy(sys, model): 
    return 0.5 * model.mass * np.sum(sys.cm_vel**2)

def rotational_energy(sys, model):
    return 0.5 * np.sum((sys.L**2) / model.I_body)