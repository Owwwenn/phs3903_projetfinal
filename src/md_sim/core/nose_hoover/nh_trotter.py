import numpy as np
from md_sim.models.three_site import spc_e
from md_sim.core.time_integrator.time_int_VECTORISED import MDSystem

kB = 0.831446
         
def kinetic_energy(sys, model): 
    return 0.5 * model.mass * np.sum(sys.cm_vel**2)

def rotational_energy(sys, model):
    return 0.5 * np.sum((sys.L**2) / model.I_body)

def update_eta(sys, model, N, dt):
    g = 6*N 
    relax_time = 100*dt
    K = kinetic_energy(sys, model) + rotational_energy(sys, model)
    Q = g * kB * sys.T * relax_time**2 
    sys.eta += dt/2/Q * (2*K - g*kB*sys.T)    

def update_PL(sys, dt):
    C = np.exp(-sys.eta * dt /2)
    sys.cm_vel *= C 
    sys.L *= C 