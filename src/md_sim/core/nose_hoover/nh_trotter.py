import numpy as np
from md_sim.caracterisation.energy import kinetic_energy, rotational_energy
from md_sim.models.three_site import spc_e
from md_sim.core.time_integrator.time_int_VECTORISED import MDSystem

kB = 0.831446
         
def update_eta(sys, model, param, N, dt):
    g = 6*N 
    relax_time = 1000*dt
    K = kinetic_energy(sys, model) + rotational_energy(sys, model)
    Q = g * kB * param.T_target * relax_time**2 
    sys.eta += dt/2/Q * (2*K - g*kB*param.T_target)    

def update_PL(sys, dt):
    C = np.exp(-sys.eta * dt /2)
    sys.cm_vel *= C 
    sys.L *= C 