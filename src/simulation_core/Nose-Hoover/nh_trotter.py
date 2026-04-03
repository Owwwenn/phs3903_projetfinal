import numpy as np
from simulation_core.time_integrator.time_int_VECTORISED import MDSystem

class spc_e:
    def __init__(self):
        self.mass = 18.015
        self.I_body = np.array([1.3743, 1.9144, 0.6001])
        self.q_o = -0.8476
        self.q_h = 0.4238
        self.OH = 1
        self.HOH_deg = 109.46
        self.HOH_rad = np.radians(self.HOH_deg / 2)
        self_O_body = np.array([0,0,0])
        self.H1_body = self.OH * np.array([0, np.sin(self.HOH_rad), np.cos(self.HOH_rad)])
        self.H1_body = self.OH * np.array([0, -np.sin(self.HOH_rad), np.cos(self.HOH_rad)])
        self.eps_LJ = 0.1553
        self.sigma_LJ = 3.166

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