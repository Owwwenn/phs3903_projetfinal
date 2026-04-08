import numpy as np

class spc_e:
    def __init__(self):
        self.mass = 18.015
        self.h_mass = 1.008
        self.o_mass = 15.999 
        self.I_body = np.array([1.77, 0.61, 1.16])
        self.q_o = 0.8476
        self.q_h = 0.4238
        self.OH = 1
        self.HOH_deg = 109.46
        self.HOH_rad = np.radians(self.HOH_deg)
        self.z_cm = 2 * self.h_mass * self.OH * np.cos(self.HOH_rad/2) / (self.o_mass + 2*self.h_mass)
        self.O_body = np.array([0,0,0])
        self.H1_body = self.OH * np.array([0, np.sin(self.HOH_rad/2), np.cos(self.HOH_rad/2)])
        self.H2_body = self.OH * np.array([0, -np.sin(self.HOH_rad/2), np.cos(self.HOH_rad/2)])
        self.eps_LJ = 0.1553
        self.sigma_LJ = 3.166