import numpy as np

class spc_e:
    def __init__(self):
        self.mass = 18.015
        self.I_body = np.array([1.3743, 1.9144, 0.6001])
        self.q_o = -0.8476
        self.q_h = 0.4238
        self.OH = 1
        self.HOH_deg = 109.46
        self.HOH_rad = np.radians(self.HOH_deg / 2)
        self.O_body = np.array([0,0,0])
        self.H1_body = self.OH * np.array([0, np.sin(self.HOH_rad), np.cos(self.HOH_rad)])
        self.H2_body = self.OH * np.array([0, -np.sin(self.HOH_rad), np.cos(self.HOH_rad)])
        self.eps_LJ = 15.53
        self.sigma_LJ = 3.166