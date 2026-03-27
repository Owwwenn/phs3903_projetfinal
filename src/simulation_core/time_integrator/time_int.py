from dataclasses import dataclass
from dataclasses import dataclass, field
import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation

@dataclass
class molecule():
     cm_pos: np.ndarray = field(default_factory=lambda: np.zeros(3)) # position du cm
     cm_vel: np.ndarray = field(default_factory=lambda: np.zeros(3)) # vitesse du CM
     force: np.ndarray = field(default_factory=lambda: np.zeros(3)) # force total sur le CM
     quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # quaternion (w, x, y, z)
     T: np.ndarray = field(default_factory=lambda: np.zeros(3)) # torque in lab frame
     L: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in body frame

mmass = 18.015 #masse molaire eau 




def initialize_molecules(N, L):
     
    
     n_side = int(np.ceil(N**(1/3)))
     d = L / n_side
     
     #initialisation des vitesses:
     kB = 0.831446
     std_div = np.sqrt(kB * 273 / mmass)
     
     
     #initialisation des positions:
     positions = []
     for i in range (n_side):
          for j in range (n_side):
               for k in range (n_side):
                    positions.append(np.array([i*d, j*d, k*d]))

     molecules = [molecule(positions[m],np.random.normal(0,std_div, size = 3)) for m in range(N)]

     mean_vel = np.mean([mol.cm_vel for mol in molecules], axis=0)
     for mol in molecules:
        mol.cm_vel -= mean_vel

     return molecules

print(initialize_molecules(10, 30))

def compute_forces(molecules):
    for mol in molecules:
        mol.force = np.zeros(3)#[1.0, 0.0, 0.0]

def compute_torques(molecules):
    for mol in molecules:
        mol.T = np.zeros(3)#[1.0, 0.0, 0.0]

def half_step_velocity(molecules, dt):
    for mol in molecules:
        mol.cm_vel += 0.5 * (mol.force / mmass) * dt

def full_step_position(molecules, dt):
    for mol in molecules:
        mol.cm_pos += mol.cm_vel * dt

#Rajoutez la fonction du gradiant qui calcule F a r+1 

def half_step_velocity_final(molecules, dt):
    for mol in molecules:
        mol.cm_vel += 0.5 * (mol.force / mmass) * dt

        




# SPC/E geometry in body frame (relative to COM), in Angstroms
OH_BOND = 1.0        # O-H bond length
HOH_ANGLE = 109.47   # degrees

# oxygen is at the center of mass
O_body = np.array([0.0, 0.0, 0.0])

# hydrogen positions in body frame
angle_rad = np.radians(HOH_ANGLE / 2)
H1_body = np.array([ np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND
H2_body = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND

I_body = np.array([1.3743, 1.9144, 0.6001])
i1 , i2, i3 = I_body[0], I_body[1], I_body[2]

# for mol in molecules:
#         # convert torque from lab frame to body frame
#         w, x, y, z = mol.quat
#         R = Rotation.from_quat([x, y, z, w]).as_matrix()

def get_atom_positions(mol):
    
    w, x, y, z = mol.quat
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    
    O_lab  = mol.cm_pos + R @ O_body
    H1_lab = mol.cm_pos + R @ H1_body
    H2_lab = mol.cm_pos + R @ H2_body
    
    return O_lab, H1_lab, H2_lab

def Ry(mol, dt):
    # angle depends on current S_y component and moments of inertia
    alpha = (dt / 2) * (1/i3 - 1/i2) * mol.L[1]
    
    
    Ry = np.array([
        [ np.cos(alpha), 0, np.sin(alpha)],
        [ 0,             1, 0            ],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])
    return Ry

def Rx(mol, dt):
    # angle depends on current S_x component and moments of inertia
    beta = (dt / 2) * (1/i3 - 1/i1) * mol.L[0]
    
    # build Rx matrix
    Rx = np.array([
        [1, 0,             0            ],
        [0,  np.cos(beta), -np.sin(beta)],
        [0,  np.sin(beta),  np.cos(beta)]
    ])

    return Rx

def E(omega):
    w1, w2, w3 = omega
    E = np.array([
        [ 0,  -w1, -w2, -w3],
        [ w1,   0,  w3, -w2],
        [ w2, -w3,   0,  w1],
        [ w3,  w2, -w1,   0]
    ])
    
    return E


def half_step_L(molecules, dt):
    for mol in molecules:
        # step 3: half step torque
        mol.L += 0.5 * dt * mol.T
        
        # step 4: Ry rotation
        mol.L = Ry(mol, dt) @ mol.L
        
        # step 5: Rx rotation (uses updated L from step 4)
        mol.L = Rx(mol, dt) @ mol.L

def get_quat(molecules, dt):
   for mol in molecules:
       omega_b = mol.L/I_body 
       mol.quat += dt * 0.5 * E(omega_b) @ mol.quat
       mol.quat /= np.linalg.norm(mol.quat)


def half_step_L_final(molecules, dt):
    for mol in molecules:
        mol.L = Rx(mol, dt) @ mol.L
        
        # step 3: Ry rotation
        mol.L = Ry(mol, dt) @ mol.L
        
        mol.L += 0.5 * dt * mol.T


N,L  = 1,30
dt= 0.001
molecules = initialize_molecules(N, L)
molecules[0].cm_vel = np.array([1.0, 0.0, 0.0])

n_steps = 1000
compute_forces(molecules)
molecules[0].L = np.array([0.0, 0.0, 1.0])



# compute_forces(molecules)
# compute_torques(molecules)
# for step in range(n_steps):
#     # translational first half
#     half_step_velocity(molecules, dt)
#     # rotational first half
#     half_step_L(molecules, dt)
#     # full steps
#     full_step_position(molecules, dt)
#     get_quat(molecules, dt)
#     # recompute interactions at new positions/orientations
#     compute_forces(molecules)
#     compute_torques(molecules)
#     # translational second half
#     half_step_velocity_final(molecules, dt)
#     # rotational second half
#     half_step_L_final(molecules, dt)
#     # after each step check:
#     O, H1, H2 = get_atom_positions(molecules[0])
#     print(np.linalg.norm(H1 - O))   # should always be OH_BOND = 1.0
#     print(np.linalg.norm(H2 - O))   # should always be OH_BOND = 1.0






