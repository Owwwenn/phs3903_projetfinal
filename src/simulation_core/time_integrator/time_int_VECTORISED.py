
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from scipy.spatial.transform import Rotation
from simulation_core.potential_force.coul_LJ import build_potential_vector_force_torque_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# =============================================================================
# CONSTANTS
# =============================================================================
mmass = 18.015         
kB    = 0.831446        
T_init = 273   
q_o = -0.8476   
q_h = 0.4238
N  = 2
L  = 15
ex = np.array([1.0,0.0,0.0])
ey = np.array([0.0,1.0,0.0])
ez = np.array([0.0,0.0,1.0])
sigma = 3.166
epsilon = 0.1553
k_coul = 1389

# Collecter les positions à chaque step
O_traj  = []
H1_traj = []
H2_traj = []
skip = 10

# SPC/E geometry in body frame
OH_BOND   =  1   
HOH_ANGLE = 109.47      
O_body    = np.array([0.0, 0.0, 0.0])
angle_rad = np.radians(HOH_ANGLE / 2)
H1_body   = np.array([ np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND
H2_body   = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * OH_BOND

# Principal moments of inertia for SPC/E water
I_body = np.array([1.3743, 1.9144, 0.6001])
i1, i2, i3 = I_body


# =============================================================================
# SYSTEM CLASS
# =============================================================================
class MDSystem:
    """Classe représentant un système de dynamique moléculaire de molécules rigides.

    Attributs:
        N (int): Nombre de molécules
        cm_pos (np.ndarray): Positions des centres de masse (N, 3)
        cm_vel (np.ndarray): Vitesses des centres de masse (N, 3)
        force (np.ndarray): Forces appliquées sur les centres de masse (N, 3)
        L (np.ndarray): Moments cinétiques angulaires dans ref lab(N, 3)
        T (np.ndarray): Couples (torques) appliqués dans ref lab (N, 3)
        quat (np.ndarray): Quaternions d’orientation (N, 4)
        r_last (np.ndarray): Positions lors du dernier rebuild de neighbour list
        neighbor_list: Liste des voisins, matrice de 0 et de 1 
        neighbor_count: Nombre de voisins
        size (np.ndarray): Taille de la boîte de simulation
    """
    def __init__(self, N):
        self.N       = N
        self.cm_pos  = np.zeros((N, 3))   # 
        self.cm_vel  = np.zeros((N, 3))   # 
        self.force   = np.zeros((N, 3))   # 
        self.L       = np.zeros((N, 3))   #
        self.T       = np.zeros((N, 3))   # 
        self.quat    = np.zeros((N, 4))   # 
        self.quat[:, 0] = 1.0             # 
        self.r_last = np.zeros((N, 3))
        self.neighbor_list = None
        self.neighbor_count = None
        self.size = np.zeros(3)
        self.U = 0.0

# =============================================================================
# INITIALIZATION
# =============================================================================
def initialize_system(N, L):
    """Initialise un système de N molécules dans une boîte cubique.

    Args:
        N (int): Nombre de molécules
        L (float): Taille de la boîte cubique

    Returns:
        MDSystem: Système initialisé avec positions, vitesses et moments angulaires
    """

    sys = MDSystem(N)

    # positions on cubic grid
    n_side = int(np.ceil(N**(1/3)))
    # spacing = min(L / (n_side + 1), L / 2)  # espacement entre molécules
    spacing = 4
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                positions.append([
                    (i + 1) * spacing,
                    (j + 1) * spacing,
                    (k + 1) * spacing
                ])
    sys.cm_pos = np.array(positions[:N])

    # random velocities from Maxwell-Boltzmann
    std_dev = np.sqrt(kB * T_init / mmass)
    sys.cm_vel = np.random.normal(0, std_dev, size=(N, 3))

    # remove center of mass drift
    sys.cm_vel -= sys.cm_vel.mean(axis=0)

    # Sample angular momentum in BODY FRAME
    Ix = i1
    Iy = i2
    Iz = i3

    # Sample angular momentum in BODY FRAME
    std_Lx = np.sqrt(Ix * kB * T_init)
    std_Ly = np.sqrt(Iy * kB * T_init)
    std_Lz = np.sqrt(Iz * kB * T_init)

    Lx = np.random.normal(0, std_Lx, size=N)
    Ly = np.random.normal(0, std_Ly, size=N)
    Lz = np.random.normal(0, std_Lz, size=N)

    sys.L[:, 0] = Lx
    sys.L[:, 1] = Ly
    sys.L[:, 2] = Lz

    # -----------------------
    # Remove total angular momentum (optional but recommended)
    # -----------------------
    L_total = sys.L.sum(axis=0)
    sys.L -= L_total / N


    return sys


# =============================================================================
# GEOMETRY
# =============================================================================
def get_atom_positions(sys):

    """Calcule les positions des atomes (O, H1, H2) dans le repère laboratoire.

    Args:
        sys (MDSystem): Système moléculaire

    Returns:
         Positions des atomes O, H1 et H2
    """

    O_lab  = np.zeros((sys.N, 3))
    H1_lab = np.zeros((sys.N, 3))
    H2_lab = np.zeros((sys.N, 3))

    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    O_lab  = sys.cm_pos + R.apply(O_body)
    H1_lab = sys.cm_pos + R.apply(H1_body)
    H2_lab = sys.cm_pos + R.apply(H2_body)

    return O_lab, H1_lab, H2_lab

def mic(dr, L):
       """Applique la convention d'image minimale (Minimum Image Convention).

    Args:
        dr (np.ndarray): Vecteur de distance
        L (float): Taille de la boîte

    Returns:
        np.ndarray: Distance corrigée avec conditions périodiques
    """
       return dr - L * np.round(dr / L)

def wrap_positions(pos, L):
    """Ramène les positions dans la boîte de simulation (conditions périodiques).

    Args:
        pos (np.ndarray): Positions
        L (float): Taille de la boîte

    Returns:
        np.ndarray: Positions corrigé dans la boîte
    """
    return pos - L * np.floor(pos / L)
    # return pos - L * np.round(pos / L)

# =============================================================================
# NEIGHBOUR LIST
# =============================================================================
def build_nl(sys, r_c, r_s, L):
    """Construit la liste des voisins pour chaque molécule.

    Args:
        sys (MDSystem): Système moléculaire
        r_c (float): Rayon de coupure
        r_s (float): Distance de skin (buffer pour ne pas calculer la nl a chaque etape)
        L (float): Taille de la boîte

    Returns:
        np.ndarray: Matrice de voisin (N, N) binaire
    """
    rt = r_c + r_s
    cm_pos = sys.cm_pos
    N = len(cm_pos)
    Nmax = int(np.ceil(1.2 * 418.9 * rt**3))
    nl = np.full((N, Nmax), -1, dtype=int)  
    counts = np.zeros(N, dtype=int)
    nbr_list = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            dr = cm_pos[i] - cm_pos[j]
            dr = mic(dr, L)

            if np.dot(dr, dr) < rt**2:
                
                nl[i, counts[i]] = j
                counts[i] += 1

                
                nl[j, counts[j]] = i
                counts[j] += 1
        
    for i, neighbors in enumerate(nl):
        neighbors = [j for j in neighbors if j >= 0]
        nbr_list[i, neighbors] = 1

        
    return nbr_list



def def_rebuild(sys, L, skin):
     
     """Détermine si la neighbour list doit être reconstruite.

    Args:
        sys (MDSystem): Système moléculaire
        L (float): Taille de la boîte
        skin (float): Distance de peau

    Returns:
        bool: True si reconstruction nécessaire
    """

     disp = sys.cm_pos - sys.r_last
     disp = mic(disp, L)
     max_disp = np.max(np.linalg.norm(disp, axis = 1))


     return max_disp > (skin / 2)




# =============================================================================
# FORCE AND TORQUE
# =============================================================================
def compute_forces_and_torques(sys, nbr_list):
    """Calcule les forces et les couples appliqués sur chaque molécule.

    Args:
        sys (MDSystem): Système moléculaire
        nbr_list (np.ndarray): Liste des voisins

    Returns:
        None: Met à jour sys.force et sys.T
    """
    v = np.concatenate([
        sys.cm_pos.flatten(),
        sys.cm_vel.flatten(),
        sys.quat.flatten(),
        sys.L.flatten()
    ])

    U, F, tau = build_potential_vector_force_torque_matrix(
        sys.N, v, L, L, L, nbr_list,
        np.radians(HOH_ANGLE), OH_BOND, q_o, q_h, epsilon, sigma, k_coul)
    
    #print(f"F_raw sample = {F[:2]}")

    sys.U = np.sum(U) / 2
    # R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    sys.force = F
    sys.T     = tau 
 
# =============================================================================
# TRANSLATIONAL INTEGRATOR
# =============================================================================
def half_step_velocity(sys, dt):
    """Effectue un demi-pas de vitesse (Velocity Verlet).

    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps
    """
    sys.cm_vel += 0.5 * (sys.force / mmass) * dt

def full_step_position(sys, dt):
    """Met à jour les positions sur un pas complet.

    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps
    """
    sys.cm_pos += sys.cm_vel * dt
    

def half_step_velocity_final(sys, dt):
    """Effectue le second demi-pas de vitesse.

    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps
    """ 
    sys.cm_vel += 0.5 * (sys.force / mmass) * dt

# =============================================================================
# ROTATIONAL INTEGRATOR
# =============================================================================
def half_step_L(sys, dt):
    """Effectue un demi-pas sur le moment cinétique angulaire.

    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps
    """
    #  Lab a body frame torque
    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    T_body = R.inv().apply(sys.T)

    #  Half-step torque update
    sys.L += 0.5 * dt * T_body

    # Ry rotation
    Lx, Ly, Lz = sys.L.T

    alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
    c = np.cos(alpha)
    s = np.sin(alpha)

    Lx_new =  c * Lx + s * Lz
    Ly_new =  Ly
    Lz_new = -s * Lx + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    #  Rx rotation
    Lx, Ly, Lz = sys.L.T

    beta = (dt / 2) * (1/i3 - 1/i1) * Lx
    c = np.cos(beta)
    s = np.sin(beta)

    Lx_new = Lx
    Ly_new =  c * Ly - s * Lz
    Lz_new =  s * Ly + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)


def quat_mul(q, r):
    """Multiplie deux quaternions.

    Args:
        q (np.ndarray): Quaternion (N, 4)
        r (np.ndarray): Quaternion (N, 4)

    Returns:
        np.ndarray: Produit quaternion (N, 4)
    """
    w1, x1, y1, z1 = q.T
    w2, x2, y2, z2 = r.T

    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=1)


    
def axis_angle_to_quat(axis, angle):
     """Convertit une rotation angulaire en quaternion.

    Args:
        axis (np.ndarray): Axe de rotation (3,)
        angle (np.ndarray): Angle de rotation (N,)

    Returns:
        np.ndarray: Quaternion correspondant (N, 4)
    """
     half = 0.5 * angle
     s = np.sin(half)
     c = np.cos(half)

     return np.stack([
        c,
        axis[0]*s,
        axis[1]*s,
        axis[2]*s
    ], axis=1)

def get_quat(sys, dt):
    """Met à jour les quaternions via l'intégration des rotations.
    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps

    Returns:
        None: Met à jour sys.quat
    """
    q = sys.quat  # (N, 4)
    omega = sys.L / I_body  # (N, 3)

  
    # rotations élémentaires 
    qy1 = axis_angle_to_quat(ey, omega[:,1] * dt/2)
    qx1 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qz  = axis_angle_to_quat(ez, omega[:,2] * dt)
    qx2 = axis_angle_to_quat(ex, omega[:,0] * dt/2)
    qy2 = axis_angle_to_quat(ey, omega[:,1] * dt/2)

    # application 
    q = quat_mul(qy1, q)
    q = quat_mul(qx1, q)
    q = quat_mul(qz,  q)
    q = quat_mul(qx2, q)
    q = quat_mul(qy2, q)

    # # normalisation 
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    sys.quat = q
            

def half_step_L_final(sys, dt):

    """Effectue le second demi-pas du moment cinétique angulaire.

    Args:
        sys (MDSystem): Système
        dt (float): Pas de temps
    """
    #Lab a body frame torque 
    R = Rotation.from_quat(sys.quat[:, [1,2,3,0]])
    T_body = R.inv().apply(sys.T)

    # Rx rotation
    Lx, Ly, Lz = sys.L.T

    beta = (dt / 2) * (1/i3 - 1/i1) * Lx
    c = np.cos(beta)
    s = np.sin(beta)

    Lx_new = Lx
    Ly_new =  c * Ly - s * Lz
    Lz_new =  s * Ly + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    # Ry rotation
    Lx, Ly, Lz = sys.L.T

    alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
    c = np.cos(alpha)
    s = np.sin(alpha)

    Lx_new =  c * Lx + s * Lz
    Ly_new =  Ly
    Lz_new = -s * Lx + c * Lz

    sys.L = np.stack((Lx_new, Ly_new, Lz_new), axis=1)

    # --- 4. Second half-step torque ---
    sys.L += 0.5 * dt * T_body

# =============================================================================
# ENERGY
# =============================================================================
def kinetic_energy(sys):
    return 0.5 * mmass * np.sum(sys.cm_vel**2)

def rotational_energy(sys):
    return 0.5 * np.sum((sys.L**2) / I_body)


    

# =============================================================================
# MAIN
# =============================================================================




dt = 0.0003
n_steps = 30000
s = np.zeros(n_steps)
en = np.zeros(n_steps)
sys = initialize_system(N, L)
sys.cm_pos = wrap_positions(sys.cm_pos, L)
sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)

dist_OO   = np.zeros(n_steps)
dist_OH1  = np.zeros(n_steps)
dist_OH2  = np.zeros(n_steps)
dist_H1H2 = np.zeros(n_steps)
dist_H1O  = np.zeros(n_steps)
dist_H2O  = np.zeros(n_steps)
dist_H1H1 = np.zeros(n_steps)
dist_H2H2 = np.zeros(n_steps)
dist_H1H2b= np.zeros(n_steps)

pos_init   = sys.cm_pos.copy()
L_init     = sys.L.sum(axis=0).copy()
U_arr = np.zeros(n_steps)

r_cut = 10.0   # cut-off radius (Å)
skin  = 4.0    # skin pour neighbour list
sys.r_last = sys.cm_pos.copy()
nbr_list = np.ones((N,N))- np.eye(N) #build_nl(sys, r_cut, skin, L)  # initial neighbour list

F_norms = np.zeros(n_steps)
T_norms = np.zeros(n_steps)

compute_forces_and_torques(sys, nbr_list)
E_init     = kinetic_energy(sys) + rotational_energy(sys) + sys.U


for step in range(n_steps):
    #if def_rebuild(sys, L, skin):
       # nbr_list = build_nl(sys, r_cut, skin, L)
        #sys.r_last = sys.cm_pos.copy()


    half_step_velocity(sys, dt)
    half_step_L(sys, dt)
    get_quat(sys, dt)
    full_step_position(sys, dt)
    sys.cm_pos = wrap_positions(sys.cm_pos, L)
    compute_forces_and_torques(sys, nbr_list)
    half_step_velocity_final(sys, dt)
    half_step_L_final(sys, dt)


    # === DIAGNOSTICS ===
    E     = kinetic_energy(sys) + rotational_energy(sys) + sys.U
    T     = 2 * kinetic_energy(sys) / (3 * N * kB)
    L_tot = sys.L.sum(axis=0)
    qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
    s[step] = step
    en[step] = E 
    U_arr[step] = sys.U
    F_norms[step] = np.linalg.norm(sys.force)
    T_norms[step] = np.linalg.norm(sys.T)
    O, H1, H2 = get_atom_positions(sys)

    # molécule 0 vs molécule 1
    dist_OO[step]   = np.linalg.norm(mic(O[0]  - O[1],  L))
    # dist_OH1[step]  = np.linalg.norm(mic(O[0]  - H1[1], L))
    # dist_OH2[step]  = np.linalg.norm(mic(O[0]  - H2[1], L))
    # dist_H1O[step]  = np.linalg.norm(mic(H1[0] - O[1],  L))
    # dist_H2O[step]  = np.linalg.norm(mic(H2[0] - O[1],  L))
    # dist_H1H1[step] = np.linalg.norm(mic(H1[0] - H1[1], L))
    # dist_H1H2[step] = np.linalg.norm(mic(H1[0] - H2[1], L))
    # dist_H1H2b[step] = np.linalg.norm(mic(H2[0] - H1[1], L))
    # dist_H2H2[step] = np.linalg.norm(mic(H2[0] - H2[1], L))


    print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
          f"T={T:.1f}K | |L_drift|={np.linalg.norm(L_tot-L_init):.2e} | "
          f"qnorm_err={qnorm:.2e}| sum T = {sys.T.sum(axis=0)}")
    
    if step % skip == 0:
        O, H1, H2 = get_atom_positions(sys)
        O_traj.append(O.copy())
        H1_traj.append(H1.copy())
        H2_traj.append(H2.copy())


fig_anim = plt.figure(figsize=(7, 7))
ax = fig_anim.add_subplot(111, projection='3d')

scat_O  = ax.scatter([], [], [], c='red',   s=80,  label='O')
scat_H1 = ax.scatter([], [], [], c='white', s=40,  edgecolors='gray', label='H')
scat_H2 = ax.scatter([], [], [], c='white', s=40,  edgecolors='gray')

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_facecolor('black')
fig_anim.patch.set_facecolor('black')
title = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)


def update(frame):
    O  = O_traj[frame]
    H1 = H1_traj[frame]
    H2 = H2_traj[frame]
    scat_O._offsets3d  = (O[:,0],  O[:,1],  O[:,2])
    scat_H1._offsets3d = (H1[:,0], H1[:,1], H1[:,2])
    scat_H2._offsets3d = (H2[:,0], H2[:,1], H2[:,2])
    title.set_text(f'frame {frame} / step {frame * skip}')
    return scat_O, scat_H1, scat_H2

ani = animation.FuncAnimation(fig_anim, update, frames=len(O_traj), interval=50, blit=False)
plt.tight_layout()
plt.show()



print(max(abs(en - E_init*np.ones(len(en)))/E_init*100))


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(F_norms)
plt.title('||F||')
plt.xlabel('step')

plt.subplot(1, 4, 2)
plt.plot(T_norms)
plt.title('||T||')
plt.xlabel('step')

plt.subplot(1, 4, 3)
plt.plot(en)
plt.title('E_tot')
plt.xlabel('step')

plt.subplot(1, 4, 4)
plt.plot(U_arr)
plt.title('U')
plt.xlabel('step')

plt.tight_layout()


plt.figure(figsize=(14, 4))
plt.plot(dist_OO,   label='O-O')
#plt.plot(dist_OH1,  label='O-H1')
#plt.plot(dist_OH2,  label='O-H2')
#plt.plot(dist_H1O,  label='H1-O')
#plt.plot(dist_H2O,  label='H2-O')
#plt.plot(dist_H1H1, label='H1-H1')
#plt.plot(dist_H1H2, label='H1-H2')
#plt.plot(dist_H1H2b, label='H2-H1')
#plt.plot(dist_H2H2, label='H2-H2')
plt.xlabel('step')
plt.ylabel('distance (Å)')
plt.title('distances inter-moléculaires')
plt.legend()
plt.tight_layout()
plt.show()

"""
TODO:
- Inclure l'énergie potentiel pour mieux diagnostiquer le energy drift
- Optimsier le calcul de la neighbour list
- faire un graph du temps de simulation en fonction du nombre de molécule pour vérifier que la neighbour list transforme t(N^2) en t(N)
- Implementer nose hoover
- Tester avec potentiel Lennard Jones
"""

#######################################################################################################################################################################################################################################################################################

# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# import numpy as np
# from scipy.spatial.transform import Rotation
# from simulation_core.potential_force.potentiel_force import build_potential_vector_force_torque_matrix
# import matplotlib.pyplot as plt


# # =============================================================================
# # CONSTANTS
# # =============================================================================

# mmass = 18.015  # g/mol
# kB = 0.831446   # (g/mol)(Å/ps)²/K
# T_init = 273    # K

# q_o = -0.8476
# q_h = 0.4238

# N = 15
# L = 6.2

# ex = np.array([1.0, 0.0, 0.0])
# ey = np.array([0.0, 1.0, 0.0])
# ez = np.array([0.0, 0.0, 1.0])

# # SPC/E geometry in body frame
# OH_BOND = 1
# HOH_ANGLE = 109.47

# O_body = np.array([0.0, 0.0, 0.0])
# angle_rad = np.radians(HOH_ANGLE / 2)

# H1_body = np.array([0.0, np.sin(angle_rad), np.cos(angle_rad)]) * OH_BOND
# H2_body = np.array([0.0, -np.sin(angle_rad), np.cos(angle_rad)]) * OH_BOND

# # Principal moments of inertia
# I_body = np.array([1.3743, 1.9144, 0.6001])
# i1, i2, i3 = I_body

# # =============================================================================
# # SYSTEM CLASS
# # =============================================================================

# class MDSystem:
#     def __init__(self, N):
#         self.N = N

#         self.cm_pos = np.zeros((N, 3))
#         self.cm_vel = np.zeros((N, 3))
#         self.force = np.zeros((N, 3))
#         self.U = 0.0

#         self.L = np.zeros((N, 3))
#         self.T = np.zeros((N, 3))

#         self.quat = np.zeros((N, 4))
#         self.quat[:, 0] = 1.0

#         self.r_last = np.zeros((N, 3))
#         self.neighbor_list = None
#         self.neighbor_count = None
#         self.size = np.zeros(3)

# # =============================================================================
# # INITIALIZATION
# # =============================================================================

# def initialize_system(N, L):
#     sys = MDSystem(N)

#     # positions
#     n_side = int(np.ceil(N**(1/3)))
#     d = L / n_side

#     positions = []
#     for i in range(n_side):
#         for j in range(n_side):
#             for k in range(n_side):
#                 positions.append([i*d, j*d, k*d])

#     sys.cm_pos = np.array(positions[:N])

#     # velocities
#     std_dev = np.sqrt(kB * T_init / mmass)
#     sys.cm_vel = np.random.normal(0, std_dev, size=(N, 3))
#     sys.cm_vel -= sys.cm_vel.mean(axis=0)

#     # angular momentum
#     std_Lx = np.sqrt(i1 * kB * T_init)
#     std_Ly = np.sqrt(i2 * kB * T_init)
#     std_Lz = np.sqrt(i3 * kB * T_init)

#     sys.L[:, 0] = np.random.normal(0, std_Lx, size=N)
#     sys.L[:, 1] = np.random.normal(0, std_Ly, size=N)
#     sys.L[:, 2] = np.random.normal(0, std_Lz, size=N)

#     # remove total angular momentum
#     sys.L -= sys.L.sum(axis=0) / N

#     return sys

# # =============================================================================
# # GEOMETRY
# # =============================================================================

# def get_atom_positions(sys):
#     R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])

#     O_lab  = sys.cm_pos + R.apply(O_body)
#     H1_lab = sys.cm_pos + R.apply(H1_body)
#     H2_lab = sys.cm_pos + R.apply(H2_body)

#     return O_lab, H1_lab, H2_lab


# def mic(dr, L):
#     return dr - L * np.round(dr / L)


# def wrap_positions(pos, L):
#     return pos - L * np.floor(pos / L)

# # =============================================================================
# # NEIGHBOUR LIST
# # =============================================================================

# def build_nl(sys, r_c, r_s, L):
#     rt = r_c + r_s
#     cm_pos = sys.cm_pos
#     N = len(cm_pos)

#     Nmax = int(np.ceil(1.2 * 418.9 * rt**3))

#     nl = np.full((N, Nmax), -1, dtype=int)
#     counts = np.zeros(N, dtype=int)
#     nbr_list = np.zeros((N, N))

#     for i in range(N):
#         for j in range(i + 1, N):
#             dr = mic(cm_pos[i] - cm_pos[j], L)

#             if np.dot(dr, dr) < rt**2:
#                 nl[i, counts[i]] = j
#                 counts[i] += 1

#                 nl[j, counts[j]] = i
#                 counts[j] += 1

#     for i, neighbors in enumerate(nl):
#         neighbors = [j for j in neighbors if j >= 0]
#         nbr_list[i, neighbors] = 1

#     return nbr_list


# def def_rebuild(sys, L, skin):
#     disp = mic(sys.cm_pos - sys.r_last, L)
#     max_disp = np.max(np.linalg.norm(disp, axis=1))
#     return max_disp > (skin / 2)



# # =============================================================================
# # FORCE AND TORQUE
# # =============================================================================

# def compute_forces_and_torques(sys):
#     v = np.concatenate([
#         sys.cm_pos.flatten(),
#         sys.cm_vel.flatten(),
#         sys.quat.flatten(),
#         sys.L.flatten()
#     ])

#     #print("nbr_list sum:", np.sum(nbr_list))
#     #print("cm_pos sample:", sys.cm_pos[:2])
#     #print("quat sample:", sys.quat[:2])

#     U, F = build_potential_vector_force_torque_matrix(
#         sys.N, v, L, L, L, nbr_list,
#         np.radians(HOH_ANGLE), OH_BOND, q_o, q_h, 0, 0)
    
#     #print("U brut:", U[:3])
#     #print("U sum:", np.sum(U)/2)

#     sys.U = np.sum(U) / 2
    
#     R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])

#     sys.force = R.apply(F[:, :3]) # body --- lab
#     sys.T     = R.apply(F[:, 3:6]) # body --- lab

# # =============================================================================
# # TRANSLATIONAL INTEGRATOR
# # =============================================================================

# def half_step_velocity(sys, dt):
#     sys.cm_vel += 0.5 * (sys.force / mmass) * dt


# def full_step_position(sys, dt):
#     sys.cm_pos += sys.cm_vel * dt


# def half_step_velocity_final(sys, dt):
#     sys.cm_vel += 0.5 * (sys.force / mmass) * dt

# # =============================================================================
# # ROTATIONAL INTEGRATOR
# # =============================================================================


# _rotation_angles = None

# def half_step_L(sys, dt):
#     R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])
#     T_body = R.inv().apply(sys.T)

#     sys.L += 0.5 * dt * T_body

#     Lx, Ly, Lz = sys.L.T

#     alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
#     c, s = np.cos(alpha), np.sin(alpha)

#     Lx, Ly, Lz = c*Lx + s*Lz, Ly, -s*Lx + c*Lz

#     beta = (dt / 2) * (1/i3 - 1/i1) * Lx
#     c, s = np.cos(beta), np.sin(beta)

#     sys.L = np.stack([
#         Lx,
#         c*Ly - s*Lz,
#         s*Ly + c*Lz
#     ], axis=1)
#     global _rotation_angles
    
#     # R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])
#     # T_body = R.inv().apply(sys.T)
#     # sys.L += 0.5 * dt * T_body

#     # Lx, Ly, Lz = sys.L.T

#     # # Ry(dt/2)
#     # alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
#     # c, s = np.cos(alpha), np.sin(alpha)
#     # Lx, Lz = c*Lx + s*Lz, -s*Lx + c*Lz

#     # # Rx(dt/2)
#     # beta = (dt / 2) * (1/i3 - 1/i1) * Lx
#     # c, s = np.cos(beta), np.sin(beta)
#     # Ly, Lz = c*Ly - s*Lz, s*Ly + c*Lz

#     # _rotation_angles = (alpha, beta)
#     # sys.L = np.stack([Lx, Ly, Lz], axis=1)


# def quat_mul(q, r):
#     w1, x1, y1, z1 = q.T
#     w2, x2, y2, z2 = r.T

#     return np.stack([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ], axis=1)


# def axis_angle_to_quat(axis, angle):
#     half = 0.5 * angle
#     return np.stack([
#         np.cos(half),
#         axis[0]*np.sin(half),
#         axis[1]*np.sin(half),
#         axis[2]*np.sin(half)
#     ], axis=1)


# def get_quat(sys, dt):
#     q = sys.quat
#     omega = sys.L / I_body

#     q = quat_mul(axis_angle_to_quat(ey, omega[:,1]*dt/2), q)
#     q = quat_mul(axis_angle_to_quat(ex, omega[:,0]*dt/2), q)
#     q = quat_mul(axis_angle_to_quat(ez, omega[:,2]*dt),   q)
#     q = quat_mul(axis_angle_to_quat(ex, omega[:,0]*dt/2), q)
#     q = quat_mul(axis_angle_to_quat(ey, omega[:,1]*dt/2), q)

#     q /= np.linalg.norm(q, axis=1, keepdims=True)
#     sys.quat = q


# def half_step_L_final(sys, dt):
#     R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])
#     T_body = R.inv().apply(sys.T)

#     Lx, Ly, Lz = sys.L.T

#     beta = (dt / 2) * (1/i3 - 1/i1) * Lx
#     c, s = np.cos(beta), np.sin(beta)

#     Lx, Ly, Lz = Lx, c*Ly - s*Lz, s*Ly + c*Lz

#     alpha = (dt / 2) * (1/i3 - 1/i2) * Ly
#     c, s = np.cos(alpha), np.sin(alpha)

#     sys.L = np.stack([
#         c*Lx + s*Lz,
#         Ly,
#         -s*Lx + c*Lz
#     ], axis=1)

#     sys.L += 0.5 * dt * T_body

#     # global _rotation_angles
#     # alpha, beta = _rotation_angles

#     # Lx, Ly, Lz = sys.L.T

#     # # Rx(-dt/2) — inverse exact de Rx(dt/2)
#     # c, s = np.cos(beta), np.sin(beta)
#     # Ly, Lz = c*Ly + s*Lz, -s*Ly + c*Lz

#     # # Ry(-dt/2) — inverse exact de Ry(dt/2)
#     # c, s = np.cos(alpha), np.sin(alpha)
#     # Lx, Lz = c*Lx - s*Lz, s*Lx + c*Lz

#     # sys.L = np.stack([Lx, Ly, Lz], axis=1)

#     # R = Rotation.from_quat(sys.quat[:, [1, 2, 3, 0]])
#     # T_body = R.inv().apply(sys.T)
#     # sys.L += 0.5 * dt * T_body

# # =============================================================================
# # ENERGY
# # =============================================================================

# def kinetic_energy(sys):
#     return 0.5 * mmass * np.sum(sys.cm_vel**2)


# def rotational_energy(sys):
#     return 0.5 * np.sum((sys.L**2) / I_body)

# =============================================================================
# MAIN
# =============================================================================




# dt = 0.0003
# n_steps = 100000
# s = np.zeros(n_steps)
# en = np.zeros(n_steps)
# sys = initialize_system(N, L)
# sys.cm_pos = wrap_positions(sys.cm_pos, L)
# sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)

# pos_init   = sys.cm_pos.copy()
# E_init     = kinetic_energy(sys) + rotational_energy(sys)
# L_init     = sys.L.sum(axis=0).copy()

# r_cut = 10.0   # cut-off radius (Å)
# skin  = 4.0    # skin pour neighbour list
# sys.r_last = sys.cm_pos.copy()
# nbr_list = build_nl(sys, r_cut, skin, L)  # initial neighbour list

# compute_forces_and_torques(sys, nbr_list)

# for step in range(n_steps):
#     if def_rebuild(sys, L, skin):
#         nbr_list = build_nl(sys, r_cut, skin, L)
#         sys.r_last = sys.cm_pos.copy()


#     half_step_velocity(sys, dt)
#     half_step_L(sys, dt)
#     get_quat(sys, dt)
#     full_step_position(sys, dt)
#     sys.cm_pos = wrap_positions(sys.cm_pos, L)
#     compute_forces_and_torques(sys, nbr_list)
#     half_step_velocity_final(sys, dt)
#     half_step_L_final(sys, dt)


#     # === DIAGNOSTICS ===
#     E     = kinetic_energy(sys) + rotational_energy(sys)
#     T     = 2 * kinetic_energy(sys) / (3 * N * kB)
#     L_tot = sys.L.sum(axis=0)
#     qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
#     s[step] = step
#     en[step] = E
#     print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
#           f"T={T:.1f}K | |L_drift|={np.linalg.norm(L_tot-L_init):.2e} | "
#           f"qnorm_err={qnorm:.2e}")


# print(max(abs(en - E_init*np.ones(len(en)))/E_init*100))
# plt.plot(s, en)
# plt.show()

# dt = 0.0003
# n_steps = 5

# s  = np.zeros(n_steps)
# en = np.zeros(n_steps)

# sys = initialize_system(N, L)
# sys.cm_pos = wrap_positions(sys.cm_pos, L)
# sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)


# L_init = sys.L.sum(axis=0).copy()

# # r_cut = 10.0   # cut-off radius (Å)
# # skin  = 2.0    # skin pour neighbour list
# # sys.r_last = sys.cm_pos.copy()
# nbr_list = np.ones((N,N))- np.eye(N)

# compute_forces_and_torques(sys)
# E_init = kinetic_energy(sys) + rotational_energy(sys) + sys.U

# for step in range(n_steps):

#     #if def_rebuild(sys, L, skin):
#         #nbr_list = build_nl(sys, r_cut, skin, L)
#         #sys.r_last = sys.cm_pos.copy()

#     half_step_velocity(sys, dt)
#     half_step_L(sys, dt)

#     get_quat(sys, dt)

#     full_step_position(sys, dt)
#     sys.cm_pos = wrap_positions(sys.cm_pos, L)

#     compute_forces_and_torques(sys)

#     half_step_velocity_final(sys, dt)
#     half_step_L_final(sys, dt)
#     Ekin = kinetic_energy(sys)
#     Erot = rotational_energy(sys)
#     E = kinetic_energy(sys) + rotational_energy(sys) + sys.U
#     T = 2 * kinetic_energy(sys) / (3 * N * kB)
 
#     L_tot = sys.L.sum(axis=0)
#     qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
    
#     s[step]  = step
#     en[step] = E

#     print(
#         f"step {step:4d} | E={E:.4f} "
#         f"dE={abs(E-E_init)/E_init*100:.4f}% | "
#         f"T={T:.1f}K | "
#         f"|L_drift|={np.linalg.norm(L_tot-L_init):.2e} | "
#         f"qnorm_err={qnorm:.2e}"
#         f"step {step:6d} | Ekin={Ekin:.2f} | Erot={Erot:.2f} | U={sys.U:.2f} | Etot={E:.2f}"
        
#     )
#     print("F sum:", sys.force.sum(axis=0))  # devrait être ~0
#     print("T sum:", sys.T.sum(axis=0))      # devrait être ~0
