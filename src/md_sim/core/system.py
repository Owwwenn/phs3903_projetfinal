import numpy as np
from scipy.spatial.transform import Rotation

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
        self.eta = 0.0

# Peut-être avoir un fichier pour différentes initial conditions 
def initialize_system(model, parameters):
    """Initialise un système de N molécules dans une boîte cubique.

    Args:
        model (class): Classe du modèle choisi (3 sites)
        parameters (class): Classe des "paramètres" de la simu

    Returns:
        MDSystem: Système initialisé avec positions, vitesses et moments angulaires
    """
    N = parameters.N
    L = parameters.L
    T_init = parameters.T_init
    kB = parameters.kB
    mmass = model.mass
    Ix, Iy, Iz, = model.I_body
    Lx, Ly, Lz = L
    sys = MDSystem(N)

    # positions on cubic grid
    # n_side = int(np.ceil(N**(1/3)))
    # # spacing = min(L / (n_side + 1), L / 2)  # espacement entre molécules
    # spacing = 4
    # positions = []
    # for i in range(n_side):
    #     for j in range(n_side):
    #         for k in range(n_side):
    #             positions.append([
    #                 (i + 1) * spacing,
    #                 (j + 1) * spacing,
    #                 (k + 1) * spacing
    #             ])
    n_side = int(np.ceil(N**(1/3)))
    xs = np.linspace(-Lx/2, Lx/2, n_side, endpoint=False)
    ys = np.linspace(-Ly/2, Ly/2, n_side, endpoint=False)
    zs = np.linspace(-Lz/2, Lz/2, n_side, endpoint=False)

    sys.cm_pos = np.array([[x, y, z] 
                        for x in xs 
                        for y in ys 
                        for z in zs])[:N]

    # random velocities from Maxwell-Boltzmann
    std_dev = np.sqrt(kB * T_init / mmass)
    sys.cm_vel = np.random.normal(0, std_dev, size=(N, 3))

    # remove center of mass drift
    sys.cm_vel -= sys.cm_vel.mean(axis=0)

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

    L_total = sys.L.sum(axis=0)
    sys.L -= L_total / N

    return sys

def mic(dr, L):
    """Applique la convention d'image minimale (Minimum Image Convention).

    Args:
        dr (np.ndarray): Vecteur de distance
        L (array): dimensions de la boîte

    Returns:
        np.ndarray: Distance corrigée avec conditions périodiques
    """    
    return dr - L * np.round(dr / L)

def wrap_positions(pos, L):
    """Ramène les positions dans la boîte de simulation (conditions périodiques).

    Args:
        pos (np.ndarray): Positions
        L (array): dimensions de la boîte

    Returns:
        np.ndarray: Positions corrigé dans la boîte
    """
    return pos - L * np.floor(pos / L)

def get_atom_positions(sys, model):

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
    O_lab  = sys.cm_pos + R.apply(model.O_body)
    H1_lab = sys.cm_pos + R.apply(model.H1_body)
    H2_lab = sys.cm_pos + R.apply(model.H2_body)

    return O_lab, H1_lab, H2_lab