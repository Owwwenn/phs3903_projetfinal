import numpy as np
import quaternion as qtn
from scipy.spatial.transform import Rotation

def build_potential_vector_force_torque_matrix(n: int, r, q, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float, q_o:float, q_h:float, eps_LJ: float, sigma_LJ: float, k_coul: float):
    """Calcule les potentiels, forces, torques pour une molécule à trois site donné, avec Coulomb et LJ.

    Args:
        n (int): Nombre de molécules
        v (_type_): Matrice d'état du système 
        Lx (float): Taille de la boîte de simulation en x
        Ly (float): Taille de la boîte de simulation en y
        Lz (float): Taille de la boîte de simulation en z
        nbr_list (_type_): Liste de booléens des voisins pour chaque molécule
        theta (float): Angle H-O-H du modèle de la molécule d'eau
        r_oh (float): Distance O-H du modèle de la molécule d'eau
        q_o (float): Charge sur l'atome d'oxygène
        q_h (float): Charge sur l'atome d'hydrogène
        eps_LJ (float): epsilon de Lennard-Jones pour l'interaction O-O 
        sigma_LJ (float): sigma de Lennard-Jones pour l'interaction O-O 
        k_coul (float): 1/4pieps0 dans les goofy ahh unités de Karim  

    Returns:
        list: U, F, tau Arrays de potentiels, Forces et Torque
    """
    list_r = r
    list_q = q
    L = np.array([Lx,Ly,Lz])

    s = np.sin(theta/2)
    c = np.cos(theta/2)
 
    # Définition des positions des sites hydrogènes dans le repère de la molécule
    r_h1 = r_oh * np.array([0,s,c])  
    r_h2 = r_oh * np.array([0,-s,c])  

    i_idx, j_idx = np.where(nbr_list)

    # Définition de r et de q
    r = list_r[j_idx] - list_r[i_idx]             
    # r = ri - L * np.round(ri / L)
    q_i = qtn.from_float_array(list_q[i_idx])
    q_j = qtn.from_float_array(list_q[j_idx])

    # Définition des vecteurs dans le repère de la molécule de référence
    u_h1 = qtn.rotate_vectors(q_i, r_h1) 
    u_h2 = qtn.rotate_vectors(q_i, r_h2) 
    v_h1 = qtn.rotate_vectors(q_j, r_h1) 
    v_h2 = qtn.rotate_vectors(q_j, r_h2) 

    rOO = r
    rOH1 = r + v_h1
    rOH2 = r + v_h2
    rH1O = r - u_h1
    rH1H1 = r - u_h1 + v_h1
    rH1H2 = r - u_h1 + v_h2
    rH2O = r - u_h2
    rH2H1 = r - u_h2 + v_h1
    rH2H2 = r - u_h2 + v_h2

    r_vec = np.array([
        rOO, rOH1, rOH2, 
        rH1O, rH1H1, rH1H2, 
        rH2O, rH2H1, rH2H2
    ])

    r_vec = r_vec - L * np.round(r_vec / L)

    eps = 1e-12
    inv_r = 1.0 / np.sqrt(np.einsum("kij,kij->ki", r_vec, r_vec) + eps**2)

    # Coulomb 
    q_left  = np.array([q_o]*3 + [q_h]*6)
    q_right = np.array([q_o, q_h, q_h] * 3)
    qq      = k_coul * q_left * q_right   
    prefactor = qq[:, None] * inv_r**3

    # LJ 
    sig6  = sigma_LJ**6; sig12 = sig6**2
    prefactor[0] += 24 * eps_LJ * (
        2 * sig12 * inv_r[0]**12 - sig6 * inv_r[0]**6
    ) * inv_r[0]**2

    # Potentials
    u_pair = (
        np.einsum("k,kp->p", qq, inv_r)
        + 4 * eps_LJ * np.sum(sig12 * inv_r[0]**12 - sig6 * inv_r[0]**6)
    ) 

    U = np.zeros(n)
    np.add.at(U, i_idx, 0.5 * u_pair)
    np.add.at(U, j_idx, 0.5 * u_pair) 

    # Forces
    F_pair = prefactor[:, :, None] * r_vec        # (9, N_pairs, 3)
    F = np.zeros((n, 3))
    for k in range(9):
        np.add.at(F, i_idx,  F_pair[k])
        np.add.at(F, j_idx, -F_pair[k])

    # Torques
    def scatter_torque(lever, F_rows, molecule_idx):
        tau = np.zeros((n, 3))
        for F_k in F_rows:
            np.add.at(tau, molecule_idx, np.cross(lever, F_k))
        return tau

    tau = np.zeros((n, 3))
    tau += scatter_torque(u_h1, F_pair[3:6],         i_idx)
    tau += scatter_torque(u_h2, F_pair[6:9],         i_idx)
    tau += scatter_torque(v_h1, -F_pair[[1, 4, 7]],  j_idx)
    tau += scatter_torque(v_h2, -F_pair[[2, 5, 8]],  j_idx)

    return U, F, tau

def compute_forces_and_torques(sys, model, param, nbr_list):
    """Calcule les forces et les couples appliqués sur chaque molécule.

    Args:
        sys (MDSystem): Système moléculaire
        nbr_list (np.ndarray): Liste des voisins

    Returns:
        None: Met à jour sys.force et sys.T
    """
    Lx, Ly, Lz = param.L
    
    U, F, tau = build_potential_vector_force_torque_matrix(
        sys.N, sys.cm_pos, sys.quat, Lx, Ly, Lz, nbr_list, 
        model.HOH_rad, model.OH, model.q_o, model.q_h,
        model.eps_LJ, model.sigma_LJ, param.k_coul)
    
    sys.U = np.sum(U) / 2
    sys.force = F
    sys.T     = tau