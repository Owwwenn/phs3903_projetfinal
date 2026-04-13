import numpy as np
import quaternion as qtn
from scipy.spatial.transform import Rotation
from md_sim.core.system import mic
from md_sim.core.neighbour_list.neighbour_list import build_nl_pairs_cells

def build_potential_vector_force_torque_matrix(n: int, r, q, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, z_cm: float, r_oh:float, q_o:float, q_h:float, eps_LJ: float, sigma_LJ: float, k_coul: float, F_cap: float, tau_cap: float):
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
    r_cm = np.array([0,0,z_cm])
    r_o = r_oh * np.array([0,0,0]) - r_cm
    r_h1 = r_oh * np.array([0,s,c]) - r_cm 
    r_h2 = r_oh * np.array([0,-s,c]) - r_cm 

    # Neighbour lists
    nbr_list1, nbr_list2 = nbr_list
    i_idx, j_idx = nbr_list1
    i_LJ, j_LJ = nbr_list2

    # Définition de r et de q
    ri = list_r[j_idx] - list_r[i_idx]             
    rw = ri - L * np.round(ri / L)
    ri_LJ = list_r[j_LJ] - list_r[i_LJ]
    rw_LJ = ri_LJ - L * np.round(ri_LJ / L)
    q_i = qtn.from_float_array(list_q[i_idx])
    q_j = qtn.from_float_array(list_q[j_idx])
    q_i_LJ = qtn.from_float_array(list_q[i_LJ])
    q_j_LJ = qtn.from_float_array(list_q[j_LJ])

    # Définition des vecteurs dans le repère monde
    u_o = qtn.rotate_vectors(q_i, r_o)
    u_h1 = qtn.rotate_vectors(q_i, r_h1) 
    u_h2 = qtn.rotate_vectors(q_i, r_h2) 
    v_o = qtn.rotate_vectors(q_j, r_o)
    v_h1 = qtn.rotate_vectors(q_j, r_h1) 
    v_h2 = qtn.rotate_vectors(q_j, r_h2) 
    u_o_LJ = qtn.rotate_vectors(q_i_LJ, r_o)
    v_o_LJ = qtn.rotate_vectors(q_j_LJ, r_o)

    rOO = rw - u_o + v_o
    rOH1 = rw - u_o + v_h1
    rOH2 = rw - u_o + v_h2
    rH1O = rw - u_h1 + v_o
    rH1H1 = rw - u_h1 + v_h1
    rH1H2 = rw - u_h1 + v_h2
    rH2O = rw - u_h2 + v_o
    rH2H1 = rw - u_h2 + v_h1
    rH2H2 = rw - u_h2 + v_h2

    r_vec = np.array([
        rOO, rOH1, rOH2, 
        rH1O, rH1H1, rH1H2, 
        rH2O, rH2H1, rH2H2
    ])
    
    rw_LJ = ri_LJ - u_o_LJ + v_o_LJ
    # rw_LJ = ri_LJ - L * np.round(ri_LJ / L)
    r_vec = r_vec - L * np.round(r_vec / L)

    inv_r = 1.0 / np.sqrt(np.einsum("kij,kij->ki", r_vec, r_vec))
    inv_r_LJ = 1.0 / np.sqrt(np.einsum("ij,ij->i", rw_LJ, rw_LJ))

    # Coulomb 
    q_left  = np.array([q_o]*3 + [q_h]*6)
    q_right = np.array([q_o, q_h, q_h] * 3)
    qq      = k_coul * q_left * q_right   
    prefactor = -qq[:, None] * inv_r**3

    # # LJ 
    sig6  = sigma_LJ**6; sig12 = sig6**2
    invLJ6 = inv_r_LJ**6; invLJ12 = invLJ6**2
    lj_scalar = 24 * eps_LJ * (
        2 * sig12 * invLJ12 - sig6 * invLJ6
    ) * inv_r_LJ**2
    u_LJ = 4 * eps_LJ * sig12 * invLJ12 - sig6 * invLJ6
    F_LJ = (lj_scalar[:, None]) * rw_LJ
    
    # Potentials
    u_pair = (
        np.einsum("k,kp->p", qq, inv_r)
    ) 

    U = np.zeros(n)
    np.add.at(U, i_idx, 0.5 * u_pair)
    np.add.at(U, j_idx, 0.5 * u_pair) 
    np.add.at(U, i_LJ, 0.5 * u_LJ)
    np.add.at(U, j_LJ, 0.5 * u_LJ)   

    # Forces
    F_pair = prefactor[:, :, None] * r_vec        # (9, N_pairs, 3)
    F = np.zeros((n, 3))
    F_i = F_pair.sum(axis=0)
    np.add.at(F, i_idx, F_i)
    np.add.at(F, j_idx, -F_i)
    np.add.at(F, i_LJ,  F_LJ)
    np.add.at(F, j_LJ, -F_LJ)

    F_norm = np.linalg.norm(F, axis=1, keepdims=True)
    mask = (F_norm > F_cap).squeeze()
    F[mask] = F[mask] / F_norm[mask] * F_cap

    # Torques
    tau = np.zeros((n, 3))

    tau_i = (
        np.cross(u_o,  F_pair[0] + F_pair[1] + F_pair[2]) +
        np.cross(u_h1, F_pair[3] + F_pair[4] + F_pair[5]) +
        np.cross(u_h2, F_pair[6] + F_pair[7] + F_pair[8])
    )
    np.add.at(tau, i_idx, tau_i)

    tau_j = (
        np.cross(v_o,  -(F_pair[0] + F_pair[3] + F_pair[6])) +
        np.cross(v_h1, -(F_pair[1] + F_pair[4] + F_pair[7])) +
        np.cross(v_h2, -(F_pair[2] + F_pair[5] + F_pair[8]))
    )
    np.add.at(tau, j_idx, tau_j)
    
    tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)
    mask = (tau_norm > tau_cap).squeeze()
    tau[mask] = tau[mask] / tau_norm[mask] * tau_cap

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
        model.HOH_rad, model.z_cm, model.OH, model.q_o, model.q_h,
        model.eps_LJ, model.sigma_LJ, param.k_coul, param.F_cap, param.tau_cap)
    
    sys.U = np.sum(U)
    sys.force = F
    sys.T     = tau

if __name__ == "__main__":
    N = 10
    Lx, Ly, Lz = 15, 15, 15
    L = np.array([Lx, Ly, Lz])
    n_side = int(np.ceil(N**(1/3)))
    xs = np.linspace(-Lx/2, Lx/2, n_side, endpoint=False)
    ys = np.linspace(-Ly/2, Ly/2, n_side, endpoint=False)
    zs = np.linspace(-Lz/2, Lz/2, n_side, endpoint=False)

    r = np.array([[x, y, z] 
                        for x in xs 
                        for y in ys 
                        for z in zs])[:N]
    
    theta = np.radians(109)
    r_oh = 1
