from md_sim.core.potential_force.norms import build_inv_norm_matrix
import numpy as np

def build_potential_vector_force_torque_matrix(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float, q_o:float, q_h:float, eps_LJ: float, sigma_LJ: float):
    """Construit le vecteur de potentiel et le vecteur de force et de torque appliqué sur chacune des n molécules du système. 

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

    Returns:
        numpy.array: Vecteur de taille n contenant le potentiel appliqué sur chaque molécule. Et la force appliquée sur chaque molécule selon (x, y, z, tau_x, tau_y, tau_z) dans le repère de la molécule.
    """    
    [inv_norm_matrix_list, deriv_matrices]= build_inv_norm_matrix(n, v, Lx, Ly, Lz, nbr_list, theta, r_oh)
    inv_norm_matrix_list = np.array(inv_norm_matrix_list)
    deriv_matrices = np.array(deriv_matrices)
    left_charge_vector_list = np.array([q_o*np.ones(n)]*3 + [q_h*np.ones(n)]*6)
    right_charge_vector_list = np.array(([q_o*np.ones(n)] + [q_h*np.ones(n)]*2) * 3)

    # Potentiel et forces pour Coulomb
    k_coul = 1389
    inml2 = (inv_norm_matrix_list**2)[..., None]
    U_coulomb = k_coul * np.einsum("ki,kij,kj->i", left_charge_vector_list, inv_norm_matrix_list, right_charge_vector_list)
    F_coulomb = k_coul * np.einsum("ij,ijkl,ik->jl", left_charge_vector_list, deriv_matrices * inml2, right_charge_vector_list)

    # Potentiel et forces pour Lennard-Jones
    inv_rOO = inv_norm_matrix_list[0]; grad_rOO = deriv_matrices[0]
    U_LJ = np.sum(4*eps_LJ*((sigma_LJ*inv_rOO)**12 - (sigma_LJ*inv_rOO)**6), axis=0)
    sr6 = (sigma_LJ * inv_rOO)**6
    sr12 = sr6**2
    prefac = 4 * eps_LJ * ( -12 * sr12 + 6 * sr6 ) * inv_rOO
    F_LJ = -np.einsum('ij,ijk->ik', prefac, grad_rOO)
    # F_LJ = np.einsum('ij,ijk->ik', 4 * eps_LJ * inv_rOO * (6*(sigma_LJ*inv_rOO)**6- 12*(sigma_LJ * inv_rOO)**12), grad_rOO)

    U = U_coulomb + U_LJ
    F = F_coulomb + F_LJ 

    return U, F

"""
TODO:
- Vérifier que je peux utiliser les quaternions monde pour passer de gradient quaternion à torque
- Transformer F dans le repère monde
"""