from simulation_core.potential.norms import build_inv_norm_matrix
import numpy as np

def build_potential_vector_force_torque_matrix(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float, q_o:float, q_h:float):
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

    Returns:
        numpy.array: Vecteur de taille n contenant le potentiel appliqué sur chaque molécule.
    """    
    list_q = v[6*n:10*n].reshape(n,4)[:, [3,0,1,2]] 
    w = list_q[:, 0]; a = list_q[:, 1]; b = list_q[:, 2]; c = list_q[:, 3]
    G = 2*np.array([[-a, w, -c, b],
                    [-b, c, w, -a],
                    [-c, -b, a, w]])

    [inv_norm_matrix_list, deriv_matrices]= build_inv_norm_matrix(n, v, Lx, Ly, Lz, nbr_list, theta, r_oh)
    inv_norm_matrix_list = np.array(inv_norm_matrix_list)
    deriv_matrices = np.array(deriv_matrices)
    left_charge_vector_list = np.array([q_o*np.ones(n)]*3 + [q_h*np.ones(n)]*6)
    right_charge_vector_list = np.array(([q_o*np.ones(n)] + [q_h*np.ones(n)]*2) * 3)

    U = np.einsum("ki,kij,kj->i", left_charge_vector_list, inv_norm_matrix_list, right_charge_vector_list)

    inml2 = (inv_norm_matrix_list**2)[..., None]
    grad = np.einsum("ij,ijkl,ik->jl", left_charge_vector_list, -deriv_matrices * inml2, right_charge_vector_list)
    F_lin = -grad[:, :3]
    tau = np.einsum("ijk,kj->ki", G, grad[:, 3:])
    F = np.array([F_lin.reshape(n,3), tau.reshape(n,3)]).reshape(n,6)

    return U, F

"""
TODO:
- Vérifier que je peux utiliser les quaternions monde pour passer de gradient quaternion à torque
- Transformer F dans le repère monde
- Calculer le gradient de potentiel pour les forces en même temps? pour éviter les appels répétés des mêmes fonctions.
- Déterminer s'il est nécessaire d'ajouter le 1/(4 pi epsilon0) ici 
"""