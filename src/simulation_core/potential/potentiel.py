from norms import build_inv_norm_matrix
import numpy as np

def build_potential_vector(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float, q_o:float, q_h:float):
    """Construit le vecteur de potentiel appliqué sur chacune des n molécules du système. 

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
    inv_norm_matrix_list = np.array(build_inv_norm_matrix(n, v, Lx, Ly, Lz, nbr_list, theta, r_oh))
    left_charge_vector_list = np.array([q_o*np.ones(n)]*3 + [q_h*np.ones(n)]*6)
    right_charge_vector_list = np.array(([q_o*np.ones(n)] + [q_h*np.ones(n)]*2) * 3)
    U = np.einsum("ki,kij,kj->i", left_charge_vector_list, inv_norm_matrix_list, right_charge_vector_list)
    return U

"""
TODO:
- Calculer le gradient de potentiel pour les forces en même temps? pour éviter les appels répétés des mêmes fonctions.
- Déterminer s'il est nécessaire d'ajouter le 1/(4 pi epsilon0) ici 
"""