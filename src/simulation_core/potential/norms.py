import numpy as np
import quaternion as qtn

def build_inv_norm_matrix(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float):
    """Construit les matrices de normes entre les différentes sites des molécules de l'eau.

    Args:
        n (int): Nombre de molécules
        v (_type_): Matrice d'état du système
        Lx (float): Taille de la boîte de simulation en x
        Ly (float): Taille de la boîte de simulation en y
        Lz (float): Taille de la boîte de simulation en z
        nbr_list (_type_): Liste de booléens des voisins pour chaque molécule
        theta (float): Angle H-O-H du modèle de la molécule d'eau
        r_oh (float): Distance O-H du modèle de la molécule d'eau

    Returns:
        [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]: 9 matrices de taille nxn qui donnent les inverses des distances entre chaque site.
    """ 
    M_OO = np.zeros((n,n))
    M_OH1 = np.zeros((n,n))
    M_OH2 = np.zeros((n,n))
    M_H1O = np.zeros((n,n))
    M_H1H1 = np.zeros((n,n))
    M_H1H2 = np.zeros((n,n))
    M_H2O = np.zeros((n,n))
    M_H2H1 = np.zeros((n,n))
    M_H2H2 = np.zeros((n,n))

    list_r = v[:3*n].reshape(3,-1).T
    list_q = v[6*n:10*n].reshape(n,4)[:, [3,0,1,2]] 
    L = np.array([Lx,Ly,Lz])

    s = np.sin(theta/2)
    c = np.cos(theta/2)
 
    # Définition des positions des sites hydrogènes dans le repère de la molécule
    r_h1 = r_oh * np.array([0,s,c])  
    r_h2 = r_oh * np.array([0,-s,c])  

    i_idx, j_idx = np.where(nbr_list)

    # Définition de r et de q relatifs
    ri = list_r[j_idx] - list_r[i_idx]             
    r = ri - L * np.round(ri / L)
    q_i = qtn.from_float_array(list_q[i_idx])
    q_j = qtn.from_float_array(list_q[j_idx])
    q = q_i.conjugate() * q_j

    # Définition des vecteurs dans le repère monde
    u_h1 = qtn.rotate_vectors(q, r_h1) 
    u_h2 = qtn.rotate_vectors(q, r_h2) 

    # Définition de la fonction pour le calcul d'inverse des normes
    def inv_norm(v):
        return 1.0 / np.sqrt(np.einsum('ij,ij->i', v, v)) 

    # Calcul des inverses des distances
    M_OO[i_idx, j_idx] = inv_norm(r)
    M_OH1[i_idx, j_idx] = inv_norm(r + u_h1)
    M_OH2[i_idx, j_idx] = inv_norm(r + u_h2)
    M_H1O[i_idx, j_idx] = inv_norm(r - r_h1)
    M_H1H1[i_idx, j_idx] = inv_norm(r - r_h1 + u_h1)
    M_H1H2[i_idx, j_idx] = inv_norm(r - r_h1 + u_h2)
    M_H2O[i_idx, j_idx] = inv_norm(r - r_h2)
    M_H2H1[i_idx, j_idx] = inv_norm(r - r_h2 + u_h1)
    M_H2H2[i_idx, j_idx] = inv_norm(r - r_h2 + u_h2)

    return [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]

"""
TODO
- Passer par une matrice triangulaire -> ajouter sa transposée (comme la matrice est symétrique)
"""
