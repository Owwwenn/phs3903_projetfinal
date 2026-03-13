import numpy as np
import quaternion as qtn

def build_norm_matrix(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float):
    """Construit les matrices de normes entre les différentes sites des molécules de l'eau.

    Args:
        n (int): Nombre de molécules
        v (_type_): Matrice d'état du système
        Lx (float): Taille de la boîte de simulation en x
        Ly (float): Taille de la boîte de simulation en y
        Lz (float): Taille de la boîte de simulation en z
        nbr_list (_type_): Liste de booléens des voisins pour chaque molécule

    Returns:
        [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]: 9 matrices de taille nxn qui donnent les distances entre chaque site.
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
    list_q = v[6*n:7*n]
    L = np.array(Lx,Ly,Lz)

    s = np.sin(theta/2)
    c = np.cos(theta/2)
 
    for i, mol1 in enumerate(nbr_list):
        for j, mol2 in enumerate(mol1):
            if mol2:
                # Définition de r et de q relatifs
                ri = list_r[j] - list_r[i]             
                r = ri - np.dot(L, np.round(np.dot(ri, 1/L)))
                q = qtn.quaternion(list_q[i]).conjugate() * qtn.quaternion(list_q[j]) 

                # Définition des 
                r_h1 =  
    """
    TODO:
    - Ajouter le calcul de distances
    - Ajouter condition frontières périodiques
    """
    
    return [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]

"""
TODO
- Ajouter les distances O-H1, O-H2, H1-O, H1-H1, H1-H2, H2-O, H2-H1, H2-H2, en fonction de la matrice déjà créée et les quaternions de chaque molécule?
- Vérifier si l'idée est possible 
"""
