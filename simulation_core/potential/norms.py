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
                q = qtn.from_float_array(list_q[i]).conjugate() * qtn.from_float_array(list_q[j]) 

                # Définition des positions des sites hydrogènes dans le repère de la molécule
                r_h1 = r_oh * np.array([0,s,c])  
                r_h2 = r_oh * np.array([0,-s,c])  

                # Définition des vecteurs dans le repère monde
                u_h1 = qtn.rotate_vectors(q, r_h1) 
                u_h2 = qtn.rotate_vectors(q, r_h2) 

                # Calcul des inverses des distances
                M_OO[i, j] = 1 / np.sqrt(r @ r)
                M_OH1[i, j] = 1 / np.sqrt((r + u_h1) @ (r + u_h1))
                M_OH2[i, j] = 1 / np.sqrt((r + u_h2) @ (r + u_h2))
                M_H1O[i, j] = 1 / np.sqrt((r - r_h1) @ (r - r_h1))
                M_H1H1[i, j] = 1 / np.sqrt((r - r_h1 + u_h1) @ (r - r_h1 + u_h1))
                M_H1H2[i, j] = 1 / np.sqrt((r - r_h1 + u_h2) @ (r - r_h1 + u_h2))
                M_H2O[i, j] = 1 / np.sqrt((r - r_h2) @ (r - r_h2))
                M_H2H1[i, j] = 1 / np.sqrt((r - r_h2 + u_h1) @ (r - r_h2 + u_h1))
                M_H2H2[i, j] = 1 / np.sqrt((r - r_h2 + u_h2) @ (r - r_h2 + u_h2))
    
    return [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]

"""
TODO
- Rajouter des tests dans ifname main pour vérifier que la fonciton fonctionne
- Enlever la double boucle et vectoriser avec numpy
- Passer à Julia si je veux garder la boucle
"""
