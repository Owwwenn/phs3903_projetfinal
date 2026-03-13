import numpy as np
import quaternion as qtn

def build_norm_matrix(n: int, v, Lx:float, Ly:float, Lz:float):
    """Construit une matrice de normes oxygène à oxygène entre les molécules de l'eau

    Args:
        n (int): Nombre de molécules
        v (_type_): Matrice d'état du système
        Lx (float): Taille de la boîte de simulation en x
        Ly (float): Taille de la boîte de simulation en y
        Lz (float): Taille de la boîte de simulation en z
    """  
    M = np.zeros(n,n)
    """
    TODO:
    - Ajouter le calcul de distances
    - Ajouter condition frontières périodiques
    """
    
    return M

"""
TODO
- Ajouter les distances O-H1, O-H2, H1-O, H1-H1, H1-H2, H2-O, H2-H1, H2-H2, en fonction de la matrice déjà créée et les quaternions de chaque molécule?
- Vérifier si l'idée est possible 
"""
