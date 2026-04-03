import numpy as np
from md_sim.core.system import mic

def build_nl(sys, r_c, r_s, L):
    """Construit la liste des voisins pour chaque molécule.

    Args:
        sys (MDSystem): Système moléculaire
        r_c (float): Rayon de coupure
        r_s (float): Distance de skin (buffer pour ne pas calculer la nl a chaque etape)
        L (float): Taille de la boîte

    Returns:
        np.ndarray: Matrice de voisin (N, N) binaire
    """
    rt = r_c + r_s
    cm_pos = sys.cm_pos
    N = len(cm_pos)
    Nmax = int(np.ceil(1.2 * 418.9 * rt**3))
    nl = np.full((N, Nmax), -1, dtype=int)  
    counts = np.zeros(N, dtype=int)
    nbr_list = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            dr = cm_pos[i] - cm_pos[j]
            dr = mic(dr, L)

            if np.dot(dr, dr) < rt**2:
                
                nl[i, counts[i]] = j
                counts[i] += 1

                
                nl[j, counts[j]] = i
                counts[j] += 1
        
    for i, neighbors in enumerate(nl):
        neighbors = [j for j in neighbors if j >= 0]
        nbr_list[i, neighbors] = 1

    return nbr_list
