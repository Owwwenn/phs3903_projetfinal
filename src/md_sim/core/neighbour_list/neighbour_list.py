import numpy as np

def build_neighbour_list(pos, L, rc):
    n = len(pos)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dr = pos[j_idx] - pos[i_idx]
    dr -= L * np.round(dr / L)
    r2 = np.einsum('ij,ij->i', dr, dr)
    mask = r2 < rc**2
    return i_idx[mask], j_idx[mask]