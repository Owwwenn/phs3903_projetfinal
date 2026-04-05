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
    N = sys.N
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

def def_rebuild(sys, L, skin):
     
     """Détermine si la neighbour list doit être reconstruite.

    Args:
        sys (MDSystem): Système moléculaire
        L (float): Taille de la boîte
        skin (float): Distance de peau

    Returns:
        bool: True si reconstruction nécessaire
    """

     disp = sys.cm_pos - sys.r_last
     disp = mic(disp, L)
     max_disp = np.max(np.einsum("ij,ij->i", disp, disp))

     return max_disp > (skin / 2) ** 2

def build_nl_pairs_cells(sys, r_c, r_s, L):
    """
    Build Verlet neighbor list using cell lists.
    
    Returns:
        i_idx, j_idx  (both shape: N_pairs,)
    """
    pos = sys.cm_pos
    N = sys.N

    rt = r_c + r_s
    rt2 = rt * rt

    # --- cell grid ---
    n_cells = np.floor(L / rt).astype(int)
    n_cells = np.maximum(n_cells, 1)   # avoid zero
    cell_size = L / n_cells

    # --- assign particles to cells ---
    cell_idx = np.floor(pos / cell_size).astype(int) % n_cells

    # flatten cell index for speed
    flat_idx = (
        cell_idx[:, 0]
        + n_cells[0] * (cell_idx[:, 1] + n_cells[1] * cell_idx[:, 2])
    )

    # build cell → particle map (contiguous)
    order = np.argsort(flat_idx)
    sorted_cells = flat_idx[order]

    # find cell boundaries
    unique_cells, start_idx = np.unique(sorted_cells, return_index=True)
    start_idx = np.append(start_idx, len(order))

    # helper: get atoms in a cell
    def get_cell_atoms(cell_id):
        i = np.searchsorted(unique_cells, cell_id)
        if i >= len(unique_cells) or unique_cells[i] != cell_id:
            return []
        return order[start_idx[i]:start_idx[i+1]]

    # --- neighbor offsets (27 cells) ---
    offsets = np.array([
        [dx, dy, dz]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ])

    i_list = []
    j_list = []

    # --- main loop over occupied cells ---
    for cell_id in unique_cells:
        atoms_i = get_cell_atoms(cell_id)
        if len(atoms_i) == 0:
            continue

        # recover 3D index
        iz = cell_id // (n_cells[0] * n_cells[1])
        rem = cell_id % (n_cells[0] * n_cells[1])
        iy = rem // n_cells[0]
        ix = rem % n_cells[0]

        base = np.array([ix, iy, iz])

        for off in offsets:
            neigh = (base + off) % n_cells
            neigh_id = (
                neigh[0]
                + n_cells[0] * (neigh[1] + n_cells[1] * neigh[2])
            )

            atoms_j = get_cell_atoms(neigh_id)
            if len(atoms_j) == 0:
                continue

            # --- pair construction ---
            for i in atoms_i:
                # vectorized against all atoms in neighbor cell
                dr = pos[atoms_j] - pos[i]
                dr -= L * np.round(dr / L)

                dist2 = np.einsum('ij,ij->i', dr, dr)

                mask = dist2 < rt2
                js = np.array(atoms_j)[mask]

                # enforce half list
                js = js[js > i]

                if len(js) > 0:
                    i_list.extend([i] * len(js))
                    j_list.extend(js.tolist())

    return np.array(i_list, dtype=int), np.array(j_list, dtype=int)