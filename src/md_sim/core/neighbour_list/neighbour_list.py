import numpy as np
from scipy.spatial import cKDTree
from numba import njit

def build_neighbour_list(pos, L, rc):
    n = len(pos)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dr = pos[j_idx] - pos[i_idx]
    dr -= L * np.round(dr / L)
    r2 = np.einsum('ij,ij->i', dr, dr)
    mask = r2 < rc**2
    return i_idx[mask], j_idx[mask]

def build_neighbour_list_kdtree(pos, L, rc):
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rc, output_type='ndarray')
    return pairs[:, 0], pairs[:, 1]

@njit(cache=True)
def build_neighbour_list_numba(pos, L, rc):
    """
    Cell-list neighbour search pour boîte orthorhombique.

    Paramètres
    ----------
    pos : (n, 3) float64  — positions (CM pour molécules)
    L   : (3,)   float64  — [Lx, Ly, Lz]
    rc  : float           — rayon de coupure

    Retourne
    --------
    i_idx, j_idx : int64 arrays, j > i toujours
    """
    n   = pos.shape[0]
    Lx  = L[0]; Ly = L[1]; Lz = L[2]
    rc2 = rc * rc

    # ── Grille de cellules ────────────────────────────────────────────────────
    # nc[d] = nb de cellules dans la direction d
    # On impose nc >= 1 même si rc > L[d] (dégénère en O(n²) mais reste correct)
    ncx = max(1, int(Lx / rc))
    ncy = max(1, int(Ly / rc))
    ncz = max(1, int(Lz / rc))

    csx = Lx / ncx   # taille de cellule
    csy = Ly / ncy
    csz = Lz / ncz

    nc_tot = ncx * ncy * ncz

    # ── Linked list : head[cid] = premier atome, nxt[i] = atome suivant ──────
    head = np.full(nc_tot, -1, dtype=np.int64)
    nxt  = np.full(n,      -1, dtype=np.int64)

    for i in range(n):
        ix  = int(pos[i, 0] / csx) % ncx
        iy  = int(pos[i, 1] / csy) % ncy
        iz  = int(pos[i, 2] / csz) % ncz
        cid = ix * ncy * ncz + iy * ncz + iz
        nxt[i]    = head[cid]
        head[cid] = i

    # ── Allocation output ─────────────────────────────────────────────────────
    buf   = max(n * 40, 256)
    out_i = np.empty(buf, dtype=np.int64)
    out_j = np.empty(buf, dtype=np.int64)
    count = 0

    # ── Boucle sur toutes les cellules ────────────────────────────────────────
    for cx in range(ncx):
        for cy in range(ncy):
            for cz in range(ncz):
                cid = cx * ncy * ncz + cy * ncz + cz

                i = head[cid]
                while i != -1:

                    # 27 cellules voisines (inclut la cellule courante)
                    for ox in range(-1, 2):
                        for oy in range(-1, 2):
                            for oz in range(-1, 2):
                                nx_ = (cx + ox) % ncx
                                ny_ = (cy + oy) % ncy
                                nz_ = (cz + oz) % ncz
                                nid = nx_ * ncy * ncz + ny_ * ncz + nz_

                                j = head[nid]
                                while j != -1:
                                    if j > i:
                                        dx = pos[j, 0] - pos[i, 0]
                                        dy = pos[j, 1] - pos[i, 1]
                                        dz = pos[j, 2] - pos[i, 2]
                                        # MIC orthorhombique
                                        if dx >  0.5*Lx: dx -= Lx
                                        if dx < -0.5*Lx: dx += Lx
                                        if dy >  0.5*Ly: dy -= Ly
                                        if dy < -0.5*Ly: dy += Ly
                                        if dz >  0.5*Lz: dz -= Lz
                                        if dz < -0.5*Lz: dz += Lz
                                        r2 = dx*dx + dy*dy + dz*dz
                                        if r2 < rc2:
                                            if count == buf:
                                                out_i = np.concatenate((out_i, np.empty(buf, dtype=np.int64)))
                                                out_j = np.concatenate((out_j, np.empty(buf, dtype=np.int64)))
                                                buf *= 2
                                            out_i[count] = i
                                            out_j[count] = j
                                            count += 1
                                    j = nxt[j]
                    i = nxt[i]

    return out_i[:count], out_j[:count]


# ── Warm-up : déclenche la compilation JIT une seule fois à l'import ─────────
_p = np.random.rand(6, 3).astype(np.float64)
_L = np.array([1.0, 1.0, 1.0])
build_neighbour_list(_p, _L, 0.4)