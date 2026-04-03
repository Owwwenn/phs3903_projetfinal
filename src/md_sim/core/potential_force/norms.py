import numpy as np
import quaternion as qtn
from scipy.spatial.transform import Rotation

def build_inv_norm_matrix(n: int, v, Lx:float, Ly:float, Lz:float, nbr_list, theta:float, r_oh:float):
    """Construit les matrices de normes entre les différentes sites des molécules de l'eau. Ainsi que leurs gradients.

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
        [M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2]: liste de 9 matrices de taille (n, n) qui donnent les inverses des distances entre chaque site.
        [dM_OO, dM_OH1, dM_OH2, dM_H1O, dM_H1H1, dM_H1H2, dM_H2O, dM_H2H1, dM_H2H2] liste de 9 tenseurs d'ordre 3 de taille (n, n, 6) contenant les gradients positionnels et angulaires (axes x,y,z) dans le repère de la molécule
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

    list_r = v[:3*n].reshape(n,3)
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
    rw = ri - L * np.round(ri / L)
    q_i = qtn.from_float_array(list_q[i_idx])
    q_j = qtn.from_float_array(list_q[j_idx])
    q_ip_rot = Rotation.from_quat(qtn.as_float_array(q_i.conjugate())[:, [1,2,3,0]])
    q = q_i.conjugate() * q_j
    # Passage du repère monde au repère de la molécule
    r = q_ip_rot.apply(rw)

    # Définition des vecteurs dans le repère de la molécule de référence
    u_h1 = qtn.rotate_vectors(q, r_h1) 
    u_h2 = qtn.rotate_vectors(q, r_h2) 

    # Définition de la fonction pour le calcul d'inverse des normes
    # def inv_norm(v):
    #     return 1.0 / np.sqrt(np.einsum('ij,ij->i', v, v)) 
    def inv_norm(v, eps=1e-12):
        return 1.0 / np.sqrt(np.einsum('ij,ij->i', v, v) + eps**2)

    # Calcul des inverses des distances
    inv_rOO = inv_norm(r)
    inv_rOH1 = inv_norm(r + u_h1)
    inv_rOH2 = inv_norm(r + u_h2)
    inv_rH1O = inv_norm(r - r_h1)
    inv_rH1H1 = inv_norm(r - r_h1 + u_h1)
    inv_rH1H2 = inv_norm(r - r_h1 + u_h2)
    inv_rH2O = inv_norm(r - r_h2)
    inv_rH2H1 = inv_norm(r - r_h2 + u_h1)
    inv_rH2H2 = inv_norm(r - r_h2 + u_h2)

    # Matrices d'inverses
    M_OO[i_idx, j_idx] = inv_rOO
    M_OH1[i_idx, j_idx] = inv_rOH1
    M_OH2[i_idx, j_idx] = inv_rOH2
    M_H1O[i_idx, j_idx] = inv_rH1O
    M_H1H1[i_idx, j_idx] = inv_rH1H1
    M_H1H2[i_idx, j_idx] = inv_rH1H2
    M_H2O[i_idx, j_idx] = inv_rH2O
    M_H2H1[i_idx, j_idx] = inv_rH2H1
    M_H2H2[i_idx, j_idx] = inv_rH2H2

    # Définition des vecteurs d'intérêt pour le calcul des gradients 
    q = qtn.as_float_array(q)
    x = r[:, 0]; y = r[:, 1]; z = r[:, 2]
    w = q[:, 0]; a = q[:, 1]; b = q[:, 2]; c = q[:, 3]
    G = 2*np.array([[-a, w, -c, b],
                    [-b, c, w, -a],
                    [-c, -b, a, w]])
    
    # Calcul des gradients
    ## Interaction O-O
    dM_OO = np.zeros((n, n, 7))
    dM_OO[i_idx, j_idx, 0] = x * inv_rOO
    dM_OO[i_idx, j_idx, 1] = y * inv_rOO
    dM_OO[i_idx, j_idx, 2] = z * inv_rOO
   
    ## Interaction O-H1
    dM_OH1 = np.zeros((n, n, 7))
    dM_OH1[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 + b*s) + 2*b*c*r_oh*w - 2*c*r_oh*s*w + x) * inv_rOH1
    dM_OH1[i_idx, j_idx, 1] = (2*c*r_oh*(b*c - a*w) - r_oh*s*(a**2 - b**2 + c**2 - w**2) + y) * inv_rOH1
    dM_OH1[i_idx, j_idx, 2] = (2*r_oh*s*(b*c + a*w) + c*r_oh*(-a**2 - b**2 + c**2 + w**2) + z) * inv_rOH1
    dM_OH1[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + c**2*r_oh*s**2*w + a**2*r_oh*(c**2 + s**2)*w + b**2*r_oh*(c**2 + s**2)*w + c**2*r_oh*w**3 + r_oh*s**2*w**3 + b*c*x - c*s*x + s*w*y + c*w*z + a*(-(c*y) + s*z))) * inv_rOH1
    dM_OH1[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x + b*s*x - c*w*y + s*w*z + a*(c**4*r_oh + b**2*r_oh*(c**2 + s**2) + c**2*r_oh*(s**2 + w**2) + s*(r_oh*s*w**2 - y) - c*z))) * inv_rOH1
    dM_OH1[i_idx, j_idx, 5] = (2*r_oh*(a**2*b*r_oh*(c**2 + s**2) + b**3*r_oh*(c**2 + s**2) + a*s*x + b*(c**4*r_oh + c**2*r_oh*(s**2 + w**2) + s*(r_oh*s*w**2 + y) - c*z) + c*(w*x + c*y + s*z))) * inv_rOH1
    dM_OH1[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*s**2 + a**2*c*r_oh*(c**2 + s**2) + b**2*c*r_oh*(c**2 + s**2) + c**3*r_oh*w**2 + c*r_oh*s**2*w**2 + a*c*x - s*w*x - c*s*y + c**2*z + b*(c*y + s*z))) * inv_rOH1

    ## Interaction O-H2
    dM_OH2 = np.zeros((n, n, 7))
    dM_OH2[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 - b*s) + 2*b*c*r_oh*w + 2*c*r_oh*s*w + x) * inv_rOH2
    dM_OH2[i_idx, j_idx, 1] = (2*c*r_oh*(b*c - a*w) + r_oh*s*(a**2 - b**2 + c**2 - w**2) + y) * inv_rOH2
    dM_OH2[i_idx, j_idx, 2] = (-2*r_oh*s*(b*c + a*w) + c*r_oh*(-a**2 - b**2 + c**2 + w**2) + z) * inv_rOH2
    dM_OH2[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + c**2*r_oh*s**2*w + a**2*r_oh*(c**2 + s**2)*w + b**2*r_oh*(c**2 + s**2)*w + c**2*r_oh*w**3 + r_oh*s**2*w**3 + b*c*x + c*s*x - s*w*y + c*w*z - a*(c*y + s*z))) * inv_rOH2
    dM_OH2[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x - c*w*y + a*(c**4*r_oh + b**2*r_oh*(c**2 + s**2) + c**2*r_oh*(s**2 + w**2) + s*(r_oh*s*w**2 + y) - c*z) - s*(b*x + w*z))) * inv_rOH2
    dM_OH2[i_idx, j_idx, 5] = (2*r_oh*(a**2*b*r_oh*(c**2 + s**2) + b**3*r_oh*(c**2 + s**2) - a*s*x + b*(c**4*r_oh + c**2*r_oh*(s**2 + w**2) + s*(r_oh*s*w**2 - y) - c*z) + c*(w*x + c*y - s*z))) * inv_rOH2
    dM_OH2[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*s**2 + a**2*c*r_oh*(c**2 + s**2) + b**2*c*r_oh*(c**2 + s**2) + c**3*r_oh*w**2 + c*r_oh*s**2*w**2 + a*c*x + s*w*x + c*s*y + c**2*z + b*(c*y - s*z))) * inv_rOH2

    ## Interaction H1-O
    dM_H1O = np.zeros((n,n,7))
    dM_H1O[i_idx, j_idx, 0] = x * inv_rH1O
    dM_H1O[i_idx, j_idx, 1] = (-r_oh*s + y) * inv_rH1O
    dM_H1O[i_idx, j_idx, 2] = (-c*r_oh + z) * inv_rH1O

    ## Interaction H1-H1
    dM_H1H1 = np.zeros((n,n,7))
    dM_H1H1[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 + b*s) + 2*b*c*r_oh*w - 2*c*r_oh*s*w + x) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 1] = (2*b*c**2*r_oh + b**3*r_oh*s - r_oh*(2*a*c*w + s*(1 + a**2 + c**2 - w**2)) + y) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 2] = (c**3*r_oh + 2*a*r_oh*s*w - c*r_oh*(1 + a**2 + b**2 - 2*b*s - w**2) + z) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + c**2*r_oh*w*(-1 + a**2 + b**2 + s**2 + w**2) + s*(r_oh*s*w*(-1 + a**2 + b**2 + w**2) + w*y + a*z) + c*(b*x - s*x - a*y + w*z))) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x + b*s*x - c*w*y + s*w*z + a*(c**4*r_oh + c**2*r_oh*(1 + b**2 + s**2 + w**2) + s*(r_oh*s*(1 + b**2 + w**2) - y) - c*z))) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 5] = (2*r_oh*(b**3*r_oh*(c**2 + s**2) + a*s*x + c**2*(-2*r_oh*s + y) + b*(c**4*r_oh + c**2*r_oh*(1 + a**2 + s**2 + w**2) + s*(r_oh*s*(-1 + a**2 + w**2) + y) - c*z) + c*(w*x + s*z))) * inv_rH1H1
    dM_H1H1[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*(-1 + a**2 + b**2 + s**2 + w**2) + c*(b**2*r_oh*s**2 + r_oh*s**2*(1 + a**2 + w**2) + a*x - s*y + b*(-2*r_oh*s + y)) + c**2*z + s*(-(w*x) + b*z))) * inv_rH1H1

    ## Interaction H1-H2
    dM_H1H2 = np.zeros((n,n,7))
    dM_H1H2[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 - b*s) + 2*b*c*r_oh*w + 2*c*r_oh*s*w + x) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 1] = (2*b*c**2*r_oh - b**2*r_oh*s - 2*a*c*r_oh*w + r_oh*s*(-1 + a**2 + c**2 - w**2) + y) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 2] = (c**3*r_oh - 2*a*r_oh*s*w - c*r_oh*(1 + a**2 + b**2 + 2*b*s - w**2) + z) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + a**2*r_oh*(c**2 + s**2)*w + c**2*r_oh*w*(-1 + b**2 + s**2 + w**2) + s*w*(r_oh*s*(1 + b**2 + w**2) - y) + a*(2*c*r_oh*s - c*y - s*z) + c*(b*x + s*x + w*z))) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x + c*w*(2*r_oh*s - y) + a*(c**4*r_oh + c**2*r_oh*(1 + b**2 + s**2 + w**2) + s*(r_oh*s*(-1 + b**2 + w**2) + y) - c*z) - s*(b*x + w*z))) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 5] = (2*r_oh*(b**3*r_oh*(c**2 + s**2) - a*s*x + b*(c**4*r_oh + c**2*r_oh*(1 + a**2 + s**2 + w**2) + s*(r_oh*s*(1 + a**2 + w**2) - y) - c*z) + c*(w*x + c*y - s*z))) * inv_rH1H2
    dM_H1H2[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*(-1 + a**2 + b**2 + s**2 + w**2) + c*(r_oh*s**2*(-1 + a**2 + b**2 + w**2) + a*x + (b + s)*y) + c**2*z + s*(w*x - b*z))) * inv_rH1H2

    ## Interaction H2-O
    dM_H2O = np.zeros((n,n,7))
    dM_H2O[i_idx, j_idx, 0] = x * inv_rH2O
    dM_H2O[i_idx, j_idx, 1] = (r_oh*s + y) * inv_rH2O
    dM_H2O[i_idx, j_idx, 2] = (-c*r_oh + z) * inv_rH2O

    ## Interaction H2-H1
    dM_H2H1 = np.zeros((n,n,7))
    dM_H2H1[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 + b*s) + 2*b*c*r_oh*w - 2*c*r_oh*s*w + x) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 1] = (2*b*c**2*r_oh + b**2*r_oh*s - 2*a*c*r_oh*w - r_oh*s*(-1 + a**2 + c**2 - w**2) + y) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 2] = (c**3*r_oh + 2*a*r_oh*s*w - c*r_oh*(1 + a**2 + b**2 - 2*b*s - w**2) + z) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + a**2*r_oh*(c**2 + s**2)*w + c**2*r_oh*w*(-1 + b**2 + s**2 + w**2) + s*w*(r_oh*s*(1 + b**2 + w**2) + y) + a*(-(c*(2*r_oh*s + y)) + s*z) + c*(b*x - s*x + w*z))) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x - c*w*(2*r_oh*s + y) + a*(c**4*r_oh + c**2*r_oh*(1 + b**2 + s**2 + w**2) + s*(r_oh*s*(-1 + b**2 + w**2) - y) - c*z) + s*(b*x + w*z))) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 5] = (2*r_oh*(b**3*r_oh*(c**2 + s**2) + a*s*x + b*(c**4*r_oh + c**2*r_oh*(1 + a**2 + s**2 + w**2) + s*(r_oh*s*(1 + a**2 + w**2) + y) - c*z) + c*(w*x + c*y + s*z))) * inv_rH2H1
    dM_H2H1[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*(-1 + a**2 + b**2 + s**2 + w**2) + c*(r_oh*s**2*(-1 + a**2 + b**2 + w**2) + a*x + b*y - s*y) + c**2*z + s*(-(w*x) + b*z))) * inv_rH2H1

    ## Interaction H2-H2
    dM_H2H2 = np.zeros((n,n,7))
    dM_H2H2[i_idx, j_idx, 0] = (2*a*r_oh*(c**2 - b*s) + 2*b*c*r_oh*w + 2*c*r_oh*s*w + x) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 1] = (r_oh*s + 2*c*r_oh*(b*c - a*w) + r_oh*s*(a**2 - b**2 + c**2 - w**2) + y) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 2] = (c**3*r_oh - 2*a*r_oh*s*w - c*r_oh*(1 + a**2 + b**2 + 2*b*s - w**2) + z) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 3] = (2*r_oh*(c**4*r_oh*w + c**2*r_oh*w*(-1 + a**2 + b**2 + s**2 + w**2) + s*(r_oh*s*w*(-1 + a**2 + b**2 + w**2) - w*y - a*z) + c*(b*x + s*x - a*y + w*z))) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 4] = (2*r_oh*(a**3*r_oh*(c**2 + s**2) + c**2*x - c*w*y + a*(c**4*r_oh + c**2*r_oh*(1 + b**2 + s**2 + w**2) + s*(r_oh*s*(1 + b**2 + w**2) + y) - c*z) - s*(b*x + w*z))) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 5] = (2*r_oh*(b**3*r_oh*(c**2 + s**2) - a*s*x + c**2*(2*r_oh*s + y) + b*(c**4*r_oh + c**2*r_oh*(1 + a**2 + s**2 + w**2) + s*(r_oh*s*(-1 + a**2 + w**2) - y) - c*z) + c*(w*x - s*z))) * inv_rH2H2
    dM_H2H2[i_idx, j_idx, 6] = (2*r_oh*(c**5*r_oh + c**3*r_oh*(-1 + a**2 + b**2 + s**2 + w**2) + c*(b**2*r_oh*s**2 + r_oh*s**2*(1 + a**2 + w**2) + a*x + s*y + b*(2*r_oh*s + y)) + c**2*z + s*(w*x - b*z))) * inv_rH2H2
    
    def to_6dof(dM, G, i_idx, j_idx):
        """Remplace les 4 colonnes quaternion (3:7) par 3 colonnes torque. s/o Claude"""
        N = dM.shape[0]
        dM_new = np.zeros((N, N, 6))
        dM_new[i_idx, j_idx, :3] = dM[i_idx, j_idx, :3]  # grad position inchangé
        grad_q = dM[i_idx, j_idx, 3:7]                    # (N_pairs, 4)
        # G: (3, 4, N_pairs) -> einsum donne (N_pairs, 3)
        dM_new[i_idx, j_idx, 3:6] = np.einsum('klp,pl->pk', G, grad_q)
        return dM_new

    dM_OO = to_6dof(dM_OO, G, i_idx, j_idx)
    dM_OH1 = to_6dof(dM_OH1, G, i_idx, j_idx)
    dM_OH2 = to_6dof(dM_OH2, G, i_idx, j_idx)
    dM_H1O = to_6dof(dM_H1O, G, i_idx, j_idx)
    dM_H1H1 = to_6dof(dM_H1H1, G, i_idx, j_idx)
    dM_H1H2 = to_6dof(dM_H1H2, G, i_idx, j_idx)
    dM_H2O = to_6dof(dM_H2O, G, i_idx, j_idx)
    dM_H2H1 = to_6dof(dM_H2H1, G, i_idx, j_idx)
    dM_H2H2 = to_6dof(dM_H2H2, G, i_idx, j_idx)

    return [[M_OO, M_OH1, M_OH2, M_H1O, M_H1H1, M_H1H2, M_H2O, M_H2H1, M_H2H2], [dM_OO, dM_OH1, dM_OH2, dM_H1O, dM_H1H1, dM_H1H2, dM_H2O, dM_H2H1, dM_H2H2]]

"""
TODO
- Passer par une matrice triangulaire -> ajouter sa transposée (comme la matrice est symétrique)
"""
