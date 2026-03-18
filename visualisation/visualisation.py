import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation

def water_mesh(theta, r_oh):
    s = np.sin(theta/2)
    c = np.cos(theta/2)
    
    O = pv.Sphere(radius=0.3, center=(0,0,0))
    H1 = pv.Sphere(radius=0.15, center=(0,r_oh*s,r_oh*c))
    H2 = pv.Sphere(radius=0.15, center=(0,-r_oh*s,r_oh*c))
    bond1 = pv.Line((0,0,0), (0,r_oh*s,r_oh*c))
    bond2 = pv.Line((0,0,0), (0,-r_oh*s,r_oh*c))

    O["rgb"]  = np.tile(np.array([255, 0, 0], dtype=np.uint8),   (O.n_points, 1))
    H1["rgb"] = np.tile(np.array([255, 255, 255], dtype=np.uint8), (H1.n_points, 1))
    H2["rgb"] = np.tile(np.array([255, 255, 255], dtype=np.uint8), (H2.n_points, 1))
    bond1["rgb"] = np.tile(np.array([50, 50, 50], dtype=np.uint8), (bond1.n_points,1))
    bond2["rgb"] = np.tile(np.array([50, 50, 50], dtype=np.uint8), (bond2.n_points,1))

    water = O.merge(H1).merge(H2).merge(bond1).merge(bond2)
    
    return water 

def make_transforms(n, r, R):
    T = np.zeros((n, 4, 4), dtype=float)
    T[:, :3, :3] = R 
    T[:, :3, 3] = r
    T[:, 3, 3] = 1.0
    return T

# Changer nom plus tard
def show_plot(n: int, v, Lx:float, Ly:float, Lz:float, theta:float, r_oh:float):
    """Crée un plot pyvista du système pour une position donnée et modèle d'eau donné (ici juste 3 sites à modifier plus tard si on change)

    Args:
        n (int): Nombre de molécules
        v (_type_): Matrice d'état du système
        Lx (float): Taille de la boîte de simulation en x
        Ly (float): Taille de la boîte de simulation en y
        Lz (float): Taille de la boîte de simulation en z
        theta (float): Angle H-O-H du modèle de la molécule d'eau
        r_oh (float): Distance O-H du modèle de la molécule d'eau
    """    
    r = v[:3*n].reshape(3,-1).T
    q = v[6*n:10*n].reshape(n,4)
    R = Rotation.from_quat(q).as_matrix()
    T = make_transforms(n, r, R)

    water = water_mesh(theta, r_oh)
    box = pv.Box(bounds=(-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2)) 

    plotter = pv.Plotter()
    plotter.add_mesh(box, style = 'wireframe', color = 'black')    
    
    meshes = []
    for i in range(n):
        mol = water.copy(deep = False)
        mol.transform(T[i], inplace = True)
        meshes.append(mol)
    merged_mesh = meshes[0].merge(meshes[1:])

    plotter.add_mesh(merged_mesh, scalars="rgb", rgb=True, line_width=5)
    plotter.show()