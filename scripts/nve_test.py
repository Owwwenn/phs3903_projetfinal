import numpy as np
from scipy.spatial.transform import Rotation
from md_sim.core.system import MDSystem, initialize_system, mic, wrap_positions
from md_sim.caracterisation.energy import kinetic_energy, rotational_energy
from md_sim.core.potential_force.coul_LJ import compute_forces_and_torques
from md_sim.core.time_integrator.time_int_VECTORISED import half_step_L, half_step_L_final, half_step_velocity, half_step_velocity_final, get_atom_positions, full_step_quat, full_step_position
from md_sim.models.three_site import spc_e

##
# Init
##

class parameters:
    def __init__(self):
        self.N = 2
        self.L = np.array([30,30,30])
        self.kB = 0.831446        
        self.T_init = 273   
        self.k_coul = 1389
        self.dt = 0.0003
        self.n_steps = 30000

param = parameters()
model = spc_e()

n_steps = param.n_steps
N = param.N
kB = param.kB
dt = param.dt

s = np.zeros(n_steps)
en = np.zeros(n_steps)

sys = initialize_system(model, param)
sys.cm_pos = wrap_positions(sys.cm_pos, param.L)
sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)

pos_init   = sys.cm_pos.copy()
L_init     = sys.L.sum(axis=0).copy()
U_arr = np.zeros(n_steps)

# r_cut = 10.0   # cut-off radius (Å)
# skin  = 4.0    # skin pour neighbour list
sys.r_last = sys.cm_pos.copy()
nbr_list = np.ones((N,N))- np.eye(N) #build_nl(sys, r_cut, skin, L)  # initial neighbour list

F_norms = np.zeros(n_steps)
T_norms = np.zeros(n_steps)

compute_forces_and_torques(sys, model, param, nbr_list)
E_init = kinetic_energy(sys, model) + rotational_energy(sys, model) + sys.U

##
# Solve
##

for step in range(n_steps):
    half_step_velocity(sys, model, dt)
    half_step_L(sys, model, dt)
    full_step_quat(sys, model, dt)
    full_step_position(sys, dt)
    sys.cm_pos = wrap_positions(sys.cm_pos, param.L)
    compute_forces_and_torques(sys, model, param, nbr_list)
    half_step_velocity_final(sys, model, dt)
    half_step_L_final(sys, model, dt)


    # === DIAGNOSTICS ===
    E     = kinetic_energy(sys, model) + rotational_energy(sys, model) + sys.U
    T     = 2 * kinetic_energy(sys, model) / (3 * N * kB)
    L_tot = sys.L.sum(axis=0)
    qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
    s[step] = step
    en[step] = E 
    U_arr[step] = sys.U
    F_norms[step] = np.linalg.norm(sys.force)
    T_norms[step] = np.linalg.norm(sys.T)
    O, H1, H2 = get_atom_positions(sys)

    print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
        f"T={T:.1f}K | |L_drift|={np.linalg.norm(L_tot-L_init):.2e} | "
        f"qnorm_err={qnorm:.2e}| sum T = {sys.T.sum(axis=0)}")