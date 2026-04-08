import numpy as np
from scipy.spatial.transform import Rotation
from md_sim.core.system import MDSystem, initialize_system, mic, wrap_positions, get_atom_positions
from md_sim.caracterisation.energy import kinetic_energy, rotational_energy
from md_sim.core.potential_force.coul_LJ import compute_forces_and_torques
from md_sim.core.time_integrator.time_int_VECTORISED import half_step_L, half_step_L_final, half_step_velocity, half_step_velocity_final, full_step_quat, full_step_position
from md_sim.models.three_site import spc_e
from md_sim.core.nose_hoover.nh_trotter import update_eta, update_PL
from md_sim.core.neighbour_list.neighbour_list import build_nl_pairs_cells, def_rebuild
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

##
# Init
##

class parameters:
    def __init__(self):
        self.N = 100
        self.L = np.array([30, 30, 30])
        # self.L = np.array([self.N/3, self.N/3, self.N/3])
        self.kB = 0.831446        
        self.T_init = 300   
        self.T_target = 300
        self.k_coul = 138.9
        self.dt = 0.003
        self.N_r = 10
        self.r_c = 10.0
        self.r_s = 1.0
        self.r_c_LJ = 3.0
        self.r_s_LJ = 0.3
        self.n_steps = 5000

param = parameters()
model = spc_e()

n_steps = param.n_steps
N = param.N
kB = param.kB
dt = param.dt
Lx, Ly, Lz = param.L

skip = 10
O_traj  = []
H1_traj = []
H2_traj = []

s = np.zeros(n_steps)
en = np.zeros(n_steps)

sys = initialize_system(model, param)
sys.cm_pos = wrap_positions(sys.cm_pos, param.L)
# sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)
sys.quat = Rotation.random(N).as_quat()[:, [3,0,1,2]]
sys.eta = 0.0

pos_init   = sys.cm_pos.copy()
L_init     = sys.L.sum(axis=0).copy()
U_arr = np.zeros(n_steps)

sys.r_last = sys.cm_pos.copy()
# nbr_list = np.ones((N,N))- np.eye(N) 
nbr_list_coul = build_nl_pairs_cells(sys, param.r_c, param.r_s, param.L)
nbr_list_LJ = build_nl_pairs_cells(sys, param.r_c_LJ, param.r_s_LJ, param.L)
nbr_list = [nbr_list_coul, nbr_list_LJ]

F_norms = np.zeros(n_steps)
T_norms = np.zeros(n_steps)

compute_forces_and_torques(sys, model, param, nbr_list)
E_init = kinetic_energy(sys, model) + rotational_energy(sys, model) + sys.U

##
# Solve
##

for step in range(n_steps):
    # === DIAGNOSTICS ===
    K = kinetic_energy(sys, model) + rotational_energy(sys, model)
    E     = K + sys.U
    T     = 2 * K / (6 * N * kB)
    L_tot = sys.L.sum(axis=0)
    # qnorm = np.max(np.abs(np.linalg.norm(sys.quat, axis=1) - 1.0))
    # s[step] = step
    en[step] = E 
    U_arr[step] = sys.U
    F_norms[step] = np.linalg.norm(sys.force)
    T_norms[step] = np.linalg.norm(sys.T)
    # T_norms[step] = T
    L_orbital = np.sum(np.cross(sys.cm_pos, sys.force), axis=0)
    L_spin    = sys.T.sum(axis=0)
    print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
        f"T={T:.1f}K | ",
        f"sum F = {sys.force.sum(axis=0)}",
        f"sum L = {L_orbital + L_spin}",
        f"sum T = {sys.T.sum(axis=0)}")
    sys.r_last = sys.cm_pos.copy()

    half_step_velocity(sys, model, dt)
    half_step_L(sys, model, dt)
    full_step_quat(sys, model, dt)
    full_step_position(sys, dt)
    sys.cm_pos = wrap_positions(sys.cm_pos, param.L)
    if step % param.N_r == 0:
        if def_rebuild(sys, param.L, param.r_s):
            nbr_list[0] = build_nl_pairs_cells(sys, param.r_c, param.r_s, param.L)
        if def_rebuild(sys, param.L, param.r_s_LJ):
            nbr_list[1] = build_nl_pairs_cells(sys, param.r_c_LJ, param.r_s_LJ, param.L)
    compute_forces_and_torques(sys, model, param, nbr_list)
    half_step_velocity_final(sys, model, dt)
    half_step_L_final(sys, model, dt)
    # update_eta(sys, model, param, N, dt)
    # update_PL(sys, dt)

    if step % skip == 0:
        O, H1, H2 = get_atom_positions(sys, model)
        O_traj.append(O.copy())
        H1_traj.append(H1.copy())
        H2_traj.append(H2.copy())


fig_anim = plt.figure(figsize=(7, 7))
ax = fig_anim.add_subplot(111, projection='3d')

scat_O  = ax.scatter([], [], [], c='red',   s=40,  label='O')
scat_H1 = ax.scatter([], [], [], c='white', s=20,  edgecolors='gray', label='H')
scat_H2 = ax.scatter([], [], [], c='white', s=20,  edgecolors='gray')

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(0, Lz)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_facecolor('black')
fig_anim.patch.set_facecolor('black')
title = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)


def update(frame):
    O  = O_traj[frame]
    H1 = H1_traj[frame]
    H2 = H2_traj[frame]
    scat_O._offsets3d  = (O[:,0],  O[:,1],  O[:,2])
    scat_H1._offsets3d = (H1[:,0], H1[:,1], H1[:,2])
    scat_H2._offsets3d = (H2[:,0], H2[:,1], H2[:,2])
    title.set_text(f'frame {frame} / step {frame * skip}')
    return scat_O, scat_H1, scat_H2

ani = animation.FuncAnimation(fig_anim, update, frames=len(O_traj), interval=10, blit=False)
plt.tight_layout()
plt.show()

# print(max(abs(en - E_init*np.ones(len(en)))/E_init*100))

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(F_norms)
plt.title('||F||')
plt.xlabel('step')

plt.subplot(1, 4, 2)
plt.plot(T_norms)
plt.title("torques")
plt.xlabel('step')

plt.subplot(1, 4, 3)
plt.plot(en)
plt.title('E_tot')
plt.xlabel('step')

plt.subplot(1, 4, 4)
plt.plot(U_arr)
plt.title('U')
plt.xlabel('step')

plt.tight_layout()
plt.show()