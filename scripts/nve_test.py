import numpy as np
from scipy.spatial.transform import Rotation
from md_sim.core.system import MDSystem, initialize_system, mic, wrap_positions, get_atom_positions
from md_sim.caracterisation.energy import kinetic_energy, rotational_energy
from md_sim.core.potential_force.coul_LJ import compute_forces_and_torques
from md_sim.core.time_integrator.time_int_VECTORISED import half_step_L, half_step_L_final, half_step_velocity, half_step_velocity_final, full_step_quat, full_step_position
from md_sim.models.three_site import spc_e
from md_sim.core.nose_hoover.nh_trotter import update_eta, update_PL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

##
# Init
##

class parameters:
    def __init__(self):
        self.N = 50
        self.L = np.array([200,200,200])
        self.kB = 0.831446        
        self.T_init = 273   
        self.T_target = 400
        self.k_coul = 1389
        self.dt = 0.0003
        self.n_steps = 100000

param = parameters()
model = spc_e()

n_steps = param.n_steps
N = param.N
kB = param.kB
dt = param.dt
Lx, Ly, Lz = param.L

skip = 100
O_traj  = []
H1_traj = []
H2_traj = []

s = np.zeros(n_steps)
en = np.zeros(n_steps)

sys = initialize_system(model, param)
sys.cm_pos = wrap_positions(sys.cm_pos, param.L)
sys.quat = np.roll(Rotation.random(N).as_quat(), 1, axis=1)
sys.eta = 0.0

pos_init   = sys.cm_pos.copy()
L_init     = sys.L.sum(axis=0).copy()
U_arr = np.zeros(n_steps)

sys.r_last = sys.cm_pos.copy()
nbr_list = np.ones((N,N))- np.eye(N) 

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
    # update_eta(sys, model, param, N, dt)
    # update_PL(sys, dt)


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
    print(f"step {step:4d} | E={E:.4f} dE={abs(E-E_init)/E_init*100:.4f}% | "
        f"T={T:.1f}K | "
        f"sum T = {sys.T.sum(axis=0)}")

    # if step % skip == 0:
    #     O, H1, H2 = get_atom_positions(sys, model)
    #     O_traj.append(O.copy())
    #     H1_traj.append(H1.copy())
    #     H2_traj.append(H2.copy())


# fig_anim = plt.figure(figsize=(7, 7))
# ax = fig_anim.add_subplot(111, projection='3d')

# scat_O  = ax.scatter([], [], [], c='red',   s=80,  label='O')
# scat_H1 = ax.scatter([], [], [], c='white', s=40,  edgecolors='gray', label='H')
# scat_H2 = ax.scatter([], [], [], c='white', s=40,  edgecolors='gray')

# ax.set_xlim(0, Lx)
# ax.set_ylim(0, Ly)
# ax.set_zlim(0, Lz)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# ax.set_facecolor('black')
# fig_anim.patch.set_facecolor('black')
# title = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)


# def update(frame):
#     O  = O_traj[frame]
#     H1 = H1_traj[frame]
#     H2 = H2_traj[frame]
#     scat_O._offsets3d  = (O[:,0],  O[:,1],  O[:,2])
#     scat_H1._offsets3d = (H1[:,0], H1[:,1], H1[:,2])
#     scat_H2._offsets3d = (H2[:,0], H2[:,1], H2[:,2])
#     title.set_text(f'frame {frame} / step {frame * skip}')
#     return scat_O, scat_H1, scat_H2

# ani = animation.FuncAnimation(fig_anim, update, frames=len(O_traj), interval=50, blit=False)
# plt.tight_layout()
# plt.show()

# print(max(abs(en - E_init*np.ones(len(en)))/E_init*100))

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(F_norms)
plt.title('||F||')
plt.xlabel('step')

plt.subplot(1, 4, 2)
plt.plot(T_norms)
plt.title('||T||')
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