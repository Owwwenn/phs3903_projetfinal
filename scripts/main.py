import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from md_sim.core.system import get_site_offsets 
from md_sim.core.system import make_initial_state
from md_sim.core.potential_force.coul_LJ_opt import compute_forces_and_torques
from md_sim.core.system import get_atom_positions
from md_sim.core.neighbour_list.neighbour_list import build_neighbour_list
from md_sim.core.system import SPCE
from md_sim.core.system import T_UNIT_PS
from md_sim.caracterisation.energy import kinetic_energy_rot
from md_sim.caracterisation.energy import kinetic_energy_trans
from md_sim.core.time_integrator.time_int_VECTORISED import velocity_verlet_step

# ─────────────────────────────────────────────
#  Simulation parameters
# ─────────────────────────────────────────────
N        = 216
rho      = 0.0334        # molecules/Å³
T_K      = 300.0
dt_fs    = 1.0          # fs  — smaller dt needed with Coulomb
nl_freq  = 20
snap_freq = 10
xi_t, xi_r = 0.0, 0.0
s_t,  s_r  = 0.0, 0.0
dt = (dt_fs * 1e-3) / T_UNIT_PS


T_start = 300.0
T_end   = 600
n_equil = 100    # steps à 300K pour équilibrer d'abord
n_ramp  = 10000  # steps pour monter de 300 → 500K
n_prod  = 100    # steps à 500K

n_steps = n_equil + n_ramp + n_prod

# Build parameter dict with precomputed geometry
r_h1, r_h2 = get_site_offsets(SPCE['theta'], SPCE['r_oh'])
p = {**SPCE, 'r_h1': r_h1, 'r_h2': r_h2}

cm_pos, cm_vel, quats, L_body, L_box = make_initial_state(N, rho, T_K, p)

L_box_new = np.array([L_box[0], L_box[1], L_box[2] * 5.0])
cm_pos[:, 2] += 2 * L_box[2]  # ← 2x pour centrer dans une boite 5x
L_box = L_box_new

nbr_list = build_neighbour_list(cm_pos, L_box, max(p['rc_LJ'], p['rc_coul']))
forces, tau, pe = compute_forces_and_torques(N, cm_pos, quats, L_box, nbr_list, p)

ke_t0 = kinetic_energy_trans(cm_vel, p['mass'])
ke_r0 = kinetic_energy_rot(L_body, p['I_body'])
E0    = ke_t0 + ke_r0 + pe

# Storage
te_arr    = np.zeros(n_steps)
pe_arr    = np.zeros(n_steps)
ket_arr   = np.zeros(n_steps)
ker_arr   = np.zeros(n_steps)
fnorm_arr = np.zeros(n_steps)
dE_arr    = np.zeros(n_steps)
T_arr = np.zeros(n_steps)
snaps_O   = []
snaps_H1  = []
snaps_H2  = []

rho_z_init, z_edges = np.histogram(cm_pos[:, 2], bins=100, range=(0, L_box[2]))
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
dz = z_edges[1] - z_edges[0]
rho_z_init = rho_z_init / (L_box[0] * L_box[1] * dz)  # molecules/Å³

print(f"{'step':>7}  {'E/N':>10}  {'PE/N':>10}  {'KE_t/N':>10}  {'KE_r/N':>10}  {'|dE/E0|':>10}")
print("-" * 65)

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
for step in range(n_steps):

    # Température cible
    if step < n_equil:
        T_K = T_start
    elif step < n_equil + n_ramp:
        frac = (step - n_equil) / n_ramp
        T_K = T_start + frac * (T_end - T_start)
    else:
        T_K = T_end

    if step % nl_freq == 0:
        nbr_list = build_neighbour_list(cm_pos, L_box, max(p['rc_LJ'], p['rc_coul']))

    cm_pos, cm_vel, quats, L_body, forces, tau, pe, xi_t, xi_r, s_t, s_r = velocity_verlet_step(
        cm_pos, cm_vel, quats, L_body, forces, tau,
        p['mass'], p['I_body'], L_box, p, dt, nbr_list,
        xi_t=xi_t, xi_r=xi_r, s_t=s_t, s_r=s_r,
        n_mol=N, T=T_K
    )

    ke_t = kinetic_energy_trans(cm_vel, p['mass'])
    ke_r = kinetic_energy_rot(L_body, p['I_body'])
    te   = ke_t + ke_r + pe

    kB_kcal = 0.001987  # kcal/mol/K
    T_inst_t = (2 * ke_t) / ((3*N - 3) * kB_kcal)
    T_inst_r = (2 * ke_r) / (3*N * kB_kcal)
    T_arr[step] = 0.5 * (T_inst_t + T_inst_r)

    te_arr[step]    = te
    pe_arr[step]    = pe
    ket_arr[step]   = ke_t
    ker_arr[step]   = ke_r
    fnorm_arr[step] = np.linalg.norm(forces)
    dE_arr[step]    = abs(te - E0) / abs(E0)

    if step % snap_freq == 0:
        O, H1, H2 = get_atom_positions(cm_pos, quats, p['r_h1'], p['r_h2'])
        snaps_O.append(O.copy())
        snaps_H1.append(H1.copy())
        snaps_H2.append(H2.copy())

    print(f"{step:>7}  {te/N:>10.4f}  {pe/N:>10.4f}  "
          f"{ke_t/N:>10.4f}  {ke_r/N:>10.4f}  {dE_arr[step]:>10.2e}")

    if not np.isfinite(te) or abs(te) > 1e12:
        print(f"  DIVERGED at step {step}")
        te_arr[step:] = np.nan; pe_arr[step:] = np.nan
        break

rho_z_final, _ = np.histogram(cm_pos[:, 2], bins=100, range=(0, L_box[2]))
rho_z_final = rho_z_final / (L_box[0] * L_box[1] * dz)

# ─────────────────────────────────────────────
#  Static plots
# ─────────────────────────────────────────────
T_target = np.where(np.arange(n_steps) < n_equil, T_start,
           np.where(np.arange(n_steps) < n_equil + n_ramp,
                    T_start + (np.arange(n_steps) - n_equil) / n_ramp * (T_end - T_start),
                    T_end))
T_err = np.abs(T_arr - T_target)

COLORS = ['#00ffaa', '#ff6b6b', '#4488ff', '#ffdd00']

def style_ax(ax, three_d=False):
    ax.set_facecolor('#111118')
    for spine in ax.spines.values(): spine.set_edgecolor('#555577')
    ax.tick_params(colors='#ddddee', labelsize=8)
    ax.xaxis.label.set_color('#ddddee'); ax.yaxis.label.set_color('#ddddee')
    ax.title.set_color('#ffffff')
    ax.grid(True, color='#2a2a40', lw=0.6, linestyle='--')
    if three_d:
        ax.zaxis.label.set_color('#ddddee')
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#2a2a40')

t_fs = np.arange(n_steps) * dt_fs

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#111118')
for ax in axes.flat: style_ax(ax)
ax_te, ax_pe, ax_ke, ax_fn, ax_dE, ax_T = axes.flat

ax_te.plot(t_fs, te_arr / N, color='#ffe66d', lw=1.5)
ax_te.set(title='Total energy / N', xlabel='Time [fs]', ylabel='kcal/mol')

ax_pe.plot(t_fs, pe_arr / N, color='#4ecdc4', lw=1.5)
ax_pe.set(title='Potential energy / N', xlabel='Time [fs]', ylabel='kcal/mol')

ax_ke.plot(t_fs, ket_arr / N, color='#ff6b6b', lw=1.5, label='KE_trans')
ax_ke.plot(t_fs, ker_arr / N, color='#4488ff', lw=1.5, label='KE_rot')
ax_ke.set(title='Kinetic energies / N', xlabel='Time [fs]', ylabel='kcal/mol')
ax_ke.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

ax_fn.plot(t_fs, fnorm_arr, color='#ff9f43', lw=1.5)
ax_fn.set(title='‖F‖ total force norm', xlabel='Time [fs]', ylabel='kcal/(mol·Å)')

valid = np.isfinite(dE_arr) & (dE_arr > 0)
ax_dE.semilogy(t_fs[valid], dE_arr[valid], color='#ff44cc', lw=1.5)
ax_dE.set(title='|ΔE/E₀| energy drift', xlabel='Time [fs]', ylabel='(log)')
ax_dE.yaxis.set_tick_params(which='both', colors='#ddddee')

ax_T.semilogy(t_fs, T_err,    color='#ff9944', lw=1.5, label='T inst')
# ax_T.plot(t_fs, T_target, color='#ffffff', lw=1.0, linestyle='--', label='T cible')
ax_T.set(title='Température instantanée', xlabel='Time [fs]', ylabel='K')
ax_T.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

fig.suptitle(f'SPC/E Water  |  LJ + Coulomb  |  Velocity Verlet  |  dt={dt_fs} fs  |  N={N}',
             color='#ffffff', fontsize=12, fontweight='bold')
plt.tight_layout()

fig2, (ax_rho1, ax_rho2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('#111118')
style_ax(ax_rho1); style_ax(ax_rho2)

ax_rho1.plot(z_centers, rho_z_init,  color='#4ecdc4', lw=1.5)
ax_rho1.axhline(rho, color='#ffffff', lw=1.0, linestyle='--', label='ρ bulk')
ax_rho1.set(title='Profil de densité initial', xlabel='z [Å]', ylabel='ρ [mol/Å³]')
ax_rho1.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

ax_rho2.plot(z_centers, rho_z_final, color='#ff6b6b', lw=1.5)
ax_rho2.axhline(rho, color='#ffffff', lw=1.0, linestyle='--', label='ρ bulk')
ax_rho2.set(title=f'Profil de densité final  (T={T_end}K)', xlabel='z [Å]', ylabel='ρ [mol/Å³]')
ax_rho2.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

fig2.suptitle('Profil de densité en z — interface liquide-vapeur',
              color='#ffffff', fontsize=12, fontweight='bold')
plt.tight_layout()

# ─────────────────────────────────────────────
#  3D Animation  (O=red, H=white)
# ─────────────────────────────────────────────
n_frames = len(snaps_O)
fig_anim = plt.figure(figsize=(9, 8))
fig_anim.patch.set_facecolor('#111118')
ax3 = fig_anim.add_subplot(111, projection='3d')
style_ax(ax3, three_d=True)
ax3.set_xlim(0, L_box[0]); ax3.set_ylim(0, L_box[1]); ax3.set_zlim(0, L_box[2])
ax3.set_xlabel('x [Å]', fontsize=8); ax3.set_ylabel('y [Å]', fontsize=8)
ax3.set_zlabel('z [Å]', fontsize=8)
ax3.set_title('SPC/E Water — O (red)  H (white)', color='#ffffff', fontsize=10)

scat_O  = ax3.scatter(*snaps_O[0].T,  s=40, c='#ff4444', alpha=0.9,
                       depthshade=True, edgecolors='none')
scat_H1 = ax3.scatter(*snaps_H1[0].T, s=15, c='#ffffff', alpha=0.8,
                       depthshade=True, edgecolors='none')
scat_H2 = ax3.scatter(*snaps_H2[0].T, s=15, c='#ffffff', alpha=0.8,
                       depthshade=True, edgecolors='none')
tt = ax3.text2D(0.03, 0.95, '', transform=ax3.transAxes, color='#ffdd88', fontsize=9,
                bbox=dict(facecolor='#1a1a2e', edgecolor='none', alpha=0.6, pad=2))

def update(frame):
    scat_O._offsets3d  = (snaps_O[frame][:,0],  snaps_O[frame][:,1],  snaps_O[frame][:,2])
    scat_H1._offsets3d = (snaps_H1[frame][:,0], snaps_H1[frame][:,1], snaps_H1[frame][:,2])
    scat_H2._offsets3d = (snaps_H2[frame][:,0], snaps_H2[frame][:,1], snaps_H2[frame][:,2])
    tt.set_text(f't = {frame * snap_freq * dt_fs:.1f} fs')

ani = animation.FuncAnimation(fig_anim, update, frames=n_frames, interval=60, blit=False)
plt.tight_layout()
plt.show()
