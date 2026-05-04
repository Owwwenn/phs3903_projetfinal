import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from md_sim.core.system import get_site_offsets 
from md_sim.core.system import make_initial_state
from md_sim.core.potential_force.coul_LJ import compute_forces_and_torques
from md_sim.core.system import get_atom_positions
from md_sim.core.neighbour_list.neighbour_list import build_neighbour_list, build_neighbour_list_kdtree, build_neighbour_list_numba
from md_sim.core.system import SPCE
from md_sim.core.system import T_UNIT_PS
from md_sim.caracterisation.pressure import get_pressure
from md_sim.caracterisation.energy import kinetic_energy_rot
from md_sim.caracterisation.energy import kinetic_energy_trans
from md_sim.caracterisation.energy import get_cv_fluctuations
from md_sim.core.time_integrator.time_int_VECTORISED import velocity_verlet_step

# ─────────────────────────────────────────────
#  Simulation parameters
# ─────────────────────────────────────────────
run_label = "30012005k5k5kUDv3"   # ← changer ce nom pour identifier la run

N        = 256
rho      = 0.0334        # molecules/Å³
# rho      = 2e-5        # molecules/Å³
# T_K      = 300.0
dt_fs    = 1.0          # fs  — smaller dt needed with Coulomb
nl_freq  = 10
snap_freq = 50
xi_t, xi_r = 0.0, 0.0
s_t,  s_r  = 0.0, 0.0
dt = (dt_fs * 1e-3) / T_UNIT_PS
# z = rho/2e-5
z = 5.0

# initial pour baro
eta_p   = 0.0
tau_P   = 10.0          # t* — couplage barostat (~500 fs)
P0      = 1.0/6.947e4  # 1 atm en kcal/(mol·Å³)

_mode = "ewald"
_phase = 'liquid'   # interface pré-formée → barrière de nucléation réduite

kB_kcal = 0.001987
T_start = 300
T_end   = 1200      # SPC/E Tb ~400-450 K; 700K suffit pour observer la vaporisation
n_equil = 5000     # équilibration plus longue pour former l'interface
n_ramp  = 5000    # rampe lente (~20 ps) pour donner le temps d'évaporer
n_prod  = 5000    # production à T_end

T_K_list = np.concatenate([
    T_start * np.ones(n_equil),           # équilibration liquide
    np.linspace(T_start, T_end, n_ramp),  # chauffe → gaz
    T_end   * np.ones(n_prod),            # production gaz
    np.linspace(T_end, T_start, n_ramp),  # refroidissement → liquide
    T_start * np.ones(n_prod),            # production liquide
])

n_steps = len(T_K_list)
Q_p     = N * kB_kcal * T_start * tau_P**2   # masse barostat

# Build parameter dict with precomputed geometry
r_h1, r_h2 = get_site_offsets(SPCE['theta'], SPCE['r_oh'])
p = {**SPCE, 'r_h1': r_h1, 'r_h2': r_h2}
# Pour précalculer G une seule fois au début de la simulation :
cm_pos, cm_vel, quats, L_body, L_box = make_initial_state(N, rho, T_start, p, seed=42, phase=_phase)

L_box_new = np.array([L_box[0], L_box[1], L_box[2] * z])
cm_pos[:, 2] += (z/2) * L_box[2]  # centre le liquide dans la boite 5x en z
L_box = L_box_new

nbr_list_LJ = build_neighbour_list(cm_pos, L_box, p['rc_LJ'])
nbr_list_coul = build_neighbour_list(cm_pos, L_box, p['rc_coul'])
nbr_list = [nbr_list_LJ, nbr_list_coul]
forces, tau, pe, virial, virial_zz = compute_forces_and_torques(N, cm_pos, quats, L_box, nbr_list, p, _mode)
 
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
T_arr  = np.zeros(n_steps)
cv_arr = np.full(n_steps, np.nan)
snaps_O   = []
snaps_H1  = []
snaps_H2  = []
snaps_box = []

rho_z_init, z_edges = np.histogram(cm_pos[:, 2], bins=100, range=(0, L_box[2]))
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
dz = z_edges[1] - z_edges[0]
rho_z_init = rho_z_init / (L_box[0] * L_box[1] * dz)  # molecules/Å³

checkpoint_fracs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
checkpoint_steps = [int(f * n_steps) for f in checkpoint_fracs]
rho_z_checkpoints = []   # liste de (step, label, rho_z)

print(f"{'step':>7}  {'E/N':>10}  {'PE/N':>10}  {'KE_t/N':>10}  {'KE_r/N':>10}  {'|dE/E0|':>10}")
print("-" * 65)

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
for step in range(n_steps):
    T_K = T_K_list[step] 

    # Rebuild every step when barostat is active (box changes each step)
    nbr_list_LJ   = build_neighbour_list(cm_pos, L_box, p['rc_LJ'])
    nbr_list_coul = build_neighbour_list(cm_pos, L_box, p['rc_coul'])
    nbr_list = [nbr_list_LJ, nbr_list_coul]

    cm_pos, cm_vel, quats, L_body, forces, tau, pe, xi_t, xi_r, s_t, s_r, virial, virial_zz, eta_p, L_box = velocity_verlet_step(
        cm_pos, cm_vel, quats, L_body, forces, tau,
        p['mass'], p['I_body'], L_box, p, dt, nbr_list,
        xi_t=xi_t, xi_r=xi_r, s_t=s_t, s_r=s_r,
        n_mol=N, T=T_K,
        mode=_mode,
        virial=virial, virial_zz=virial_zz,
        eta_p=eta_p, Q_p=Q_p, P0=P0,
        barostat='none'
    )


    ke_t = kinetic_energy_trans(cm_vel, p['mass'])
    ke_r = kinetic_energy_rot(L_body, p['I_body'])
    te   = ke_t + ke_r + pe

    T_inst_t = (2 * ke_t) / ((3*N - 3) * kB_kcal)
    T_inst_r = (2 * ke_r) / (3*N * kB_kcal)
    T_inst = 0.5 * (T_inst_t + T_inst_r)
    T_arr[step] = T_inst
    te_arr[step]    = te

    cv_arr[step] = get_cv_fluctuations(te_arr, T_arr, step, N, kB_kcal, window=200)

    Pzz_atm = eta_p * 6.947e4   # eta_p holds Pzz returned by semi_isotropic_barostat_z
    pe_arr[step]    = pe
    ket_arr[step]   = ke_t
    ker_arr[step]   = ke_r
    fnorm_arr[step] = np.linalg.norm(forces)
    dE_arr[step]    = abs(te - E0) / abs(E0)

    print(f"{step:>7}  {te/N:>10.4f}  {pe/N:>10.4f}  "
            f"{ke_t/N:>10.4f}  {ke_r/N:>10.4f}  {dE_arr[step]:>10.2e}  "
            f"Pzz={Pzz_atm:8.1f} atm  Lz={L_box[2]:7.2f} Å")
    if step in checkpoint_steps:
        rho_z_ck, _ = np.histogram(cm_pos[:, 2], bins=100, range=(0, L_box[2]))
        rho_z_ck = rho_z_ck / (L_box[0] * L_box[1] * dz)
        frac = checkpoint_fracs[checkpoint_steps.index(step)]
        rho_z_checkpoints.append((step, f"{int(frac*100)}%", rho_z_ck))

    if step % snap_freq == 0:
        O, H1, H2 = get_atom_positions(cm_pos, quats, p['r_h1'], p['r_h2'])
        snaps_O.append(O.copy() % L_box)
        snaps_H1.append(H1.copy() % L_box)
        snaps_H2.append(H2.copy() % L_box)
        snaps_box.append(L_box.copy())

    if not np.isfinite(te) or abs(te) > 1e12:
        print(f"  DIVERGED at step {step}")
        te_arr[step:] = np.nan; pe_arr[step:] = np.nan
        break

rho_z_final, _ = np.histogram(cm_pos[:, 2], bins=100, range=(0, L_box[2]))
rho_z_final = rho_z_final / (L_box[0] * L_box[1] * dz)

# ─────────────────────────────────────────────
#  Static plots
# ─────────────────────────────────────────────
T_target = T_K_list
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

ax_T.plot(t_fs, T_arr,    color='#ff9944', lw=1.5, label='T inst')
ax_T.plot(t_fs, T_target, color='#ffffff', lw=1.0, linestyle='--', label='T cible')
ax_T.set(title='Température instantanée', xlabel='Time [fs]', ylabel='K')
ax_T.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

fig.suptitle(f'SPC/E Water  |  LJ + Coulomb  |  Velocity Verlet  |  dt={dt_fs} fs  |  N={N}',
             color='#ffffff', fontsize=12, fontweight='bold')
plt.tight_layout()

_rho_panels = [('initial', '#4ecdc4', rho_z_init, 0)] + \
              [(lbl, '#ffdd00', ck, st) for st, lbl, ck in rho_z_checkpoints] + \
              [(f'final (T={T_end}K)', '#ff6b6b', rho_z_final, n_steps - 1)]
n_panels = len(_rho_panels)
fig2, _axes_rho = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
if n_panels == 1:
    _axes_rho = [_axes_rho]
fig2.patch.set_facecolor('#111118')
for _ax, (lbl, col, rho_z, st) in zip(_axes_rho, _rho_panels):
    style_ax(_ax)
    _ax.plot(z_centers, rho_z, color=col, lw=1.5)
    _ax.axhline(rho, color='#ffffff', lw=1.0, linestyle='--', label='ρ bulk')
    t_label = f"  (t={t_fs[min(st, n_steps-1)]:.0f} fs)" if st > 0 else ''
    _ax.set(title=f'Profil de densité — {lbl}{t_label}', xlabel='z [Å]', ylabel='ρ [mol/Å³]')
    _ax.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)

fig2.suptitle('Profil de densité en z — interface liquide-vapeur',
              color='#ffffff', fontsize=12, fontweight='bold')
plt.tight_layout()

fig3, ax_cv = plt.subplots(figsize=(10, 4))
fig3.patch.set_facecolor('#111118')
style_ax(ax_cv)
valid_cv = np.isfinite(cv_arr) & (cv_arr > 0)
ax_cv.scatter(T_arr[valid_cv], cv_arr[valid_cv], s=4, color='#aaffcc', alpha=0.6, label='Cv (fluctuations NVT)')
ax_cv.axhline(3 * kB_kcal, color='#ffffff', lw=1.0, ls='--', label='Cv idéal 3kB (équipartition)')
ax_cv.set(title='Capacité calorifique Cv/molécule (fenêtre W=200 steps, T stable seulement)',
          xlabel='T [K]', ylabel='kcal/(mol·K)')
ax_cv.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)
fig3.suptitle(f'SPC/E  |  N={N}  |  NVT Nosé-Hoover', color='#ffffff', fontsize=11)
plt.tight_layout()

# ─────────────────────────────────────────────
#  Save plots + raw data
# ─────────────────────────────────────────────
import os
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)

base = (f"{run_label}_N{N}_rho{rho:.4f}_Ts{T_start}-Te{T_end}"
        f"_dt{dt_fs}fs_mode{_mode}_phase{_phase}")

fig.savefig( os.path.join(results_dir, f"{base}_energies.png"), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(results_dir, f"{base}_density.png"),  dpi=150, bbox_inches='tight')
fig3.savefig(os.path.join(results_dir, f"{base}_cv.png"),       dpi=150, bbox_inches='tight')

_ck_steps  = np.array([st  for st, _, _  in rho_z_checkpoints])
_ck_labels = np.array([lbl for _,  lbl, _ in rho_z_checkpoints], dtype=object)
_ck_arrays = np.array([ck  for _,  _, ck  in rho_z_checkpoints])
np.savez(os.path.join(results_dir, f"{base}_data.npz"),
         te_arr=te_arr, pe_arr=pe_arr, ket_arr=ket_arr, ker_arr=ker_arr,
         fnorm_arr=fnorm_arr, dE_arr=dE_arr, T_arr=T_arr, cv_arr=cv_arr,
         t_fs=t_fs, T_K_list=T_K_list,
         rho_z_init=rho_z_init, rho_z_final=rho_z_final, z_centers=z_centers,
         snaps_box=np.array(snaps_box),
         rho_z_checkpoints=_ck_arrays,
         checkpoint_steps=_ck_steps,
         checkpoint_labels=_ck_labels)
print(f"Saved → results/{base}_{{energies,density,cv}}.png + _data.npz")

# ─────────────────────────────────────────────
#  3D Animation  (O=red, H=white)
# ─────────────────────────────────────────────
n_frames = len(snaps_O)
fig_anim = plt.figure(figsize=(9, 8))
fig_anim.patch.set_facecolor('#111118')
ax3 = fig_anim.add_subplot(111, projection='3d')
style_ax(ax3, three_d=True)
ax3.set_xlim(0, snaps_box[0][0]); ax3.set_ylim(0, snaps_box[0][1]); ax3.set_zlim(0, snaps_box[0][2])
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
    box = snaps_box[frame]
    ax3.set_xlim(0, box[0]); ax3.set_ylim(0, box[1]); ax3.set_zlim(0, box[2])
    tt.set_text(f't = {frame * snap_freq * dt_fs:.1f} fs  |  L = {box[0]:.1f} Å')

ani = animation.FuncAnimation(fig_anim, update, frames=n_frames, interval=60, blit=False)
plt.tight_layout()

gif_name = f"{base}.gif"
gif_path = os.path.join(results_dir, gif_name)
print(f"Saving animation → {gif_path}")
ani.save(gif_path, writer='pillow', fps=15)
print("Done.")
plt.show()
