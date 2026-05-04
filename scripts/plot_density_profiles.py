"""
Lit un fichier _data.npz et sauvegarde chaque profil de densité en z
dans un fichier PNG séparé.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Fichier à analyser ──────────────────────────────────────────────────────
NPZ = os.path.join(os.path.dirname(__file__), '..', 'results',
                   '30012005k5k5kUDv2_N256_rho0.0334_Ts300-Te1200_dt1.0fs_modeewald_phaseliquid_data.npz')

d = np.load(NPZ, allow_pickle=True)

z_centers        = d['z_centers']
rho_z_init       = d['rho_z_init']
rho_z_final      = d['rho_z_final']
rho_z_checkpoints = d['rho_z_checkpoints']   # shape (n_ck, n_bins)
checkpoint_labels = d['checkpoint_labels']
checkpoint_steps  = d['checkpoint_steps']
t_fs             = d['t_fs']
T_K_list         = d['T_K_list']
n_steps          = len(t_fs)

# rho bulk = moyenne de rho_z_init sur la région liquide
rho_bulk = 0.0334   # approx mid-density reference

COLORS_CK = ['#ffdd00', '#ffaa00', '#ff7700', '#ff4400',
             '#ff0088', '#cc00ff', '#7700ff', '#0044ff', '#00aaff']

def style_ax(ax):
    ax.set_facecolor('#111118')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')
    ax.tick_params(colors='#ddddee', labelsize=9)
    ax.xaxis.label.set_color('#ddddee')
    ax.yaxis.label.set_color('#ddddee')
    ax.title.set_color('#ffffff')
    ax.grid(True, color='#2a2a40', lw=0.6, linestyle='--')


FIG_W, FIG_H = 6, 4   # ratio commun à tous les graphiques

def save_profile(z, rho_z, rho_ref, color, title, fname):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#111118')
    style_ax(ax)
    ax.plot(z, rho_z, color=color, lw=1.8)
    ax.axhline(rho_ref, color='#ffffff', lw=1.0, linestyle='--', label='ρ bulk')
    ax.set(title=title, xlabel='z [Å]', ylabel='ρ [molécules/Å³]')
    ax.legend(facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#ffffff', fontsize=8)
    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
base = os.path.splitext(os.path.basename(NPZ))[0].replace('_data', '')

# ── Profil initial ──────────────────────────────────────────────────────────
save_profile(
    z_centers, rho_z_init, rho_bulk,
    color='#4ecdc4',
    title='Profil de densité — initial  (t=0 fs)',
    fname=os.path.join(results_dir, f"{base}_density_00_initial.png")
)

# ── Checkpoints ─────────────────────────────────────────────────────────────
for i, (lbl, st, rho_z) in enumerate(zip(checkpoint_labels, checkpoint_steps, rho_z_checkpoints)):
    t_label = f"t={t_fs[min(st, n_steps-1)]:.0f} fs"
    T_label = f"T={T_K_list[min(st, n_steps-1)]:.0f} K"
    col = COLORS_CK[i % len(COLORS_CK)]
    save_profile(
        z_centers, rho_z, rho_bulk,
        color=col,
        title=f'Profil de densité — {lbl}  ({t_label}, {T_label})',
        fname=os.path.join(results_dir, f"{base}_density_{i+1:02d}_{lbl}.png")
    )

# ── Profil final ─────────────────────────────────────────────────────────────
T_end = int(T_K_list[-1])
save_profile(
    z_centers, rho_z_final, rho_bulk,
    color='#ff6b6b',
    title=f'Profil de densité — final  (T={T_end} K)',
    fname=os.path.join(results_dir, f"{base}_density_{len(checkpoint_labels)+1:02d}_final.png")
)

print("Done.")
