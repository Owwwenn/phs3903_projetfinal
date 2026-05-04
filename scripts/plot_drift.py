"""
Exporte le graphique |ΔE/E₀| (drift) pour les runs ewald NVE gas et liquid.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

RUNS = {
    'gas':    'ewald_nve_gas_N256_rho0.0000_Ts300-Te300_dt1.0fs_modeewald_phasegas_data.npz',
    'liquid': 'ewald_nve_liquid_N256_rho0.0334_Ts300-Te300_dt1.0fs_modeewald_phaseliquid_data.npz',
}

FIG_W, FIG_H = 8, 5

def style_ax(ax):
    ax.set_facecolor('#111118')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')
    ax.tick_params(colors='#ddddee', labelsize=9)
    ax.xaxis.label.set_color('#ddddee')
    ax.yaxis.label.set_color('#ddddee')
    ax.title.set_color('#ffffff')
    ax.grid(True, color='#2a2a40', lw=0.6, linestyle='--')


for label, fname in RUNS.items():
    d = np.load(os.path.join(RESULTS, fname))
    t_fs   = d['t_fs']
    dE_arr = d['dE_arr']

    valid = np.isfinite(dE_arr) & (dE_arr > 0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#111118')
    style_ax(ax)
    ax.semilogy(t_fs[valid], dE_arr[valid], color='#ff44cc', lw=1.5)
    ax.set(
        title=f'Dérive énergétique |ΔE/E₀| — Ewald NVE {label}',
        xlabel='Temps [fs]',
        ylabel='|ΔE/E₀|  (log)',
    )
    plt.tight_layout()

    out = os.path.join(RESULTS, fname.replace('_data.npz', '_drift.png'))
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")

print("Done.")
