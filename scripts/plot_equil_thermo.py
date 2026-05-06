import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "results/test_equil_thermo_liquid_300_N256_rho0.0334_Ts300-Te300_dt1.0fs_modeewald_phasegas_data.npz"
T_TARGET = 300.0

data = np.load(NPZ_PATH, allow_pickle=True)

T_arr = data["T_arr"]       # Tinst (K)
t_fs  = data["t_fs"]        # temps (fs)
steps = np.arange(len(T_arr))

T_err = T_arr - T_TARGET    # erreur absolue

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
# fig.suptitle(f"Équilibration thermostatique — liquide, $T_{{\\rm target}}$ = {T_TARGET:.0f} K", fontsize=13)

# --- panneau 1 : Tinst vs time step ---
ax1 = axes[0]
ax1.plot(steps, T_arr, lw=0.6, alpha=0.8, color="steelblue", label="$T_{\\rm inst}$")
ax1.axhline(T_TARGET, color="crimson", ls="--", lw=1.2, label=f"$T_{{\\rm target}}$ = {T_TARGET:.0f} K")
ax1.set_ylabel("$T_{\\rm inst}$ (K)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- panneau 2 : erreur de T vs time step ---
ax2 = axes[1]
ax2.semilogy(steps, T_err, lw=0.6, alpha=0.8, color="darkorange", label="$T_{\\rm inst} - T_{\\rm target}$")
ax2.axhline(0, color="crimson", ls="--", lw=1.2)
ax2.set_xlabel("Time step", fontsize=11)
ax2.set_ylabel("$\\Delta T$ (K)", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# axe secondaire en fs
def steps_to_fs(x):
    return x  # dt = 1 fs donc time step = temps en fs

ax1_top = ax1.secondary_xaxis("top", functions=(steps_to_fs, steps_to_fs))
ax1_top.set_xlabel("Temps (fs)", fontsize=10)

plt.tight_layout()
out = "results/test_equil_thermo_liquid2_300_Tinst_erreur.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure sauvegardée : {out}")
plt.show()
