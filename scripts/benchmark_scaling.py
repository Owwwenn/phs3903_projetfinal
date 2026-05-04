"""
Étude de performance de la simulation MD en fonction du nombre de molécules N.

Mesures par step :
  - build_neighbour_list (LJ + Coulomb)
  - velocity_verlet_step complet (inclut compute_forces_and_torques)
  - calculs d'énergie cinétique
  - overhead total

Ressources système (thread dédié) :
  - CPU moyen (psutil)
  - RAM RSS pic et moyenne

Figures produites (une par fichier) :
  fig1_temps_total.png      — temps/step vs N  (liquid + gas)
  fig2_temps_par_mol.png    — temps/step/molécule vs N
  fig3_nl_temps.png         — temps neighbour list seul vs N
  fig4_paires.png           — nombre de paires NL vs N
  fig5_breakdown_both.png   — décomposition stacked côte à côte (liquide & gaz), tous N
  fig6_pct_both.png         — % par étape côte à côte (liquide & gaz), N_ref
  fig7_cpu.png              — CPU moyen vs N
  fig8_ram.png              — RAM pic vs N
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import threading
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from md_sim.core.system import (
    get_site_offsets, make_initial_state, SPCE, T_UNIT_PS
)
from md_sim.core.potential_force.coul_LJ_opt import compute_forces_and_torques
from md_sim.core.neighbour_list.neighbour_list import build_neighbour_list_numba as build_neighbour_list
from md_sim.core.time_integrator.time_int_VECTORISED import velocity_verlet_step
from md_sim.caracterisation.energy import kinetic_energy_trans, kinetic_energy_rot

# ─────────────────────────────────────────────────────────────
#  Paramètres du benchmark
# ─────────────────────────────────────────────────────────────
N_VALUES      = [32, 64, 128, 256, 512, 1024]
PHASES        = ['liquid', 'gas']
RHO_LIQUID    = 0.0334       # molecules/Å³
RHO_GAS       = 2e-5         # molecules/Å³
T_K           = 300.0
DT_FS         = 1.0
N_WARMUP      = 3
N_BENCH       = 15
MODE          = 'ewald'
CPU_SAMPLE_HZ = 20

# ─────────────────────────────────────────────────────────────
#  Style graphique
# ─────────────────────────────────────────────────────────────
DARK_BG = '#111118'
C_LIQ   = '#00c8ff'
C_GAS   = '#ff7043'
C_NL    = '#ffd54f'
C_VV    = '#80cbc4'
C_KE    = '#ce93d8'
C_OTH   = '#607d8b'

def style(ax):
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444466')
    ax.tick_params(colors='#ccccdd', labelsize=9)
    ax.xaxis.label.set_color('#ccccdd')
    ax.yaxis.label.set_color('#ccccdd')
    ax.title.set_color('#ffffff')
    ax.grid(True, color='#2a2a40', lw=0.5, ls='--')

def new_fig(title='', w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    style(ax)
    if title:
        ax.set_title(title, color='#ffffff', fontsize=11)
    return fig, ax

def leg(ax):
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466',
              labelcolor='#ffffff', fontsize=9)

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {path}")

# ─────────────────────────────────────────────────────────────
#  Moniteur CPU / RAM  (thread séparé)
# ─────────────────────────────────────────────────────────────
class ResourceMonitor:
    def __init__(self, interval=1.0 / CPU_SAMPLE_HZ):
        self.proc     = psutil.Process(os.getpid())
        self.interval = interval
        self._running = False
        self._thread  = None
        self.cpu_samples = []
        self.ram_samples = []

    def start(self):
        self.cpu_samples = []
        self.ram_samples = []
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                self.cpu_samples.append(self.proc.cpu_percent(interval=None))
                self.ram_samples.append(self.proc.memory_info().rss / 1e6)
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    @property
    def mean_cpu(self):
        s = [x for x in self.cpu_samples if x > 0]
        return float(np.mean(s)) if s else 0.0

    @property
    def peak_ram(self):
        return float(max(self.ram_samples)) if self.ram_samples else 0.0

    @property
    def mean_ram(self):
        return float(np.mean(self.ram_samples)) if self.ram_samples else 0.0


# ─────────────────────────────────────────────────────────────
#  Benchmark d'une configuration (N, phase)
# ─────────────────────────────────────────────────────────────
def benchmark_config(N, phase, rho, mode=MODE, n_warmup=N_WARMUP, n_bench=N_BENCH):
    dt = (DT_FS * 1e-3) / T_UNIT_PS

    r_h1, r_h2 = get_site_offsets(SPCE['theta'], SPCE['r_oh'])
    p = {**SPCE, 'r_h1': r_h1, 'r_h2': r_h2}

    cm_pos, cm_vel, quats, L_body, L_box = make_initial_state(
        N, rho, T_K, p, seed=42, phase=phase
    )

    nl_LJ   = build_neighbour_list(cm_pos, L_box, p['rc_LJ'])
    nl_coul = build_neighbour_list(cm_pos, L_box, p['rc_coul'])
    nbr_list = [nl_LJ, nl_coul]

    forces, tau, pe, virial, virial_zz = compute_forces_and_torques(
        N, cm_pos, quats, L_box, nbr_list, p, mode
    )

    # État mutable transmis par référence via dict
    st = dict(
        cm_pos=cm_pos, cm_vel=cm_vel, quats=quats,
        L_body=L_body, forces=forces, tau=tau,
        xi_t=0.0, xi_r=0.0, s_t=0.0, s_r=0.0,
        virial=virial, virial_zz=virial_zz,
        eta_p=0.0, L_box=L_box.copy(), pe=pe,
        # dernier compte de paires (mis à jour dans one_step)
        n_pairs_LJ=len(nl_LJ[0]), n_pairs_coul=len(nl_coul[0]),
    )

    def one_step():
        t0 = time.perf_counter()

        # ── neighbour lists ────────────────────────────────────────
        t_nl0 = time.perf_counter()
        cur_nl_LJ   = build_neighbour_list(st['cm_pos'], st['L_box'], p['rc_LJ'])
        cur_nl_coul = build_neighbour_list(st['cm_pos'], st['L_box'], p['rc_coul'])
        cur_nbr     = [cur_nl_LJ, cur_nl_coul]
        t_nl = time.perf_counter() - t_nl0

        # stocker le dernier compte de paires dans le dict partagé
        st['n_pairs_LJ']   = len(cur_nl_LJ[0])
        st['n_pairs_coul'] = len(cur_nl_coul[0])

        # ── velocity verlet (forces incluses à l'intérieur) ────────
        t_vv0 = time.perf_counter()
        (st['cm_pos'], st['cm_vel'], st['quats'], st['L_body'],
         st['forces'], st['tau'], st['pe'],
         st['xi_t'], st['xi_r'], st['s_t'], st['s_r'],
         st['virial'], st['virial_zz'], st['eta_p'], st['L_box']
        ) = velocity_verlet_step(
            st['cm_pos'], st['cm_vel'], st['quats'], st['L_body'],
            st['forces'], st['tau'],
            p['mass'], p['I_body'], st['L_box'], p, dt, cur_nbr,
            xi_t=st['xi_t'], xi_r=st['xi_r'],
            s_t=st['s_t'], s_r=st['s_r'],
            n_mol=N, T=T_K, mode=mode,
            virial=st['virial'], virial_zz=st['virial_zz'],
            eta_p=st['eta_p'], Q_p=1.0, P0=1.0 / 6.947e4,
            barostat='none'
        )
        t_vv = time.perf_counter() - t_vv0

        # ── énergie cinétique ──────────────────────────────────────
        t_ke0 = time.perf_counter()
        kinetic_energy_trans(st['cm_vel'], p['mass'])
        kinetic_energy_rot(st['L_body'], p['I_body'])
        t_ke = time.perf_counter() - t_ke0

        t_total = time.perf_counter() - t0
        return dict(nl=t_nl, vv=t_vv, ke=t_ke, total=t_total)

    # Warmup
    print(f"    warmup ({n_warmup} steps)...", flush=True)
    for _ in range(n_warmup):
        one_step()

    # Bench avec monitoring
    monitor = ResourceMonitor()
    monitor.start()
    timings = defaultdict(list)
    print(f"    bench  ({n_bench} steps)...", flush=True)
    for _ in range(n_bench):
        t = one_step()
        for k, v in t.items():
            timings[k].append(v)
    monitor.stop()

    stats = {}
    for k, vals in timings.items():
        arr = np.array(vals)
        stats[k] = dict(
            mean=arr.mean(), std=arr.std(),
            min=arr.min(), max=arr.max(),
            per_molecule=arr.mean() / N,
        )

    return dict(
        N=N, phase=phase, rho=rho,
        stats=stats,
        mean_cpu=monitor.mean_cpu,
        peak_ram=monitor.peak_ram,
        mean_ram=monitor.mean_ram,
        n_pairs_LJ=st['n_pairs_LJ'],
        n_pairs_coul=st['n_pairs_coul'],
    )


# ─────────────────────────────────────────────────────────────
#  Lancer tous les benchmarks
# ─────────────────────────────────────────────────────────────
results = {}

for phase in PHASES:
    rho = RHO_LIQUID if phase == 'liquid' else RHO_GAS
    for N in N_VALUES:
        print(f"\n{'='*55}\n  N={N:>5}   phase={phase}   rho={rho:.2e}\n{'='*55}")
        try:
            res = benchmark_config(N, phase, rho)
            results[(N, phase)] = res
            s = res['stats']
            tot = s['total']['mean']
            print(f"  total      : {tot*1000:7.2f} ms  (±{s['total']['std']*1000:.2f})")
            print(f"  neigh list : {s['nl']['mean']*1000:7.2f} ms  ({s['nl']['mean']/tot*100:.1f}%)")
            print(f"  verlet+F   : {s['vv']['mean']*1000:7.2f} ms  ({s['vv']['mean']/tot*100:.1f}%)")
            print(f"  énergie KE : {s['ke']['mean']*1000:7.2f} ms  ({s['ke']['mean']/tot*100:.1f}%)")
            print(f"  CPU moy    : {res['mean_cpu']:5.1f}%    RAM pic : {res['peak_ram']:.1f} Mo")
        except Exception as exc:
            print(f"  ERREUR: {exc}")
            import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
#  Sauvegarde données brutes
# ─────────────────────────────────────────────────────────────
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'benchmark')
os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, 'benchmark_results.npy'), results)
print(f"\nDonnées → {results_dir}/benchmark_results.npy")

# ─────────────────────────────────────────────────────────────
#  Helpers pour extraire séries par phase
# ─────────────────────────────────────────────────────────────
def serie(phase, key_chain):
    Ns = sorted(k[0] for k in results if k[1] == phase)
    vals = []
    for N in Ns:
        r = results[(N, phase)]
        v = r
        for k in key_chain:
            v = v[k]
        vals.append(v)
    return Ns, vals

PHASE_STYLE = [
    ('liquid', C_LIQ, 'Liquide'),
    ('gas',    C_GAS, 'Gaz'),
]

# ─────────────────────────────────────────────────────────────
#  Fig 1 — Temps total / step vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig(f'Temps total / step vs N  (mode={MODE}, dt={DT_FS} fs)', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, means = serie(phase, ['stats', 'total', 'mean'])
    _, stds   = serie(phase, ['stats', 'total', 'std'])
    means_ms  = [v * 1000 for v in means]
    stds_ms   = [v * 1000 for v in stds]
    ax.errorbar(Ns, means_ms, yerr=stds_ms,
                marker='o', color=color, label=label, lw=2, capsize=5)
ax.set(xlabel='N (molécules)', ylabel='Temps (ms)')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig1_temps_total.png'))

# ─────────────────────────────────────────────────────────────
#  Fig 2 — Temps / step / molécule vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig('Temps / step / molécule vs N', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, vals = serie(phase, ['stats', 'total', 'per_molecule'])
    vals_ms  = [v * 1000 for v in vals]
    ax.plot(Ns, vals_ms, marker='s', color=color, label=label, lw=2)
ax.set(xlabel='N (molécules)', ylabel='ms / molécule')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig2_temps_par_mol.png'))

# ─────────────────────────────────────────────────────────────
#  Fig 3 — Temps neighbour list seul vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig('Temps neighbour list seul vs N', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, vals = serie(phase, ['stats', 'nl', 'mean'])
    ax.plot(Ns, [v * 1000 for v in vals],
            marker='^', color=color, label=label, lw=2)
ax.set(xlabel='N (molécules)', ylabel='Temps (ms)')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig3_nl_temps.png'))

# ─────────────────────────────────────────────────────────────
#  Fig 4 — Nombre de paires dans la NL vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig('Nombre de paires dans la neighbour list vs N', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, pairs_LJ   = serie(phase, ['n_pairs_LJ'])
    _, pairs_coul  = serie(phase, ['n_pairs_coul'])
    ax.plot(Ns, pairs_LJ,   marker='o', color=color, label=f'{label} — LJ', lw=2)
    ax.plot(Ns, pairs_coul, marker='x', color=color, ls='--',
            label=f'{label} — Coulomb', lw=1.5)
ax.set(xlabel='N (molécules)', ylabel='Nombre de paires')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig4_paires.png'))

# ─────────────────────────────────────────────────────────────
#  Fig 5 — Décomposition stacked, liquide ET gaz côte à côte
# ─────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches

def stacked_breakdown_both(fname):
    Ns = sorted(set(k[0] for k in results))
    x  = np.arange(len(Ns))
    w  = 0.35

    fig, ax = new_fig(
        f'Décomposition temps/step — Liquide vs Gaz  (mode={MODE})', w=11, h=5
    )

    layers = [
        ('nl',  C_NL,  'Neighbour list'),
        ('vv',  C_VV,  'Verlet + Forces'),
        ('ke',  C_KE,  'Énergie cinétique'),
        (None,  C_OTH, 'Autre'),
    ]

    for offset, phase, hatch in [(-w / 2, 'liquid', ''), (w / 2, 'gas', '//')]:
        b0 = np.zeros(len(Ns))
        for key, color, _ in layers:
            if key is not None:
                vals = np.array([
                    results[(N, phase)]['stats'][key]['mean'] * 1000
                    for N in Ns
                ])
            else:
                tots = np.array([results[(N, phase)]['stats']['total']['mean'] * 1000 for N in Ns])
                nls  = np.array([results[(N, phase)]['stats']['nl']['mean']    * 1000 for N in Ns])
                vvs  = np.array([results[(N, phase)]['stats']['vv']['mean']    * 1000 for N in Ns])
                kes  = np.array([results[(N, phase)]['stats']['ke']['mean']    * 1000 for N in Ns])
                vals = np.maximum(tots - nls - vvs - kes, 0)
            ax.bar(x + offset, vals, w, bottom=b0, color=color,
                   hatch=hatch, edgecolor='#ccccdd', linewidth=0.4)
            b0 += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set(xlabel='N (molécules)', ylabel='Temps (ms)')

    # Légende : 4 patches couleur (étapes) + 2 patches hatch (phases)
    step_patches = [mpatches.Patch(color=c, label=lbl) for _, c, lbl in layers]
    phase_patches = [
        mpatches.Patch(facecolor='#888888', hatch='',   edgecolor='#ccccdd', label='Liquide'),
        mpatches.Patch(facecolor='#888888', hatch='//', edgecolor='#ccccdd', label='Gaz'),
    ]
    ax.legend(handles=step_patches + phase_patches,
              facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ffffff', fontsize=8)
    savefig(fig, os.path.join(results_dir, fname))

stacked_breakdown_both('fig5_breakdown_both.png')

# ─────────────────────────────────────────────────────────────
#  Fig 6 — % par étape N_ref, liquide ET gaz côte à côte
# ─────────────────────────────────────────────────────────────
N_ref = max(k[0] for k in results)

def pct_bars_both(fname):
    cat_labels = ['Neighbour\nlist', 'Verlet\n+ Forces', 'Énergie\ncinétique', 'Autre']
    cat_keys   = ['nl', 'vv', 'ke', None]
    cat_colors = [C_NL, C_VV, C_KE, C_OTH]
    x = np.arange(len(cat_labels))
    w = 0.35

    fig, ax = new_fig(
        f'% temps par étape — Liquide vs Gaz  (N={N_ref}, mode={MODE})', w=9, h=5
    )

    for offset, phase, label, hatch in [(-w / 2, 'liquid', 'Liquide', ''),
                                         ( w / 2, 'gas',    'Gaz',    '//')]:
        if (N_ref, phase) not in results:
            continue
        s   = results[(N_ref, phase)]['stats']
        tot = s['total']['mean']
        pcts = []
        for key in cat_keys:
            if key is not None:
                pcts.append(s[key]['mean'] / tot * 100)
            else:
                rest = max(tot - s['nl']['mean'] - s['vv']['mean'] - s['ke']['mean'], 0)
                pcts.append(rest / tot * 100)

        for i, (pct, color) in enumerate(zip(pcts, cat_colors)):
            bar = ax.bar(x[i] + offset, pct, w, color=color,
                         hatch=hatch, edgecolor='#ccccdd', linewidth=0.4,
                         label=label if i == 0 else '_nolegend_')
            ax.text(x[i] + offset, pct + 0.5, f'{pct:.1f}%',
                    ha='center', va='bottom', color='#ffffff', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel('% du temps total')
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)

    step_patches  = [mpatches.Patch(color=c, label=lbl) for c, lbl in zip(cat_colors, ['NL', 'Verlet+F', 'KE', 'Autre'])]
    phase_patches = [
        mpatches.Patch(facecolor='#888888', hatch='',   edgecolor='#ccccdd', label='Liquide'),
        mpatches.Patch(facecolor='#888888', hatch='//', edgecolor='#ccccdd', label='Gaz'),
    ]
    ax.legend(handles=step_patches + phase_patches,
              facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ffffff', fontsize=8)
    savefig(fig, os.path.join(results_dir, fname))

pct_bars_both('fig6_pct_both.png')

# ─────────────────────────────────────────────────────────────
#  Fig 7 — CPU moyen vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig('Consommation CPU moyenne vs N', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, vals = serie(phase, ['mean_cpu'])
    ax.plot(Ns, vals, marker='D', color=color, label=label, lw=2)
ax.set(xlabel='N (molécules)', ylabel='CPU moyen (%)')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig7_cpu.png'))

# ─────────────────────────────────────────────────────────────
#  Fig 8 — RAM pic vs N
# ─────────────────────────────────────────────────────────────
fig, ax = new_fig('RAM (pic RSS) vs N', w=8, h=5)
for phase, color, label in PHASE_STYLE:
    Ns, vals = serie(phase, ['peak_ram'])
    ax.plot(Ns, vals, marker='P', color=color, label=label, lw=2)
ax.set(xlabel='N (molécules)', ylabel='RAM pic (Mo)')
leg(ax)
savefig(fig, os.path.join(results_dir, 'fig8_ram.png'))

# ─────────────────────────────────────────────────────────────
#  Tableau récapitulatif
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print(f"{'N':>6}  {'Phase':>7}  {'Total ms':>9}  {'NL %':>6}  {'VV+F %':>7}  "
      f"{'KE %':>6}  {'CPU %':>6}  {'RAM Mo':>8}  {'Paires LJ':>10}  {'Paires C':>10}")
print("-" * 95)
for phase in PHASES:
    for N in N_VALUES:
        if (N, phase) not in results:
            continue
        r   = results[(N, phase)]
        s   = r['stats']
        tot = s['total']['mean'] * 1000
        pct = lambda k: s[k]['mean'] / s['total']['mean'] * 100  # noqa: E731
        print(f"{N:>6}  {phase:>7}  {tot:>9.2f}  {pct('nl'):>6.1f}  "
              f"{pct('vv'):>7.1f}  {pct('ke'):>6.1f}  "
              f"{r['mean_cpu']:>6.1f}  {r['peak_ram']:>8.1f}  "
              f"{r['n_pairs_LJ']:>10}  {r['n_pairs_coul']:>10}")
print("=" * 95)
print(f"\nBenchmark terminé. Figures → {results_dir}/")
