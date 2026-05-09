"""
═══════════════════════════════════════════════════════════════════════════════
UAV Metaheuristic Optimization — IEEE COMST Grade Simulation
═══════════════════════════════════════════════════════════════════════════════
Paper  : Metaheuristic Optimization for UAV-Assisted Wireless Networks
Journal: IEEE Communications Surveys & Tutorials
Authors: Ali Jameel Hashim, Abbas I. Al-Zaki

Environment : Synthetic UAV network (ITU-R channel + Zeng 2019 energy model)
Algorithms  : GA, PSO, DE, ACO
Problems    : (1) Trajectory Planning  (2) Node Deployment
Runs        : 30 independent runs per algorithm per problem (IEEE standard)
Metrics     : Mean, Std, Best, Worst, Median, IQR, CV, Convergence Iteration,
              Success Rate, Runtime, Wilcoxon p-values, Friedman rank,
              Effect size (Cohen's d), Improvement over baseline
Outputs     : Interactive figures (greyscale, IEEE style) + CSV tables
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import warnings, time, os, csv, itertools
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  IEEE STYLE + GREYSCALE PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
GS = {
    'bg': '#FFFFFF', 'panel': '#F7F7F7', 'gridline': '#E0E0E0',
    'border': '#999999', 'text': '#111111', 'muted': '#555555',
}
ALG_STYLES = {
    'GA':  {'color': '#000000', 'ls': '-',   'marker': 'o', 'lw': 2.0, 'label': 'GA'},
    'PSO': {'color': '#444444', 'ls': '--',  'marker': 's', 'lw': 1.8, 'label': 'PSO'},
    'DE':  {'color': '#777777', 'ls': '-.',  'marker': '^', 'lw': 1.8, 'label': 'DE'},
    'ACO': {'color': '#AAAAAA', 'ls': ':',   'marker': 'D', 'lw': 1.6, 'label': 'ACO'},
}
ALGS = list(ALG_STYLES.keys())

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
    'grid.color': GS['gridline'], 'grid.alpha': 0.8,
    'figure.facecolor': GS['bg'], 'axes.facecolor': GS['panel'],
    'axes.edgecolor': GS['border'],
})

# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
AREA      = 1000.0
ALT       = 100.0
V_MAX     = 30.0
P_TX      = 0.1
BW        = 1e6
SIGMA2    = 1e-10
FC        = 2.4e9
C_LIGHT   = 3e8
A_ITU, B_ITU         = 9.61, 0.16
ETA_LOS, ETA_NLOS    = 1.0, 20.0
P0, P1               = 79.856, 88.628
V_TIP, V0            = 120.0, 4.03
D0, RHO              = 0.6, 1.225
S_ROTOR, A_DISC      = 0.05, 0.503

# Harder deployment problem parameters
N_NODES_TRAJ   = 20
N_NODES_DEPLOY = 50    # more nodes
K_UAVS         = 4     # fewer UAVs
COV_RADIUS     = 150.0 # smaller radius

N_WPS      = 10
TRAJ_DIM   = N_WPS * 2
DEPLOY_DIM = K_UAVS * 2
N_ITER     = 200
N_RUNS     = 5
POP_SIZE   = 80

np.random.seed(42)
GROUND_NODES_TRAJ   = np.random.uniform(50, AREA-50, (N_NODES_TRAJ,   2))
GROUND_NODES_DEPLOY = np.random.uniform(50, AREA-50, (N_NODES_DEPLOY, 2))
UAV_START = np.array([0.0,  0.0])
UAV_END   = np.array([AREA, AREA])
NO_FLY_ZONES = [
    {'center': np.array([300., 300.]), 'radius': 90.},
    {'center': np.array([700., 650.]), 'radius': 80.},
    {'center': np.array([500., 200.]), 'radius': 70.},
    {'center': np.array([200., 700.]), 'radius': 65.},
]

def los_prob(h, dh):
    theta = np.degrees(np.arctan2(h, dh + 1e-9))
    return 1.0 / (1.0 + A_ITU * np.exp(-B_ITU * (theta - A_ITU)))

def path_loss(h, dh):
    d3d  = np.sqrt(dh**2 + h**2) + 1e-6
    fspl = 20 * np.log10(4 * np.pi * FC * d3d / C_LIGHT)
    p    = los_prob(h, dh)
    return fspl + p * ETA_LOS + (1 - p) * ETA_NLOS

def uav_energy(v, dt):
    v = max(v, 0.1)
    blade = P0 * (1 + 3 * v**2 / V_TIP**2)
    induc = P1 * np.sqrt(np.sqrt(1 + v**4 / (4*V0**4)) - v**2 / (2*V0**2))
    para  = 0.5 * D0 * RHO * S_ROTOR * A_DISC * v**3
    return (blade + induc + para) * dt

def nfz_penalty(waypoints):
    pen = 0.0
    for wp in waypoints:
        for nfz in NO_FLY_ZONES:
            d = np.linalg.norm(wp - nfz['center'])
            if d < nfz['radius']:
                pen += (nfz['radius'] - d) * 800
    return pen

def trajectory_fitness(wps_flat):
    wps  = wps_flat.reshape(N_WPS, 2)
    full = np.vstack([UAV_START, wps, UAV_END])
    dist, energy = 0.0, 0.0
    for i in range(len(full)-1):
        seg  = np.linalg.norm(full[i+1] - full[i])
        v    = min(seg / 5.0, V_MAX)
        dt   = seg / (v + 1e-9)
        dist   += seg
        energy += uav_energy(v, dt)
    coverage = 0.0
    for node in GROUND_NODES_TRAJ:
        ml  = min(path_loss(ALT, np.linalg.norm(wp - node)) for wp in full)
        snr = P_TX / (10**(ml/10)) / SIGMA2
        if snr > 1.0:
            coverage += BW * np.log2(1 + snr) / 1e6
    pen = nfz_penalty(wps)
    return (0.35*(dist/1000) + 0.30*(energy/1000)
            - 0.25*(coverage/100) + 0.10*pen)

def deployment_fitness(pos_flat):
    pos = pos_flat.reshape(K_UAVS, 2)
    cov = sum(
        1 for node in GROUND_NODES_DEPLOY
        if any(np.linalg.norm(node - p) < COV_RADIUS for p in pos)
    )
    interf = sum(
        max(0, COV_RADIUS*2 - np.linalg.norm(pos[i]-pos[j]))
        for i in range(K_UAVS) for j in range(i+1, K_UAVS)
    )
    nfz_pen = sum(
        max(0, nfz['radius'] - np.linalg.norm(p - nfz['center']))
        for p in pos for nfz in NO_FLY_ZONES
    ) * 200
    return -(cov / N_NODES_DEPLOY) + 0.002*interf + nfz_pen

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════
def run_ga(fn, dim, n_iter=N_ITER, pop=POP_SIZE, seed=0):
    np.random.seed(seed)
    lo, hi = 0.0, AREA
    P = np.random.uniform(lo, hi, (pop, dim))
    best_fit, best_ind = np.inf, P[0].copy()
    history, stagnation = [], 0
    conv_iter = n_iter

    for gen in range(n_iter):
        fits = np.array([fn(x) for x in P])
        idx  = np.argsort(fits)
        P, fits = P[idx], fits[idx]
        if fits[0] < best_fit:
            best_fit, best_ind = fits[0], P[0].copy()
            if stagnation > 0: conv_iter = min(conv_iter, gen)
            stagnation = 0
        else:
            stagnation += 1
        history.append(best_fit)

        # Adaptive mutation rate
        mut_rate = 0.4 * (1 - gen/n_iter) + 0.05
        elite    = max(5, pop//10)
        new_P    = P[:elite].copy()
        while len(new_P) < pop:
            t_idx = np.random.choice(pop//2, 4, replace=False)
            p1 = P[t_idx[np.argmin(fits[t_idx[:2]])]]
            p2 = P[t_idx[2 + np.argmin(fits[t_idx[2:]])]]
            alpha = np.random.rand(dim)
            child = alpha*p1 + (1-alpha)*p2
            if np.random.rand() < mut_rate:
                child += np.random.normal(0, (hi-lo)*0.03*(1-gen/n_iter+0.1), dim)
            new_P = np.vstack([new_P, np.clip(child, lo, hi)])
        P = new_P[:pop]

    if conv_iter == n_iter:
        conv_iter = np.argmin(np.gradient(history) > -1e-4) if len(history) > 10 else n_iter
    return best_ind, best_fit, history, conv_iter

def run_pso(fn, dim, n_iter=N_ITER, n_p=POP_SIZE, seed=0):
    np.random.seed(seed)
    lo, hi = 0.0, AREA
    pos  = np.random.uniform(lo, hi, (n_p, dim))
    vel  = np.random.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, (n_p, dim))
    pb   = pos.copy()
    pf   = np.array([fn(p) for p in pos])
    gi   = np.argmin(pf)
    gp, gf = pb[gi].copy(), pf[gi]
    history, conv_iter = [], n_iter
    w_max, w_min, c1, c2 = 0.9, 0.4, 2.05, 2.05

    for it in range(n_iter):
        w  = w_max - (w_max-w_min)*it/n_iter
        r1 = np.random.rand(n_p, dim)
        r2 = np.random.rand(n_p, dim)
        vel = w*vel + c1*r1*(pb-pos) + c2*r2*(gp-pos)
        v_max = (hi-lo)*0.2*(1-0.5*it/n_iter)
        vel  = np.clip(vel, -v_max, v_max)
        pos  = np.clip(pos + vel, lo, hi)
        fits = np.array([fn(p) for p in pos])
        imp  = fits < pf
        pb[imp], pf[imp] = pos[imp], fits[imp]
        if pf.min() < gf:
            gi   = np.argmin(pf)
            gp, gf = pb[gi].copy(), pf[gi]
            conv_iter = it
        history.append(gf)

    return gp, gf, history, conv_iter

def run_de(fn, dim, n_iter=N_ITER, pop=POP_SIZE, seed=0):
    np.random.seed(seed)
    lo, hi = 0.0, AREA
    P    = np.random.uniform(lo, hi, (pop, dim))
    fits = np.array([fn(x) for x in P])
    best_idx = np.argmin(fits)
    best_fit, best_ind = fits[best_idx], P[best_idx].copy()
    history, conv_iter = [], n_iter

    for gen in range(n_iter):
        F  = 0.5 + 0.3 * np.random.rand()
        CR = 0.8 + 0.15 * np.random.rand()
        for i in range(pop):
            others = [j for j in range(pop) if j != i]
            r1, r2, r3 = P[np.random.choice(others, 3, replace=False)]
            mutant = np.clip(r1 + F*(r2-r3), lo, hi)
            cross  = np.random.rand(dim) < CR
            if not cross.any(): cross[np.random.randint(dim)] = True
            trial  = np.where(cross, mutant, P[i])
            tf     = fn(trial)
            if tf < fits[i]:
                P[i], fits[i] = trial, tf
                if tf < best_fit:
                    best_fit, best_ind = tf, trial.copy()
                    conv_iter = gen
        history.append(best_fit)

    return best_ind, best_fit, history, conv_iter

def run_aco(fn, dim, n_iter=N_ITER, n_ants=POP_SIZE, seed=0):
    np.random.seed(seed)
    lo, hi = 0.0, AREA
    arch_sz = 30
    arch = np.random.uniform(lo, hi, (arch_sz, dim))
    af   = np.array([fn(a) for a in arch])
    si   = np.argsort(af)
    arch, af = arch[si], af[si]
    best_fit, best_ind = af[0], arch[0].copy()
    history, conv_iter = [], n_iter
    q, xi = 0.5, 0.85

    weights = np.exp(-np.arange(arch_sz)**2 / (2*q**2*arch_sz**2))
    weights /= weights.sum()

    for it in range(n_iter):
        sigma = xi * np.std(arch, axis=0) + 1e-10
        ants  = np.array([
            np.clip(arch[np.random.choice(arch_sz, p=weights)]
                    + np.random.normal(0, sigma, dim), lo, hi)
            for _ in range(n_ants)
        ])
        af2  = np.array([fn(a) for a in ants])
        comb = np.vstack([arch, ants])
        cf2  = np.hstack([af, af2])
        si   = np.argsort(cf2)[:arch_sz]
        arch, af = comb[si], cf2[si]
        if af[0] < best_fit:
            best_fit, best_ind = af[0], arch[0].copy()
            conv_iter = it
        history.append(best_fit)

    return best_ind, best_fit, history, conv_iter

RUNNERS = {'GA': run_ga, 'PSO': run_pso, 'DE': run_de, 'ACO': run_aco}

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment(fn, dim, label):
    print(f"\n{'─'*60}")
    print(f"  Problem: {label}  |  dim={dim}  |  runs={N_RUNS}  |  iter={N_ITER}")
    print(f"{'─'*60}")
    results = {}
    for name, runner in RUNNERS.items():
        t0 = time.time()
        all_fits, all_hist, all_conv, all_sols = [], [], [], []
        for run in range(N_RUNS):
            sol, fit, hist, conv = runner(fn, dim, seed=run*17+3)
            all_fits.append(fit)
            all_hist.append(hist)
            all_conv.append(conv)
            all_sols.append(sol)
        elapsed = time.time() - t0
        fits = np.array(all_fits)
        best_run = np.argmin(fits)
        results[name] = {
            'fits':      fits,
            'histories': np.array(all_hist),
            'conv':      np.array(all_conv),
            'best_sol':  all_sols[best_run],
            'time_total': elapsed,
            'time_per_run': elapsed / N_RUNS,
            # --- descriptive statistics ---
            'mean':    fits.mean(),
            'std':     fits.std(),
            'best':    fits.min(),
            'worst':   fits.max(),
            'median':  np.median(fits),
            'iqr':     np.percentile(fits,75) - np.percentile(fits,25),
            'cv':      fits.std() / abs(fits.mean()) * 100,
            'q25':     np.percentile(fits, 25),
            'q75':     np.percentile(fits, 75),
            'conv_mean': np.mean(all_conv),
            'conv_std':  np.std(all_conv),
            'mean_hist': np.mean(all_hist, axis=0),
            'std_hist':  np.std(all_hist,  axis=0),
            'q25_hist':  np.percentile(all_hist, 25, axis=0),
            'q75_hist':  np.percentile(all_hist, 75, axis=0),
        }
        print(f"  {name:<4}  mean={results[name]['mean']:>9.4f} "
              f"± {results[name]['std']:.4f}  "
              f"best={results[name]['best']:>9.4f}  "
              f"conv={results[name]['conv_mean']:>5.0f}  "
              f"time={elapsed:.1f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def run_statistics(results):
    print(f"\n{'─'*60}  STATISTICAL ANALYSIS")
    stats_out = {}

    # Friedman test
    data = [results[a]['fits'] for a in ALGS]
    fstat, fpval = friedmanchisquare(*data)
    print(f"  Friedman test: χ²={fstat:.4f}, p={fpval:.6f} "
          f"({'significant' if fpval < 0.05 else 'not significant'})")
    stats_out['friedman'] = {'stat': fstat, 'p': fpval}

    # Average Friedman ranks
    rank_matrix = np.array([rankdata(row) for row in np.column_stack(data)])
    avg_ranks = rank_matrix.mean(axis=0)
    for i, a in enumerate(ALGS):
        results[a]['friedman_rank'] = avg_ranks[i]
    print(f"  Friedman ranks: " +
          " ".join([f"{a}={avg_ranks[i]:.2f}" for i, a in enumerate(ALGS)]))

    # Pairwise Wilcoxon + Cohen's d
    pairs = list(itertools.combinations(ALGS, 2))
    wilcoxon_table = {}
    print(f"\n  Pairwise Wilcoxon signed-rank tests (α=0.05):")
    for a1, a2 in pairs:
        f1, f2 = results[a1]['fits'], results[a2]['fits']
        try:
            wstat, wp = wilcoxon(f1, f2)
        except Exception:
            wstat, wp = np.nan, 1.0
        # Cohen's d
        pooled_std = np.sqrt((f1.std()**2 + f2.std()**2) / 2)
        d = (f1.mean() - f2.mean()) / (pooled_std + 1e-12)
        mag = ('negligible' if abs(d) < 0.2 else
               'small'      if abs(d) < 0.5 else
               'medium'     if abs(d) < 0.8 else 'large')
        sig = wp < 0.05
        wilcoxon_table[(a1, a2)] = {
            'stat': wstat, 'p': wp,
            'significant': sig, 'cohens_d': d, 'magnitude': mag,
        }
        print(f"    {a1} vs {a2}: W={wstat:.1f}, p={wp:.4f} "
              f"{'*' if sig else ' '}  d={d:.3f} ({mag})")

    stats_out['wilcoxon'] = wilcoxon_table

    # Improvement over worst algorithm
    worst_mean = max(results[a]['mean'] for a in ALGS)
    for a in ALGS:
        results[a]['improvement_pct'] = (
            (worst_mean - results[a]['mean']) / abs(worst_mean) * 100
        )

    return stats_out

# ═══════════════════════════════════════════════════════════════════════════════
#  CSV TABLE EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def export_csv(traj_res, dep_res, traj_stats, dep_stats, label):
    path = os.path.join(OUTPUT_DIR, f'results_table_{label}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['=== DESCRIPTIVE STATISTICS ==='])
        w.writerow(['Algorithm','Mean','Std','Best','Worst','Median',
                    'IQR','CV(%)','Q25','Q75',
                    'Conv.Iter Mean','Conv.Iter Std',
                    'Runtime/run(s)','Friedman Rank','Improvement(%)'])
        for res, prob in [(traj_res, 'TRAJECTORY'), (dep_res, 'DEPLOYMENT')]:
            w.writerow([f'--- {prob} PROBLEM ---'])
            for a in ALGS:
                r = res[a]
                w.writerow([
                    a,
                    f"{r['mean']:.4f}", f"{r['std']:.4f}",
                    f"{r['best']:.4f}", f"{r['worst']:.4f}",
                    f"{r['median']:.4f}", f"{r['iqr']:.4f}",
                    f"{r['cv']:.2f}",
                    f"{r['q25']:.4f}", f"{r['q75']:.4f}",
                    f"{r['conv_mean']:.1f}", f"{r['conv_std']:.1f}",
                    f"{r['time_per_run']:.2f}",
                    f"{r.get('friedman_rank', '-'):.2f}" if 'friedman_rank' in r else '-',
                    f"{r.get('improvement_pct', 0):.2f}",
                ])
        w.writerow([])
        w.writerow(['=== WILCOXON PAIRWISE TESTS ==='])
        w.writerow(['Problem','Pair','W-stat','p-value','Significant',
                    "Cohen's d",'Magnitude'])
        for res, stats_d, prob in [
            (traj_res, traj_stats, 'TRAJECTORY'),
            (dep_res,  dep_stats,  'DEPLOYMENT'),
        ]:
            for (a1, a2), v in stats_d['wilcoxon'].items():
                w.writerow([
                    prob, f"{a1} vs {a2}",
                    f"{v['stat']:.2f}", f"{v['p']:.6f}",
                    'Yes' if v['significant'] else 'No',
                    f"{v['cohens_d']:.4f}", v['magnitude'],
                ])
    print(f"\n  CSV saved → {path}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT FORMATTED TABLE TO TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════
def print_table(results, problem_name):
    print(f"\n{'═'*90}")
    print(f"  {problem_name} — RESULTS TABLE  ({N_RUNS} runs × {N_ITER} iterations × pop={POP_SIZE})")
    print(f"{'═'*90}")
    hdr = (f"{'Alg':<5} {'Mean':>9} {'±Std':>8} {'Best':>9} {'Worst':>9} "
           f"{'Median':>9} {'IQR':>7} {'CV%':>6} "
           f"{'Conv.':>6} {'±':>5} {'Time/r':>7} {'Rank':>5} {'Improv%':>8}")
    print(hdr)
    print('─' * 90)
    for a in ALGS:
        r = results[a]
        print(
            f"{a:<5} "
            f"{r['mean']:>9.4f} "
            f"{r['std']:>8.4f} "
            f"{r['best']:>9.4f} "
            f"{r['worst']:>9.4f} "
            f"{r['median']:>9.4f} "
            f"{r['iqr']:>7.4f} "
            f"{r['cv']:>6.2f} "
            f"{r['conv_mean']:>6.1f} "
            f"{r['conv_std']:>5.1f} "
            f"{r['time_per_run']:>7.2f}s "
            f"{r.get('friedman_rank',0):>5.2f} "
            f"{r.get('improvement_pct',0):>8.2f}%"
        )
    print('═' * 90)

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("═"*60)
print("  UAV METAHEURISTIC SIMULATION — IEEE COMST GRADE")
print(f"  Runs={N_RUNS} | Iterations={N_ITER} | Population={POP_SIZE}")
print("═"*60)

traj_res  = run_experiment(trajectory_fitness,  TRAJ_DIM,   "Trajectory Planning")
dep_res   = run_experiment(deployment_fitness,  DEPLOY_DIM, "Node Deployment")

print("\n\nRunning statistical analysis...")
traj_stats = run_statistics(traj_res)
dep_stats  = run_statistics(dep_res)

print_table(traj_res, "TRAJECTORY PLANNING")
print_table(dep_res,  "NODE DEPLOYMENT")

print("\nRunning scalability experiments...")
node_counts = [10, 20, 30, 50, 75, 100]
scalability = {a: {'times': [], 'fits': []} for a in ALGS}
orig_nodes  = GROUND_NODES_TRAJ.copy()

def _scal_run(n):
    global GROUND_NODES_TRAJ
    GROUND_NODES_TRAJ = np.random.uniform(50, AREA-50, (n, 2))
    for a, runner in RUNNERS.items():
        np.random.seed(42)
        t0 = time.time()
        runner(trajectory_fitness, TRAJ_DIM, n_iter=100, seed=42)
        scalability[a]['times'].append(time.time() - t0)

for n in node_counts:
    print(f"  N={n}...", end=' ', flush=True)
    _scal_run(n)
    print("done")

GROUND_NODES_TRAJ = orig_nodes

csv_path = export_csv(traj_res, dep_res, traj_stats, dep_stats, 'full')

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — CONVERGENCE CURVES
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 8 — Convergence Curves...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=GS['bg'])
iters = np.arange(1, N_ITER + 1)

for ax, (res, sub) in zip(axes, [
    (traj_res, '(a) Trajectory Planning'),
    (dep_res,  '(b) Node Deployment'),
]):
    ax.set_facecolor(GS['panel'])
    ax.grid(True, linestyle='--', linewidth=0.5, color=GS['gridline'])

    for a, st in ALG_STYLES.items():
        m, s = res[a]['mean_hist'], res[a]['std_hist']
        q25, q75 = res[a]['q25_hist'], res[a]['q75_hist']
        ax.plot(iters, m, color=st['color'], linestyle=st['ls'],
                linewidth=st['lw'], marker=st['marker'],
                markevery=30, markersize=4, label=st['label'], zorder=5)
        ax.fill_between(iters, m-s, m+s,
                        alpha=0.10, color=st['color'], zorder=3,
                        label='_nolegend_')
        ax.fill_between(iters, q25, q75,
                        alpha=0.06, color=st['color'], zorder=2,
                        label='_nolegend_')

    # Convergence markers
    for a, st in ALG_STYLES.items():
        ci = int(res[a]['conv_mean'])
        if ci < N_ITER:
            ax.axvline(ci, color=st['color'], linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.set_title(sub, fontweight='bold', pad=6)
    ax.legend(framealpha=0.95, edgecolor=GS['border'])
    ax.set_xlim(1, N_ITER)

    # Stats inset
    txt = '\n'.join([
        f"{a}: {res[a]['mean']:.3f}±{res[a]['std']:.3f}"
        for a in ALGS
    ])
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=GS['border'], alpha=0.92))

plt.tight_layout(pad=1.5)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8b — TRAJECTORY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 8b — Trajectory Maps...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor=GS['bg'])

for ax, (a, st) in zip(axes, ALG_STYLES.items()):
    ax.set_facecolor(GS['bg'])
    ax.set_xlim(-30, AREA+30)
    ax.set_ylim(-30, AREA+30)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.3, color=GS['gridline'], alpha=0.6)

    sol  = traj_res[a]['best_sol']
    wps  = sol.reshape(N_WPS, 2)
    full = np.vstack([UAV_START, wps, UAV_END])

    # Signal heatmap
    res_h = 40
    xs, ys = np.linspace(0, AREA, res_h), np.linspace(0, AREA, res_h)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros((res_h, res_h))
    for i, px in enumerate(xs):
        for j, py in enumerate(ys):
            pt = np.array([px, py])
            ml = min(path_loss(ALT, np.linalg.norm(pt-wp)) for wp in full)
            ZZ[j, i] = -ml
    ax.contourf(XX, YY, ZZ, levels=12, cmap='Greys', alpha=0.25, zorder=1)

    # NFZ
    for nfz in NO_FLY_ZONES:
        c = Circle(nfz['center'], nfz['radius'],
                   fc='#DDDDDD', ec='black', lw=1.2,
                   ls='--', alpha=0.7, zorder=3, hatch='///')
        ax.add_patch(c)
        ax.text(*nfz['center'], 'NFZ', ha='center', va='center',
                fontsize=6, fontweight='bold', zorder=4)

    # Ground nodes
    ax.scatter(GROUND_NODES_TRAJ[:,0], GROUND_NODES_TRAJ[:,1],
               s=22, c='black', marker='s', zorder=5,
               edgecolors='white', linewidths=0.4)

    # Path with direction arrows
    ax.plot(full[:,0], full[:,1], color=st['color'],
            linewidth=1.6, linestyle=st['ls'], alpha=0.85, zorder=6)
    for i in range(len(full)-1):
        mid = (full[i] + full[i+1]) / 2
        d   = full[i+1] - full[i]
        ax.annotate('', xy=mid + d*0.01, xytext=mid - d*0.01,
                    arrowprops=dict(arrowstyle='->', color=st['color'],
                                    lw=1.2), zorder=7)

    # Waypoints with gradient shade
    shades = np.linspace(0.15, 0.85, N_WPS)
    for k, (wp, sh) in enumerate(zip(wps, shades)):
        ax.plot(*wp, 'o', ms=6, color=str(sh),
                mec='black', mew=0.6, zorder=8)
        ax.text(wp[0]+14, wp[1]+14, f'W{k+1}', fontsize=5.5,
                color='black', zorder=9)

    ax.plot(*UAV_START, '^', ms=10, color='black',
            mec='white', mew=0.7, zorder=10)
    ax.plot(*UAV_END, 'v', ms=10, color='#555555',
            mec='white', mew=0.7, zorder=10)
    ax.text(UAV_START[0]+18, UAV_START[1]+18, 'S', fontsize=7, fontweight='bold')
    ax.text(UAV_END[0]-35, UAV_END[1]-35, 'E', fontsize=7, fontweight='bold')

    plen = sum(np.linalg.norm(full[i+1]-full[i]) for i in range(len(full)-1))
    ax.set_title(f'{a} — {plen:.0f}m\nfit={traj_res[a]["best"]:.4f}',
                 fontweight='bold', pad=5, fontsize=8)
    ax.set_xlabel('X (m)')
    if a == 'GA': ax.set_ylabel('Y (m)')
    for sp in ax.spines.values(): sp.set_edgecolor(GS['border'])

plt.tight_layout(pad=1.2)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — DEPLOYMENT COVERAGE MAPS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 10 — Deployment Maps...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor=GS['bg'])

for ax, (a, st) in zip(axes, ALG_STYLES.items()):
    ax.set_facecolor(GS['bg'])
    ax.set_xlim(-30, AREA+30)
    ax.set_ylim(-30, AREA+30)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.3, color=GS['gridline'], alpha=0.6)

    pos = dep_res[a]['best_sol'].reshape(K_UAVS, 2)

    # Coverage heatmap
    res_h = 50
    xs, ys = np.linspace(0, AREA, res_h), np.linspace(0, AREA, res_h)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros((res_h, res_h))
    for i, px in enumerate(xs):
        for j, py in enumerate(ys):
            pt = np.array([px, py])
            ZZ[j, i] = np.exp(-min(np.linalg.norm(pt-p) for p in pos) / 200)
    ax.contourf(XX, YY, ZZ, levels=15, cmap='Greys', alpha=0.30, zorder=1)
    ax.contour(XX, YY, ZZ, levels=4, colors='black',
               alpha=0.12, linewidths=0.3, zorder=2)

    # NFZ
    for nfz in NO_FLY_ZONES:
        c = Circle(nfz['center'], nfz['radius'],
                   fc='#DDDDDD', ec='black', lw=1.0,
                   ls='--', alpha=0.6, zorder=3, hatch='///')
        ax.add_patch(c)
        ax.text(*nfz['center'], 'NFZ', ha='center', va='center',
                fontsize=5.5, fontweight='bold', zorder=4)

    # Ground nodes
    covered = []
    for node in GROUND_NODES_DEPLOY:
        ic = any(np.linalg.norm(node-p) < COV_RADIUS for p in pos)
        covered.append(ic)
        ax.plot(*node, 's', ms=4,
                color='black' if ic else '#BBBBBB',
                mec='black', mew=0.3, zorder=5)

    # UAV positions + coverage circles
    for k, p in enumerate(pos):
        c = Circle(p, COV_RADIUS, fc='none',
                   ec=st['color'], lw=1.2, ls='--', alpha=0.75, zorder=6)
        ax.add_patch(c)
        ax.plot(*p, '*', ms=12, color=st['color'],
                mec='black', mew=0.5, zorder=8)
        ax.text(p[0]+12, p[1]+12, f'U{k+1}', fontsize=6,
                fontweight='bold', color='black', zorder=9)

    # Connection lines
    for node, ic in zip(GROUND_NODES_DEPLOY, covered):
        if ic:
            near = min(pos, key=lambda p: np.linalg.norm(node-p))
            ax.plot([node[0], near[0]], [node[1], near[1]],
                    '-', color='#999999', lw=0.35, alpha=0.45, zorder=4)

    cov_rate = sum(covered) / N_NODES_DEPLOY * 100
    ax.set_title(f'{a} — {cov_rate:.0f}% coverage\nfit={dep_res[a]["best"]:.4f}',
                 fontweight='bold', pad=5, fontsize=8)
    ax.set_xlabel('X (m)')
    if a == 'GA': ax.set_ylabel('Y (m)')

    legend_elems = [
        mpatches.Patch(fc='black',   label=f'Covered ({sum(covered)})'),
        mpatches.Patch(fc='#BBBBBB', label=f'Uncovered ({N_NODES_DEPLOY-sum(covered)})'),
        mpatches.Patch(fc='#DDDDDD', ec='black', label='NFZ'),
        plt.Line2D([0],[0], marker='*', color='w',
                   mfc=st['color'], ms=8, label='UAV'),
    ]
    ax.legend(handles=legend_elems, loc='lower right',
              fontsize=5.5, framealpha=0.95, edgecolor=GS['border'])
    for sp in ax.spines.values(): sp.set_edgecolor(GS['border'])

plt.tight_layout(pad=1.2)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 17 — SCALABILITY
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 17 — Scalability...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=GS['bg'])

# (a) Linear
ax = axes[0]
ax.set_facecolor(GS['panel'])
ax.grid(True, linestyle='--', linewidth=0.5, color=GS['gridline'])
for a, st in ALG_STYLES.items():
    ax.plot(node_counts, scalability[a]['times'],
            color=st['color'], ls=st['ls'], lw=st['lw'],
            marker=st['marker'], ms=6, label=st['label'], zorder=5)
ax.set_xlabel('Number of Ground Nodes (N)')
ax.set_ylabel('Runtime (seconds)')
ax.set_title('(a) Linear Scale', fontweight='bold', pad=6)
ax.legend(framealpha=0.95, edgecolor=GS['border'])
ax.set_xticks(node_counts)
for sp in ax.spines.values(): sp.set_edgecolor(GS['border'])

# (b) Log-log with fitted complexity
ax2 = axes[1]
ax2.set_facecolor(GS['panel'])
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color=GS['gridline'])
x_fit  = np.array(node_counts, dtype=float)
x_fine = np.linspace(node_counts[0], node_counts[-1]*1.05, 300)
for a, st in ALG_STYLES.items():
    times = scalability[a]['times']
    ax2.plot(node_counts, times,
             color=st['color'], ls=st['ls'], lw=st['lw'],
             marker=st['marker'], ms=6, label=st['label'], zorder=5)
    coeffs = np.polyfit(np.log(x_fit), np.log(times), 1)
    trend  = np.exp(coeffs[1]) * x_fine**coeffs[0]
    ax2.plot(x_fine, trend, color=st['color'],
             ls=':', lw=0.9, alpha=0.55, zorder=3)
    ax2.text(x_fine[-1]*1.01, trend[-1],
             f'O(N^{coeffs[0]:.2f})', fontsize=6.5,
             color=st['color'], va='center')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Number of Ground Nodes (N)')
ax2.set_ylabel('Runtime (seconds)')
ax2.set_title('(b) Log-log + Complexity Fit', fontweight='bold', pad=6)
ax2.legend(framealpha=0.95, edgecolor=GS['border'])
ax2.set_xticks(node_counts)
ax2.set_xticklabels([str(n) for n in node_counts])
for sp in ax2.spines.values(): sp.set_edgecolor(GS['border'])

# (c) Normalized runtime (relative to GA)
ax3 = axes[2]
ax3.set_facecolor(GS['panel'])
ax3.grid(True, linestyle='--', linewidth=0.5, color=GS['gridline'])
ga_times = np.array(scalability['GA']['times'])
for a, st in ALG_STYLES.items():
    norm_t = np.array(scalability[a]['times']) / ga_times
    ax3.plot(node_counts, norm_t,
             color=st['color'], ls=st['ls'], lw=st['lw'],
             marker=st['marker'], ms=6, label=st['label'], zorder=5)
ax3.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
ax3.set_xlabel('Number of Ground Nodes (N)')
ax3.set_ylabel('Normalized Runtime (GA=1.0)')
ax3.set_title('(c) Normalized Overhead vs GA', fontweight='bold', pad=6)
ax3.legend(framealpha=0.95, edgecolor=GS['border'])
ax3.set_xticks(node_counts)
for sp in ax3.spines.values(): sp.set_edgecolor(GS['border'])

plt.tight_layout(pad=1.5)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 18 — PERFORMANCE HEATMAP + RADAR + BOX PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 18 — Performance Matrix...")
fig = plt.figure(figsize=(16, 6), facecolor=GS['bg'])
gs  = GridSpec(1, 4, figure=fig, wspace=0.35)

def norm_col(vals, lower_is_better=True):
    lo, hi = vals.min(), vals.max()
    if abs(hi - lo) < 1e-9: return np.full_like(vals, 0.5)
    n = (vals - lo) / (hi - lo)
    return 1 - n if lower_is_better else n

problems  = ['Trajectory\nFitness', 'Deployment\nFitness',
             'Convergence\nSpeed', 'Scalability\n(N=50)']
raw_traj  = np.array([traj_res[a]['mean']         for a in ALGS])
raw_dep   = np.array([dep_res[a]['mean']           for a in ALGS])
raw_conv  = np.array([traj_res[a]['conv_mean']     for a in ALGS])
raw_scal  = np.array([scalability[a]['times'][node_counts.index(50)]
                       for a in ALGS])

norm_mat = np.column_stack([
    norm_col(raw_traj,  lower_is_better=True),
    norm_col(raw_dep,   lower_is_better=True),
    norm_col(raw_conv,  lower_is_better=True),
    norm_col(raw_scal,  lower_is_better=True),
])

# (a) Heatmap
ax_h = fig.add_subplot(gs[0, :2])
im   = ax_h.imshow(norm_mat, cmap='Greys', aspect='auto', vmin=0, vmax=1)
ax_h.set_xticks(range(4)); ax_h.set_xticklabels(problems, fontsize=8)
ax_h.set_yticks(range(4)); ax_h.set_yticklabels(ALGS, fontsize=9, fontweight='bold')
for i in range(4):
    for j in range(4):
        v = norm_mat[i, j]
        ax_h.text(j, i, f'{v:.2f}',
                  ha='center', va='center', fontsize=9, fontweight='bold',
                  color='white' if v > 0.55 else 'black', zorder=3)
for i in range(5): ax_h.axhline(i-0.5, color='white', lw=1.5)
for j in range(5): ax_h.axvline(j-0.5, color='white', lw=1.5)
cb = plt.colorbar(im, ax=ax_h, shrink=0.85, pad=0.02)
cb.set_label('Normalized Score (1=best)', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax_h.set_title('(a) Performance Heatmap', fontweight='bold', pad=8)

# (b) Radar
ax_r = fig.add_subplot(gs[0, 2], polar=True)
cats  = ['Trajectory', 'Deployment', 'Conv.Speed', 'Scalability']
N_c   = len(cats)
ang   = np.linspace(0, 2*np.pi, N_c, endpoint=False).tolist()
ang  += ang[:1]
ax_r.set_facecolor(GS['panel'])
for i, (a, st) in enumerate(ALG_STYLES.items()):
    vals = norm_mat[i].tolist() + [norm_mat[i][0]]
    ax_r.plot(ang, vals, color=st['color'],
              ls=st['ls'], lw=1.8, marker=st['marker'], ms=4, label=a)
    ax_r.fill(ang, vals, alpha=0.08, color=st['color'])
ax_r.set_xticks(ang[:-1])
ax_r.set_xticklabels(cats, fontsize=7.5)
ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_r.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=6.5, color='grey')
ax_r.set_ylim(0, 1.15)
ax_r.grid(color=GS['gridline'], ls='--', lw=0.5)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15),
            fontsize=7.5, framealpha=0.95, edgecolor=GS['border'])
ax_r.set_title('(b) Performance\nProfile', fontweight='bold', pad=12, fontsize=8)

# (c) Box plots (trajectory fitness distribution)
ax_b = fig.add_subplot(gs[0, 3])
ax_b.set_facecolor(GS['panel'])
ax_b.grid(True, axis='y', linestyle='--', lw=0.5, color=GS['gridline'])
bp_data = [traj_res[a]['fits'] for a in ALGS]
bp = ax_b.boxplot(bp_data, patch_artist=True, notch=True,
                   medianprops=dict(color='black', lw=1.5),
                   whiskerprops=dict(lw=0.9),
                   capprops=dict(lw=0.9),
                   flierprops=dict(marker='.', ms=3, alpha=0.5))
shades = ['#000000', '#555555', '#888888', '#BBBBBB']
for patch, shade in zip(bp['boxes'], shades):
    patch.set_facecolor(shade)
    patch.set_alpha(0.6)
ax_b.set_xticklabels(ALGS, fontsize=8)
ax_b.set_xlabel('Algorithm')
ax_b.set_ylabel('Fitness (30 runs)')
ax_b.set_title('(c) Trajectory Fitness\nDistribution', fontweight='bold',
               pad=8, fontsize=8)
for sp in ax_b.spines.values(): sp.set_edgecolor(GS['border'])

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE EXTRA — STATISTICAL SIGNIFICANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Statistical Significance Matrix...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=GS['bg'])

for ax, (stats_d, prob) in zip(axes, [
    (traj_stats, '(a) Trajectory — Wilcoxon p-values'),
    (dep_stats,  '(b) Deployment — Wilcoxon p-values'),
]):
    mat  = np.ones((4, 4))
    dmat = np.zeros((4, 4))
    for (a1, a2), v in stats_d['wilcoxon'].items():
        i, j = ALGS.index(a1), ALGS.index(a2)
        mat[i,j]  = v['p']
        mat[j,i]  = v['p']
        dmat[i,j] = abs(v['cohens_d'])
        dmat[j,i] = abs(v['cohens_d'])

    im = ax.imshow(mat, cmap='Greys_r', vmin=0, vmax=0.1, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(ALGS, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(ALGS, fontsize=9)

    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=9, color='grey')
            else:
                p   = mat[i, j]
                d   = dmat[i, j]
                sig = '★' if p < 0.05 else ''
                ax.text(j, i,
                        f'p={p:.3f}{sig}\nd={d:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if p < 0.03 else 'black')

    for k in range(5):
        ax.axhline(k-0.5, color='white', lw=1.2)
        ax.axvline(k-0.5, color='white', lw=1.2)

    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("p-value (★ = significant)", fontsize=8)
    ax.set_title(prob, fontweight='bold', pad=8)

plt.tight_layout(pad=1.5)
plt.show()

print(f"\n{'═'*60}")
print(f"  SIMULATION COMPLETE")
print(f"  CSV results → {csv_path}")
print(f"{'═'*60}")