#!/usr/bin/env python3
"""
N-Body Barnes-Hut — Performance Benchmarking Script (v2 — Scaling Edition)
===========================================================================

NEW in v2
---------
  • Strong-scaling sweep over MPI processes  (1 → 8)
  • Strong-scaling sweep over OMP threads    (1 → 16)
  • Particle-count sweep                     (N = 1000 → 50000)
  • All three sweeps produce line plots comparing every binary
  • All raw sweep data saved to CSV alongside per-run CSVs
  • Original single-P bar charts are preserved (run with --no-sweep)

Binaries targeted
-----------------
  pnbody_baseline    — pure MPI, original block partition
  pnbody_orb         — pure MPI, ORB load balancing
  pnbody_omp         — MPI+OMP hybrid (mpirun -np P_mpi, OMP_NUM_THREADS=T)
  pnbody_theta       — pure MPI, adaptive theta
  pnbody_combined    — MPI+OMP hybrid
  pnbody_parallel_tree — pure MPI

Usage
-----
  python3 benchmark.py [options]

  Core options (single-point run):
    --n       Particles            (default 10000)
    --t       Timesteps            (default 500)
    --p       MPI ranks / OMP thr  (default 4)
    --runs    Repeats per config   (default 3, median taken)
    --quick   N=1000 T=50 fast test

  Scaling sweep options (override single-point):
    --sweep-mpi          Run MPI process sweep  1,2,4,8,8
    --sweep-omp          Run OMP thread sweep   1,2,4,8,16
    --sweep-n            Run particle-count sweep 1k,2k,5k,10k,20k,50k
    --sweep-all          Enable all three sweeps
    --no-sweep           Skip sweeps, single-point run only (default behaviour)

    --mpi-max    INT     Upper bound for MPI sweep   (default 8)
    --omp-max    INT     Upper bound for OMP sweep   (default 16)
    --t-sweep    INT     Timesteps used during sweeps (default 200)

Outputs (results/)
------------------
  summary.csv                  — single-point bar chart data
  scaling_mpi.csv              — MPI sweep raw data
  scaling_omp.csv              — OMP sweep raw data
  scaling_n.csv                — particle-count sweep raw data

  wall_time.png                — bar: wall time per binary (single point)
  speedup.png                  — bar: speedup over baseline
  efficiency.png               — bar: parallel efficiency
  comm_overhead.png            — bar: MPI% (mpiP)
  mpi_breakdown.png            — stacked bar: Allgather/Bcast/Scatter/Allreduce
  load_balance.png             — per-rank force times
  theta_evolution.png          — adaptive theta per rank
  variance.png                 — box plot timing variance

  scaling_mpi_time.png         — line: wall time vs #processes (strong scaling)
  scaling_mpi_speedup.png      — line: speedup vs #processes
  scaling_mpi_efficiency.png   — line: efficiency vs #processes
  scaling_omp_time.png         — line: wall time vs #OMP threads
  scaling_omp_speedup.png      — line: speedup vs #OMP threads
  scaling_omp_efficiency.png   — line: efficiency vs #OMP threads
  scaling_n_time.png           — line: wall time vs N (particles)
  scaling_n_throughput.png     — line: particles·steps / second vs N
"""

import subprocess, os, re, sys, glob, time, argparse, csv, statistics
from pathlib import Path

# ── matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — CSV results saved but no plots. "
          "Install: pip install matplotlib")

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "results"

# (label, executable, mode)
#   mode "mpi"     → mpirun -np P  ./<exe>  (OMP_NUM_THREADS=1)
#   mode "omp"     → mpirun -np 1  ./<exe>  (OMP_NUM_THREADS=P)
#   mode "hybrid"  → mpirun -np P_mpi ./<exe>  (OMP_NUM_THREADS=P_omp)
#                    P_mpi and P_omp supplied per-call
BINARIES = [
    ("Baseline",      "pnbody_baseline",      "mpi"),
    ("ORB",           "pnbody_orb",           "mpi"),
    ("OpenMP",        "pnbody_omp",           "omp"),
    ("Adaptive-θ",    "pnbody_theta",         "mpi"),
    ("Combined",      "pnbody_combined",      "omp"),
    ("Parallel-Tree", "pnbody_parallel_tree", "mpi"),
]

COLOURS = {
    "Baseline":      "#555555",
    "ORB":           "#2196F3",
    "OpenMP":        "#4CAF50",
    "Adaptive-θ":    "#FF9800",
    "Combined":      "#9C27B0",
    "Parallel-Tree": "#F44336",
}
MARKERS = {
    "Baseline":      "o",
    "ORB":           "s",
    "OpenMP":        "^",
    "Adaptive-θ":    "D",
    "Combined":      "P",
    "Parallel-Tree": "X",
}

# Default sweep points
MPI_SWEEP_POINTS = [1, 2, 4, 8]
OMP_SWEEP_POINTS = [1, 2, 4, 8, 16]
N_SWEEP_POINTS   = [5000, 10000, 20000, 50000, 100000]

# ═════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="N-Body scaling benchmark runner")
    # Single-point
    p.add_argument("--n",       type=int, default=10000, help="Particle count (single-point)")
    p.add_argument("--t",       type=int, default=500,   help="Timesteps (single-point)")
    p.add_argument("--p",       type=int, default=4,     help="MPI ranks / OMP threads (single-point)")
    p.add_argument("--runs",    type=int, default=3,     help="Repeats per config (median taken)")
    p.add_argument("--quick",   action="store_true",     help="Fast test: N=1000 T=50 P=2")
    # Sweep on/off
    p.add_argument("--sweep-mpi",  action="store_true",  help="MPI process scaling sweep")
    p.add_argument("--sweep-omp",  action="store_true",  help="OMP thread scaling sweep")
    p.add_argument("--sweep-n",    action="store_true",  help="Particle-count scaling sweep")
    p.add_argument("--sweep-all",  action="store_true",  help="Enable all three sweeps")
    p.add_argument("--no-sweep",   action="store_true",  help="Disable all sweeps (default)")
    # Sweep parameters
    p.add_argument("--mpi-max",  type=int, default=8,   help="Max MPI ranks for sweep")
    p.add_argument("--omp-max",  type=int, default=16,   help="Max OMP threads for sweep")
    p.add_argument("--t-sweep",  type=int, default=200,  help="Timesteps used in scaling sweeps")
    return p.parse_args()

# ═════════════════════════════════════════════════════════════════════════════
# mpiP parsing
# ═════════════════════════════════════════════════════════════════════════════

def find_latest_mpip_file(exe_name):
    candidates = sorted(glob.glob("*.mpiP"), key=os.path.getmtime, reverse=True)
    for c in candidates:
        if exe_name in c:
            return c
    return candidates[0] if candidates else None


def parse_mpip(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    result = {
        "mpi_pct": 0.0, "app_time": 0.0, "mpi_time": 0.0,
        "allgather_pct": 0.0, "bcast_pct": 0.0,
        "scatter_pct": 0.0,   "allreduce_pct": 0.0,
        "allgather_s": 0.0,   "bcast_s": 0.0,
        "scatter_s": 0.0,     "allreduce_s": 0.0,
    }
    try:
        text = open(filepath).read()
    except Exception:
        return None

    mpi_time_block = re.search(
        r"@---\s*MPI Time.*?---+\s*\n(.*?)\n\n", text, re.DOTALL)
    if mpi_time_block:
        lines = mpi_time_block.group(1).strip().split("\n")
        app_times, mpi_times, mpi_pcts = [], [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    int(parts[0])
                    app_times.append(float(parts[1]))
                    mpi_times.append(float(parts[2]))
                    mpi_pcts.append(float(parts[3]))
                except ValueError:
                    continue
        if app_times:
            result["app_time"] = statistics.mean(app_times)
            result["mpi_time"] = statistics.mean(mpi_times)
            result["mpi_pct"]  = statistics.mean(mpi_pcts)

    call_block = re.search(
        r"@---\s*Callsite Time statistics.*?---+\s*\n(.*?)(?:\n@---|\Z)",
        text, re.DOTALL)
    func_map = {
        "MPI_Allgather": ("allgather_pct", "allgather_s"),
        "MPI_Bcast":     ("bcast_pct",     "bcast_s"),
        "MPI_Scatter":   ("scatter_pct",   "scatter_s"),
        "MPI_Allreduce": ("allreduce_pct", "allreduce_s"),
    }
    if call_block:
        for line in call_block.group(1).split("\n"):
            parts = line.split()
            if len(parts) < 8:
                continue
            fname = parts[0]
            if fname in func_map:
                try:
                    mean_time    = float(parts[5])
                    mpi_pct_call = float(parts[-1])
                    pct_key, s_key = func_map[fname]
                    result[s_key]   = mean_time
                    result[pct_key] = mpi_pct_call
                except (ValueError, IndexError):
                    continue
    return result

# ═════════════════════════════════════════════════════════════════════════════
# Stdout parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_stdout_force_times(stdout):
    times = []
    for m in re.finditer(r"\[ForceTime\]\s+Rank\s+(\d+):\s+([\d.e+\-]+)\s*s", stdout):
        times.append((int(m.group(1)), float(m.group(2))))
    return times

def parse_stdout_theta(stdout):
    records = []
    for m in re.finditer(r"step\s+(\d+)\s*\|\s*thetas:(.*)", stdout):
        step = int(m.group(1))
        thetas = {}
        for tm in re.finditer(r"r(\d+)=([\d.]+)", m.group(2)):
            thetas[int(tm.group(1))] = float(tm.group(2))
        records.append((step, thetas))
    return records

def parse_wall_time(stdout, stderr):
    m = re.search(r"\[Time\].*?wall.*?:\s*([\d.]+)\s*s", stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"real\s+(\d+)m([\d.]+)s", stderr)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"elapsed.*?([\d.]+)s", stdout)
    if m:
        return float(m.group(1))
    return None

# ═════════════════════════════════════════════════════════════════════════════
# Core runner — single binary, configurable P_mpi and P_omp
# ═════════════════════════════════════════════════════════════════════════════

def run_binary(label, exe, mode, N, T, P_mpi, P_omp):
    """
    Run one binary with the given parallelism configuration.

    mode "mpi"  → mpirun -np P_mpi ./<exe>  with OMP_NUM_THREADS=1
    mode "omp"  → mpirun -np 4     ./<exe>  with OMP_NUM_THREADS=P_omp
    mode "hybrid" → mpirun -np P_mpi ./<exe> with OMP_NUM_THREADS=P_omp

    Returns (wall_time_s, stdout, stderr, mpip_file) or (None,...) on failure.
    """
    if not os.path.exists(f"./{exe}"):
        print(f"  [SKIP] ./{exe} not found")
        return None, "", "", None

    env = os.environ.copy()

    if mode == "omp":
        env["OMP_NUM_THREADS"] = str(P_omp)
        cmd = ["mpirun", "-np", "4", f"./{exe}", str(N), str(T)]
    elif mode == "hybrid":
        env["OMP_NUM_THREADS"] = str(P_omp)
        cmd = ["mpirun", "-np", str(P_mpi), f"./{exe}", str(N), str(T)]
    else:  # "mpi"
        env["OMP_NUM_THREADS"] = "1"
        cmd = ["mpirun", "-np", str(P_mpi), f"./{exe}", str(N), str(T)]

    print(f"  CMD: {' '.join(cmd)}  OMP={env['OMP_NUM_THREADS']}", flush=True)

    for old in glob.glob("*.mpiP"):
        try:
            os.remove(old)
        except Exception:
            pass

    t_start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {exe}")
        return None, "", "", None
    except FileNotFoundError:
        print("  [ERROR] mpirun not found in PATH")
        return None, "", "", None

    wall = time.perf_counter() - t_start
    stdout, stderr = proc.stdout, proc.stderr

    prog_wall = parse_wall_time(stdout, stderr)
    if prog_wall is not None:
        wall = prog_wall

    mpip_file = find_latest_mpip_file(exe)

    if proc.returncode != 0:
        print(f"  [WARN] exit code {proc.returncode}: {stderr[:200]}")

    return wall, stdout, stderr, mpip_file


def run_median(label, exe, mode, N, T, P_mpi, P_omp, runs):
    """Run `runs` times and return (median_wall, last_stdout, last_mpip)."""
    walls = []
    last_stdout = ""
    last_mpip   = None
    for r in range(runs):
        print(f"  Run {r+1}/{runs}:", end=" ", flush=True)
        wall, stdout, stderr, mpip = run_binary(label, exe, mode, N, T, P_mpi, P_omp)
        if wall is None:
            break
        print(f"{wall:.2f}s")
        walls.append(wall)
        last_stdout = stdout
        last_mpip   = mpip
    if not walls:
        return None, "", None
    med = statistics.median(walls)
    print(f"  → median: {med:.3f}s")
    return med, last_stdout, last_mpip

# ═════════════════════════════════════════════════════════════════════════════
# Single-point benchmark (original behaviour)
# ═════════════════════════════════════════════════════════════════════════════

def run_all_single(args):
    N, T, P, RUNS = args.n, args.t, args.p, args.runs
    if args.quick:
        N, T, P = 1000, 50, 2
        print(f"[Quick] N={N} T={T} P={P}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    wall_times = {}
    mpip_data  = {}
    force_data = {}
    theta_data = {}
    all_walls  = {}

    print(f"\n{'='*60}")
    print(f"  Single-point run  |  N={N}  T={T}  P={P}  runs={RUNS}")
    print(f"{'='*60}\n")

    for (label, exe, mode) in BINARIES:
        print(f"\n── {label} ──")
        # For single-point: MPI binaries use P ranks, OMP binaries use P threads
        P_mpi = P if mode == "mpi" else 1
        P_omp = P if mode == "omp" else 1

        med, last_stdout, last_mpip = run_median(
            label, exe, mode, N, T, P_mpi, P_omp, RUNS)
        if med is None:
            continue

        wall_times[label] = med
        all_walls[label]  = [med]  # simplified; full list not needed here

        if mode == "mpi" and last_mpip:
            mp = parse_mpip(last_mpip)
            if mp:
                mpip_data[label] = mp
                print(f"  → MPI overhead: {mp['mpi_pct']:.1f}%  "
                      f"Allgather: {mp['allgather_s']:.3f}s")

        ft = parse_stdout_force_times(last_stdout)
        if ft:
            force_data[label] = ft

        th = parse_stdout_theta(last_stdout)
        if th:
            theta_data[label] = th

    return wall_times, mpip_data, force_data, theta_data, all_walls, N, T, P

# ═════════════════════════════════════════════════════════════════════════════
# Scaling sweeps
# ═════════════════════════════════════════════════════════════════════════════

def sweep_mpi(args):
    """
    Strong scaling sweep: fix N and T, vary number of MPI ranks.
    Applies to "mpi" binaries only (OMP binaries are swept separately).
    Returns dict: label → {p: wall_time}
    """
    N, T  = args.n, args.t_sweep
    RUNS  = max(1, args.runs - 1)  # fewer repeats in sweep to save time
    max_p = args.mpi_max

    sweep_points = [p for p in MPI_SWEEP_POINTS if p <= max_p]
    if not sweep_points:
        sweep_points = [1, 2, 4]

    print(f"\n{'='*60}")
    print(f"  MPI PROCESS SWEEP  |  N={N}  T={T}  ranks={sweep_points}")
    print(f"{'='*60}")

    results = {}   # label → {p: time}
    mpi_bins = [(l, e, m) for l, e, m in BINARIES if m == "mpi"]

    for (label, exe, mode) in mpi_bins:
        results[label] = {}
        print(f"\n── {label} (MPI sweep) ──")
        for p in sweep_points:
            print(f"  P={p}:", end=" ", flush=True)
            med, _, _ = run_median(label, exe, mode, N, T, p, 1, RUNS)
            results[label][p] = med  # may be None if binary missing

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "scaling_mpi.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Binary", "N", "T", "P_mpi", "Wall_s",
                    "Speedup_vs_P1", "Efficiency_pct"])
        for label, pd in results.items():
            t1 = pd.get(1) or pd.get(min(pd.keys()))  # serial baseline
            for p, t in sorted(pd.items()):
                if t is None:
                    continue
                spd = (t1 / t) if t1 else 1.0
                eff = (spd / p) * 100
                w.writerow([label, N, T, p, f"{t:.4f}", f"{spd:.3f}", f"{eff:.1f}"])
    print(f"\n[CSV] {csv_path}")
    return results, sweep_points, N, T


def sweep_omp(args):
    """
    OMP thread scaling sweep: fix N and T, vary OMP_NUM_THREADS.
    Applies to "omp" binaries only.
    """
    N, T  = args.n, args.t_sweep
    RUNS  = max(1, args.runs - 1)
    max_t = args.omp_max

    sweep_points = [t for t in OMP_SWEEP_POINTS if t <= max_t]
    if not sweep_points:
        sweep_points = [1, 2, 4]

    print(f"\n{'='*60}")
    print(f"  OMP THREAD SWEEP  |  N={N}  T={T}  threads={sweep_points}")
    print(f"{'='*60}")

    results = {}
    omp_bins = [(l, e, m) for l, e, m in BINARIES if m == "omp"]

    for (label, exe, mode) in omp_bins:
        results[label] = {}
        print(f"\n── {label} (OMP sweep) ──")
        for t in sweep_points:
            print(f"  T={t} threads:", end=" ", flush=True)
            med, _, _ = run_median(label, exe, mode, N, T, 1, t, RUNS)
            results[label][t] = med

    csv_path = os.path.join(RESULTS_DIR, "scaling_omp.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Binary", "N", "T_steps", "OMP_threads", "Wall_s",
                    "Speedup_vs_T1", "Efficiency_pct"])
        for label, td in results.items():
            t1 = td.get(1) or td.get(min(td.keys()))
            for t, wall in sorted(td.items()):
                if wall is None:
                    continue
                spd = (t1 / wall) if t1 else 1.0
                eff = (spd / t) * 100
                w.writerow([label, N, T, t, f"{wall:.4f}", f"{spd:.3f}", f"{eff:.1f}"])
    print(f"\n[CSV] {csv_path}")
    return results, sweep_points, N, T


def sweep_n(args):
    """
    Particle-count sweep: fix P/threads, vary N.
    Runs all binaries at their natural parallelism (--p setting).
    """
    P, T  = args.p, args.t_sweep
    RUNS  = max(1, args.runs - 1)

    if args.quick:
        n_points = [500, 1000, 2000, 5000]
    else:
        n_points = N_SWEEP_POINTS

    print(f"\n{'='*60}")
    print(f"  PARTICLE-COUNT SWEEP  |  P={P}  T={T}  N={n_points}")
    print(f"{'='*60}")

    results = {}  # label → {n: time}

    for (label, exe, mode) in BINARIES:
        results[label] = {}
        print(f"\n── {label} (N sweep) ──")
        P_mpi = P if mode == "mpi" else 1
        P_omp = P if mode == "omp" else 1
        for n in n_points:
            print(f"  N={n}:", end=" ", flush=True)
            med, _, _ = run_median(label, exe, mode, n, T, P_mpi, P_omp, RUNS)
            results[label][n] = med

    csv_path = os.path.join(RESULTS_DIR, "scaling_n.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Binary", "P", "T_steps", "N", "Wall_s", "Throughput_Nsteps_per_s"])
        for label, nd in results.items():
            for n, wall in sorted(nd.items()):
                if wall is None:
                    continue
                tput = (n * T) / wall if wall > 0 else 0.0
                w.writerow([label, P, T, n, f"{wall:.4f}", f"{tput:.1f}"])
    print(f"\n[CSV] {csv_path}")
    return results, n_points, P, T

# ═════════════════════════════════════════════════════════════════════════════
# Single-point plots  (unchanged from v1)
# ═════════════════════════════════════════════════════════════════════════════

def _bar_plot(ax, labels, values, title, ylabel, colour_map):
    bars = ax.bar(labels, values,
                  color=[colour_map.get(l, "#888888") for l in labels],
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if values:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    return bars


def plot_wall_time(wall_times):
    if not HAS_MPL or not wall_times:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(wall_times.keys())
    vals   = [wall_times[l] for l in labels]
    _bar_plot(ax, labels, vals, "Wall-Clock Time per Binary", "Time (s)", COLOURS)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "wall_time.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_speedup(wall_times, P):
    if not HAS_MPL or "Baseline" not in wall_times:
        return
    baseline = wall_times["Baseline"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels   = [l for l in wall_times if l != "Baseline"]
    speedups = [baseline / wall_times[l] for l in labels]
    _bar_plot(ax, labels, speedups,
              f"Speedup over Baseline (P={P})", "Speedup (×)", COLOURS)
    ax.axhline(y=P,  color="red",  linestyle="--", linewidth=1, label=f"Ideal ({P}×)")
    ax.axhline(y=1,  color="grey", linestyle=":",  linewidth=0.8, label="Baseline (1×)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "speedup.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_efficiency(wall_times, P):
    if not HAS_MPL or "Baseline" not in wall_times:
        return
    baseline = wall_times["Baseline"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [l for l in wall_times if l != "Baseline"]
    effs   = [(baseline / wall_times[l]) / P * 100 for l in labels]
    _bar_plot(ax, labels, effs,
              f"Parallel Efficiency = Speedup / P  (P={P})", "Efficiency (%)", COLOURS)
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1, label="Ideal 100%")
    if effs:
        ax.set_ylim(0, max(effs + [110]))
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "efficiency.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_comm_overhead(mpip_data):
    if not HAS_MPL or not mpip_data:
        return
    labels = list(mpip_data.keys())
    vals   = [mpip_data[l]["mpi_pct"] for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_plot(ax, labels, vals,
              "MPI Communication Overhead (mpiP)", "MPI% of total runtime", COLOURS)
    if vals:
        ax.set_ylim(0, max(vals) * 1.2 + 1)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "comm_overhead.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_mpi_breakdown(mpip_data):
    if not HAS_MPL or not mpip_data:
        return
    labels  = list(mpip_data.keys())
    ag_vals = [mpip_data[l]["allgather_s"]  for l in labels]
    bc_vals = [mpip_data[l]["bcast_s"]      for l in labels]
    sc_vals = [mpip_data[l]["scatter_s"]    for l in labels]
    ar_vals = [mpip_data[l]["allreduce_s"]  for l in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, ag_vals, label="Allgather", color="#2196F3")
    ax.bar(x, bc_vals, bottom=ag_vals,    label="Bcast",    color="#4CAF50")
    bot2 = [a+b for a,b in zip(ag_vals, bc_vals)]
    ax.bar(x, sc_vals, bottom=bot2,       label="Scatter",  color="#FF9800")
    bot3 = [a+b for a,b in zip(bot2, sc_vals)]
    ax.bar(x, ar_vals, bottom=bot3,       label="Allreduce",color="#9C27B0")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_title("MPI Call Time Breakdown (mpiP)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "mpi_breakdown.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_load_balance(force_data):
    if not HAS_MPL or not force_data:
        return
    relevant = {k: v for k, v in force_data.items()
                if k in ("Baseline", "ORB", "Adaptive-θ")}
    if not relevant:
        return
    fig, axes = plt.subplots(1, len(relevant),
                             figsize=(4 * len(relevant), 4), sharey=True)
    if len(relevant) == 1:
        axes = [axes]
    for ax, (label, ft) in zip(axes, relevant.items()):
        ranks = [x[0] for x in ft]
        times = [x[1] for x in ft]
        mean  = sum(times) / len(times)
        ax.bar([f"R{r}" for r in ranks], times,
               color=COLOURS.get(label, "#888888"), edgecolor="white")
        ax.axhline(mean, color="red", linestyle="--", linewidth=1,
                   label=f"mean={mean:.4f}s")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Rank")
        if ax is axes[0]:
            ax.set_ylabel("Force Compute Time (s)")
        ax.legend(fontsize=7)
    fig.suptitle("Per-Rank Force Computation Time — Load Balance", fontsize=11)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "load_balance.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_theta_evolution(theta_data):
    if not HAS_MPL:
        return
    for label, records in theta_data.items():
        if not records:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = [r[0] for r in records]
        ranks = sorted(records[0][1].keys()) if records else []
        for rank in ranks:
            vals = [r[1].get(rank, None) for r in records]
            ax.plot(steps, vals, marker="o", markersize=3, label=f"Rank {rank}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8,
                   label="θ=1.0 (baseline)")
        ax.set_title(f"Adaptive Theta Evolution — {label}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Timestep"); ax.set_ylabel("θ (opening angle)")
        ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
        ax.set_ylim(0.2, 1.6)
        plt.tight_layout()
        p = os.path.join(RESULTS_DIR, "theta_evolution.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[Plot] {p}")
        break


def plot_run_variance(all_walls):
    if not HAS_MPL or not all_walls:
        return
    labels = [l for l in all_walls if len(all_walls[l]) > 1]
    if not labels:
        return
    data = [all_walls[l] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLOURS.get(label, "#888888"))
        patch.set_alpha(0.7)
    ax.set_title("Run-to-Run Timing Variance", fontsize=11, fontweight="bold")
    ax.set_ylabel("Wall Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "variance.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")

# ═════════════════════════════════════════════════════════════════════════════
# Scaling-sweep plots
# ═════════════════════════════════════════════════════════════════════════════

def _scaling_line_plot(ax, sweep_results, x_points, title, xlabel, ylabel,
                       ideal_fn=None, ideal_label="Ideal"):
    """
    Generic helper: draw one line per binary.
    sweep_results: dict  label → {x: y}   (None y values are skipped)
    """
    for label, data in sweep_results.items():
        xs = [x for x in x_points if data.get(x) is not None]
        ys = [data[x] for x in xs]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=COLOURS.get(label, "#888888"),
                marker=MARKERS.get(label, "o"),
                linewidth=1.8, markersize=6,
                label=label)

    if ideal_fn is not None and x_points:
        ix = sorted(x_points)
        iy = [ideal_fn(x) for x in ix]
        ax.plot(ix, iy, "k--", linewidth=1, alpha=0.5, label=ideal_label)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x_points)
    ax.legend(fontsize=8, loc="best")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ── MPI scaling plots ─────────────────────────────────────────────────────────

def plot_mpi_scaling_time(mpi_results, sweep_points, N, T):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _scaling_line_plot(ax, mpi_results, sweep_points,
                       f"Strong Scaling — Wall Time vs MPI Processes\n"
                       f"N={N:,}  T={T}",
                       "Number of MPI Processes", "Wall Time (s)")
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_mpi_time.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_mpi_scaling_speedup(mpi_results, sweep_points, N, T):
    if not HAS_MPL:
        return

    # Build speedup relative to each binary's own P=1 time
    speedup_data = {}
    for label, pd in mpi_results.items():
        t1 = pd.get(1)
        if t1 is None:
            # Use smallest available P as baseline
            available = {k: v for k, v in pd.items() if v is not None}
            if not available:
                continue
            t1 = available[min(available.keys())]
        speedup_data[label] = {p: (t1/t if t else None) for p, t in pd.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: speedup curves
    _scaling_line_plot(axes[0], speedup_data, sweep_points,
                       f"Strong Scaling — Speedup vs MPI Processes\nN={N:,}  T={T}",
                       "Number of MPI Processes", "Speedup (×)",
                       ideal_fn=lambda x: x,
                       ideal_label="Ideal (linear)")
    axes[0].set_ylim(bottom=0)

    # Right: parallel efficiency
    eff_data = {}
    for label, sd in speedup_data.items():
        eff_data[label] = {p: (s/p*100 if s else None) for p, s in sd.items()}

    _scaling_line_plot(axes[1], eff_data, sweep_points,
                       f"Strong Scaling — Parallel Efficiency vs MPI Processes\nN={N:,}  T={T}",
                       "Number of MPI Processes", "Efficiency (%)",
                       ideal_fn=lambda x: 100,
                       ideal_label="Ideal 100%")
    axes[1].set_ylim(0, 120)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_mpi_speedup.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


# ── OMP scaling plots ─────────────────────────────────────────────────────────

def plot_omp_scaling_time(omp_results, sweep_points, N, T):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _scaling_line_plot(ax, omp_results, sweep_points,
                       f"Strong Scaling — Wall Time vs OMP Threads\n"
                       f"N={N:,}  T={T}",
                       "Number of OMP Threads", "Wall Time (s)")
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_omp_time.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_omp_scaling_speedup(omp_results, sweep_points, N, T):
    if not HAS_MPL:
        return

    speedup_data = {}
    for label, td in omp_results.items():
        t1 = td.get(1)
        if t1 is None:
            available = {k: v for k, v in td.items() if v is not None}
            if not available:
                continue
            t1 = available[min(available.keys())]
        speedup_data[label] = {t: (t1/w if w else None) for t, w in td.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _scaling_line_plot(axes[0], speedup_data, sweep_points,
                       f"Strong Scaling — Speedup vs OMP Threads\nN={N:,}  T={T}",
                       "Number of OMP Threads", "Speedup (×)",
                       ideal_fn=lambda x: x,
                       ideal_label="Ideal (linear)")
    axes[0].set_ylim(bottom=0)

    eff_data = {}
    for label, sd in speedup_data.items():
        eff_data[label] = {t: (s/t*100 if s else None) for t, s in sd.items()}

    _scaling_line_plot(axes[1], eff_data, sweep_points,
                       f"Strong Scaling — Parallel Efficiency vs OMP Threads\nN={N:,}  T={T}",
                       "Number of OMP Threads", "Efficiency (%)",
                       ideal_fn=lambda x: 100,
                       ideal_label="Ideal 100%")
    axes[1].set_ylim(0, 120)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_omp_speedup.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


# ── N-scaling plots ───────────────────────────────────────────────────────────

def plot_n_scaling(n_results, n_points, P, T):
    if not HAS_MPL:
        return

    # ── Wall time vs N ──
    fig, ax = plt.subplots(figsize=(8, 5))
    _scaling_line_plot(ax, n_results, n_points,
                       f"Wall Time vs Particle Count  (P={P}  T={T})",
                       "Number of Particles (N)", "Wall Time (s)")
    # Annotate with O(N log N) reference line using first binary as anchor
    ref_label = next(iter(n_results), None)
    if ref_label:
        ref_data = {k: v for k, v in n_results[ref_label].items() if v is not None}
        if ref_data:
            n0 = min(ref_data)
            t0 = ref_data[n0]
            ref_y = [t0 * (n * (n / n0) * (
                        max(1, n.bit_length()) / max(1, n0.bit_length())))
                     / n0 for n in n_points]
            # Simpler: just scale proportional to N*logN
            import math
            ref_y = [t0 * (n * math.log2(max(n,2))) /
                     (n0 * math.log2(max(n0,2))) for n in n_points]
            ax.plot(n_points, ref_y, "k:", linewidth=1,
                    alpha=0.4, label="O(N log N) reference")
            ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_n_time.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")

    # ── Throughput vs N  (particle-steps / second) ──
    import math
    tput_data = {}
    for label, nd in n_results.items():
        tput_data[label] = {n: (n * T / w if w else None)
                            for n, w in nd.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    _scaling_line_plot(ax, tput_data, n_points,
                       f"Throughput vs Particle Count  (P={P}  T={T})",
                       "Number of Particles (N)", "Particle-steps / second")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "scaling_n_throughput.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")

# ═════════════════════════════════════════════════════════════════════════════
# CSV — single-point
# ═════════════════════════════════════════════════════════════════════════════

def save_csv(wall_times, mpip_data, force_data, N, T, P):
    path = os.path.join(RESULTS_DIR, "summary.csv")
    baseline_t = wall_times.get("Baseline")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Binary","N","T","P",
                    "Wall_Time_s","Speedup","Efficiency_pct",
                    "MPI_pct","MPI_time_s",
                    "Allgather_s","Bcast_s","Scatter_s","Allreduce_s",
                    "ForceTime_max_s","ForceTime_min_s","LoadImbalance_ratio"])
        for label in wall_times:
            wt  = wall_times[label]
            spd = (baseline_t / wt) if baseline_t else 1.0
            eff = (spd / P) * 100
            mp  = mpip_data.get(label, {})
            ft  = force_data.get(label, [])
            ftv = [x[1] for x in ft]
            ft_max = max(ftv) if ftv else 0.0
            ft_min = min(ftv) if ftv else 0.0
            imb    = (ft_max / (sum(ftv)/len(ftv))) if ftv else 1.0
            w.writerow([label, N, T, P,
                        f"{wt:.4f}", f"{spd:.3f}", f"{eff:.1f}",
                        f"{mp.get('mpi_pct',0):.2f}",
                        f"{mp.get('mpi_time',0):.4f}",
                        f"{mp.get('allgather_s',0):.4f}",
                        f"{mp.get('bcast_s',0):.4f}",
                        f"{mp.get('scatter_s',0):.4f}",
                        f"{mp.get('allreduce_s',0):.4f}",
                        f"{ft_max:.4f}", f"{ft_min:.4f}", f"{imb:.3f}"])
    print(f"\n[CSV] {path}")

# ═════════════════════════════════════════════════════════════════════════════
# Human-readable report
# ═════════════════════════════════════════════════════════════════════════════

def write_report(wall_times, mpip_data, force_data, N, T, P):
    baseline = wall_times.get("Baseline")
    lines = []
    lines.append("=" * 62)
    lines.append(f"  N-BODY BENCHMARK REPORT  |  N={N}  T={T}  P={P}")
    lines.append("=" * 62)

    lines.append("\nWALL-CLOCK TIME & SPEEDUP")
    lines.append("-" * 62)
    lines.append(f"  {'Binary':<20} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
    for label in wall_times:
        wt  = wall_times[label]
        spd = (baseline / wt) if baseline else 1.0
        eff = (spd / P) * 100
        flag = ""
        if spd > 1.2: flag = "  ← FASTER"
        if spd < 0.9: flag = "  ← slower"
        lines.append(f"  {label:<20} {wt:>10.3f} {spd:>10.2f}× {eff:>10.1f}%{flag}")

    if mpip_data:
        lines.append("\nMPI COMMUNICATION OVERHEAD")
        lines.append("-" * 62)
        lines.append(f"  {'Binary':<20} {'MPI%':>8} {'Allgather(s)':>14} "
                     f"{'Bcast(s)':>10} {'Scatter(s)':>11}")
        for label, mp in mpip_data.items():
            lines.append(
                f"  {label:<20} {mp['mpi_pct']:>7.1f}% "
                f"{mp['allgather_s']:>14.4f} "
                f"{mp['bcast_s']:>10.4f} "
                f"{mp['scatter_s']:>11.4f}")

    if force_data:
        lines.append("\nLOAD BALANCE — PER-RANK FORCE COMPUTE TIME")
        lines.append("-" * 62)
        for label, ft in force_data.items():
            times = [x[1] for x in ft]
            if not times: continue
            mean = sum(times)/len(times)
            imb  = max(times)/mean if mean > 0 else 1.0
            lines.append(f"  {label}:")
            for rank, t in ft:
                lines.append(f"    Rank {rank}: {t:.5f} s")
            lines.append(f"    → Imbalance (max/mean): {imb:.3f}  "
                         f"[1.0=perfect  >1.1=imbalanced]")

    lines.append("\nSUMMARY")
    lines.append("-" * 62)
    if wall_times:
        fastest = min(wall_times, key=wall_times.get)
        lines.append(f"  Fastest binary : {fastest}  ({wall_times[fastest]:.3f} s)")
        if baseline:
            best_spd = baseline / wall_times[fastest]
            lines.append(f"  Best speedup   : {best_spd:.2f}×")
    lines.append(f"\n  Plots  → results/")
    lines.append(f"  CSV    → results/summary.csv")
    lines.append("=" * 62)

    report = "\n".join(lines)
    print("\n" + report)
    rpath = os.path.join(RESULTS_DIR, "summary_report.txt")
    with open(rpath, "w") as f:
        f.write(report)
    print(f"\n[Report] {rpath}")

# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    # Resolve sweep flags
    do_mpi_sweep = args.sweep_mpi or args.sweep_all
    do_omp_sweep = args.sweep_omp or args.sweep_all
    do_n_sweep   = args.sweep_n   or args.sweep_all
    do_any_sweep = do_mpi_sweep or do_omp_sweep or do_n_sweep

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── (A) Single-point bar-chart run ────────────────────────────────────────
    print("\n" + "="*62)
    print("  PHASE 1 — Single-point run")
    print("="*62)
    wall_times, mpip_data, force_data, theta_data, all_walls, N, T, P = \
        run_all_single(args)

    if wall_times:
        save_csv(wall_times, mpip_data, force_data, N, T, P)
        if HAS_MPL:
            print("\n── Single-point plots ──")
            plot_wall_time(wall_times)
            plot_speedup(wall_times, P)
            plot_efficiency(wall_times, P)
            plot_comm_overhead(mpip_data)
            plot_mpi_breakdown(mpip_data)
            plot_load_balance(force_data)
            plot_theta_evolution(theta_data)
            plot_run_variance(all_walls)
        write_report(wall_times, mpip_data, force_data, N, T, P)
    else:
        print("\n[WARN] No binaries succeeded in single-point run.")
        print("  Make sure binaries are compiled and in the current directory.")

    if not do_any_sweep:
        print("\n[INFO] No scaling sweeps requested.")
        print("  Use --sweep-mpi / --sweep-omp / --sweep-n / --sweep-all")
        print("  to run strong-scaling experiments.")
        sys.exit(0)

    # ── (B) MPI process scaling sweep ─────────────────────────────────────────
    if do_mpi_sweep:
        print("\n" + "="*62)
        print("  PHASE 2 — MPI process scaling sweep")
        print("="*62)
        mpi_results, mpi_pts, mpi_N, mpi_T = sweep_mpi(args)
        if HAS_MPL and mpi_results:
            print("\n── MPI scaling plots ──")
            plot_mpi_scaling_time(mpi_results, mpi_pts, mpi_N, mpi_T)
            plot_mpi_scaling_speedup(mpi_results, mpi_pts, mpi_N, mpi_T)

    # ── (C) OMP thread scaling sweep ──────────────────────────────────────────
    if do_omp_sweep:
        print("\n" + "="*62)
        print("  PHASE 3 — OMP thread scaling sweep")
        print("="*62)
        omp_results, omp_pts, omp_N, omp_T = sweep_omp(args)
        if HAS_MPL and omp_results:
            print("\n── OMP scaling plots ──")
            plot_omp_scaling_time(omp_results, omp_pts, omp_N, omp_T)
            plot_omp_scaling_speedup(omp_results, omp_pts, omp_N, omp_T)

    # ── (D) Particle-count sweep ───────────────────────────────────────────────
    if do_n_sweep:
        print("\n" + "="*62)
        print("  PHASE 4 — Particle-count scaling sweep")
        print("="*62)
        n_results, n_pts, n_P, n_T = sweep_n(args)
        if HAS_MPL and n_results:
            print("\n── N-scaling plots ──")
            plot_n_scaling(n_results, n_pts, n_P, n_T)

    print("\n" + "="*62)
    print("  ALL DONE — results saved to results/")
    print("="*62)
