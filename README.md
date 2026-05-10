# Parallel N-Body Simulation — Barnes-Hut Algorithm

MPI-based Barnes-Hut N-body gravitational simulation with four parallel
optimisations and a benchmarking suite.

---

## Overview

This project implements the Barnes-Hut O(N log N) algorithm for simulating
gravitational N-body systems in parallel using MPI. Four optimisations are
implemented independently and then combined into a single binary.

| Binary | Optimisation |
|---|---|
| `pnbody_baseline` | Pure MPI, index-based partition |
| `pnbody_orb` | Orthogonal Recursive Bisection (load balancing) |
| `pnbody_theta` | Adaptive opening angle (per-rank θ) |
| `pnbody_omp` | Hybrid MPI + OpenMP (parallel force loop) |
| `pnbody_parallel_tree` | Arena allocator (no malloc/free per step) |
| `pnbody_combined` | All four optimisations combined |

---

## Requirements

- GCC 9+ with OpenMP support
- Open MPI 4.0+ (or any MPI-2 compliant library)
- Python 3.8+ with `matplotlib` (for benchmarking only)
- Linux / macOS

Install dependencies on Ubuntu:
```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-matplotlib
```

---

## Building

Compile all binaries with:
```bash
make
```

Or compile individually:
```bash
# Pure MPI binaries
mpicc -O2 -o pnbody_baseline   nbody_baseline.c        -lm
mpicc -O2 -o pnbody_orb        nbody_orb.c             -lm
mpicc -O2 -o pnbody_theta      nbody_adaptive_theta.c  -lm

# Hybrid MPI + OpenMP binaries
mpicc -O2 -fopenmp -o pnbody_omp           nbody_hybrid_omp.c      -lm
mpicc -O2 -fopenmp -o pnbody_parallel_tree nbody_parallel_tree.c   -lm
mpicc -O2 -fopenmp -o pnbody_combined      nbody_combined.c        -lm
```

---

## Running

```bash
mpirun -np  ./  
```

**Examples:**
```bash
# Baseline — 4 ranks, 50000 particles, 200 timesteps
mpirun -np 4 ./pnbody_baseline 50000 200

# Hybrid binary — 4 MPI ranks × 4 OMP threads
export OMP_NUM_THREADS=4
mpirun -np 4 ./pnbody_combined 50000 200
```

**Constraints:**
- N must be divisible by the number of MPI ranks
- Output position files are written to `results/` — create it first:

```bash
mkdir -p results
```

---

## Benchmarking

The benchmarking script runs all binaries, collects timing data, and produces
plots and CSV files in `results/`.

**Single-point run** (default N=50000, T=200, 4 MPI ranks, 4 OMP threads):
```bash
python3 benchmark.py
```

**Full scaling sweep:**
```bash
python3 benchmark.py --sweep-all
```

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 50000 | Particle count |
| `--t` | 200 | Timesteps |
| `--p-mpi` | 4 | MPI ranks |
| `--p-omp` | 4 | OMP threads per rank |
| `--runs` | 3 | Repeats per config (median taken) |
| `--sweep-mpi` | off | MPI rank scaling sweep |
| `--sweep-omp` | off | OMP thread scaling sweep |
| `--sweep-n` | off | Particle count scaling sweep |
| `--sweep-all` | off | Enable all three sweeps |
| `--t-sweep` | 100 | Timesteps used during sweeps |
| `--quick` | off | Smoke test: N=5000, T=50, P=2 |

**Recommended minimum N for meaningful results: 50000.**
Below 25000 particles the communication overhead dominates and differences
between variants are not statistically significant.

---

## Output Files

| File | Contents |
|---|---|
| `results/summary.csv` | Single-point wall times, speedup, efficiency |
| `results/scaling_mpi.csv` | MPI rank sweep raw data |
| `results/scaling_omp.csv` | OMP thread sweep raw data |
| `results/scaling_n.csv` | Particle count sweep raw data |
| `results/wall_time.png` | Bar chart: wall time per binary |
| `results/speedup.png` | Bar chart: speedup over baseline |
| `results/scaling_mpi.png` | Line chart: MPI strong scaling |
| `results/scaling_omp.png` | Line chart: OMP thread scaling |
| `results/scaling_n_time.png` | Line chart: wall time vs N |
| `results/scaling_n_relative_speedup.png` | Line chart: speedup vs baseline across N |
| `results/*_positions.dat` | Final particle positions per binary |

---

## Project Structure

    .
    ├── nbody_baseline.c          # Baseline MPI implementation
    ├── nbody_orb.c               # ORB load balancing
    ├── nbody_adaptive_theta.c    # Adaptive opening angle
    ├── nbody_hybrid_omp.c        # Hybrid MPI + OpenMP
    ├── nbody_parallel_tree.c     # Arena allocator + OpenMP
    ├── nbody_combined.c          # All optimisations combined
    ├── benchmark.py              # Benchmarking and plotting script
    ├── Makefile                  # Build all binaries
    └── results/                  # Output directory (create before running)

---

## Experimental Setup

| Item | Detail |
|---|---|
| Host CPU | Intel Core i7-1165G7 @ 2.80 GHz (4 cores / 8 threads) |
| VM | VirtualBox, Ubuntu 20.04 LTS, 8 vCPUs, 7.6 GB RAM |
| Compiler | GCC 9.4.0 |
| MPI | Open MPI 4.0.3 |
| Flags | `-O2 -fopenmp` |

**Recommended thread configuration for this hardware:**
`mpirun -np 4` with `OMP_NUM_THREADS=2` (4 × 2 = 8 threads = 8 vCPUs).
Setting OMP threads higher causes over-subscription and performance degradation.

---

## Key Results

- **Best MPI speedup:** 2.42× (Combined, P=4)
- **Best large-N speedup:** 1.41× over baseline at N=200k (Combined)
- **OMP sweet spot:** OMP=2 with 4 MPI ranks on this hardware
- **Most consistent variant:** Combined — only binary above 1.4× for all N ≥ 50k
- **ORB note:** Best at small N (2.98× at N=10k) but advantage shrinks at large N as particle distributions naturally balance

---

## References

- Barnes, J. & Hut, P. (1986). A hierarchical O(N log N) force-calculation algorithm. *Nature*, 324, 446–449.
- Open MPI: https://www.open-mpi.org
- OpenMP specification: https://www.openmp.org/specifications/
