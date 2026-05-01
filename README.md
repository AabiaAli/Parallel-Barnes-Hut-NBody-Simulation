# Parallel N-Body Simulation with Barnes-Hut Algorithm

[![MPI](https://img.shields.io/badge/MPI-Parallel-blue.svg)](https://www.mpi-forum.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Multithreading-green.svg)](https://www.openmp.org/)
[![C](https://img.shields.io/badge/Language-C-orange.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a parallel N-body gravitational simulation using the **Barnes-Hut algorithm** for force approximation. The implementation leverages **MPI (Message Passing Interface)** for distributed memory parallelism and **OpenMP** for shared memory parallelism, exploring various optimization strategies to achieve significant speedups while maintaining acceptable accuracy.

**Authors**: Zara Noor (23i-0681) and Aabia Ali (23i-0704)

**Course**: Parallel and Distributed Computing

---

## Problem Statement

The N-body problem requires computing gravitational interactions between N particles over discrete time steps. For each particle i, the gravitational force from all other particles is:

Fᵢ = G Σⱼ≠ᵢ (mᵢ mⱼ / rᵢⱼ²) · r̂ᵢⱼ


where:
- G is the gravitational constant (6.67300 × 10⁻¹¹ m³ kg⁻¹ s⁻²)
- mᵢ, mⱼ are masses of particles i and j
- rᵢⱼ is the distance between particles i and j
- r̂ᵢⱼ is the unit vector from i to j

The naive O(N²) particle-particle approach becomes computationally infeasible for large N. This project addresses this challenge through:

1. **Barnes-Hut Algorithm** - Reduces complexity from O(N²) to O(N log N)
2. **MPI Parallelization** - Distributes work across multiple nodes
3. **OpenMP Parallelization** - Shared memory parallelism within nodes
4. **Optimization Strategies** - ORB load balancing, adaptive theta, gravitational softening

---

## Implementations

### 1. Baseline Implementation (`pnbody.c`)

- Static block partitioning of particles across MPI processes
- Barnes-Hut tree built on each process
- Fixed opening angle (θ = 1.0)
- O(N log N) time complexity per iteration

### 2. ORB Load Balancing (`nbody_orb.c`)

**Optimization**: Orthogonal Recursive Bisection
- Recursively partitions space to ensure equal particles per process
- Dynamically rebuilds partition after each time step
- **Result**: 1.28× speedup with perfect accuracy

### 3. Hybrid MPI+OpenMP (`nbody_hybrid_omp.c`)

**Optimization**: Two-level parallelism
- MPI for inter-node communication
- OpenMP for intra-node force computation
- **Result**: 1.42× speedup with perfect accuracy

### 4. Adaptive Theta (`nbody_adaptive_theta.c`)

**Optimization**: Dynamic opening angle adjustment
- θ adapts per rank based on local density and load imbalance
- Range: [0.8, 1.2] with step size 0.02
- **Result**: 2.02× speedup with 40% accuracy trade-off

### 5. Gravitational Softening (`nbody_softened.c`)

**Optimization**: Numerical stability
- Adds softening length (ε = 1000m) to prevent singularities
- Smoothes forces when particles get extremely close
- **Result**: 1.32× speedup with improved stability

---

## Performance Results

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Particles | 20,000 |
| Time Steps | 100 |
| MPI Processes | 2 |
| OpenMP Threads | 2 (Hybrid only) |
| Domain Size | 1,000,000 m |

### Speedup & Accuracy Comparison

| Implementation | Time (s) | Speedup | Efficiency | Accuracy (RMS Error) |
|----------------|----------|---------|------------|----------------------|
| Baseline | 25.88 | 1.00× | 100% | Reference |
| ORB | 20.23 | **1.28×** | 64% | 0.00 m ✓ |
| Hybrid (MPI+OMP) | 18.19 | **1.42×** | 71% | 0.00 m ✓ |
| Adaptive Theta | 12.81 | **2.02×** | 101% | 335,988 m ⚠ |
| Softened Baseline | 19.61 | **1.32×** | 66% | 0.00 m ✓ |

### Key Insights

- **Best Accuracy**: ORB & Hybrid (0% error, mathematically identical to baseline)
- **Best Speed**: Adaptive Theta (2× faster, 40% error trade-off)
- **Best Balance**: Hybrid (1.42× speedup, perfect accuracy)
- **Super-linear Speedup**: Adaptive Theta (101% efficiency) due to better cache utilization

---

## Getting Started

### Prerequisites

Install required packages on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install mpich libmpich-dev
sudo apt-get install gcc make
sudo apt-get install python3 python3-numpy
```
Verify installations:
```bash

mpicc --version
gcc --version
python3 --version
```
Installation

Clone the repository and build all versions:

```bash
git clone <repository-url>
cd nbody-simulation
make clean
make
```
Verify builds:
```bash
ls -la pnbody nbody_orb nbody_hybrid nbody_adaptive nbody_softened
```

Usage Guide
Quick Run Examples

Run baseline with 2 processes, 10000 particles, 1000 steps:
```bash
make run-baseline P=2 N=10000 T=1000
```
