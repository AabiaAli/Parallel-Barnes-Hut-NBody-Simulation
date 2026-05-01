#!/usr/bin/env python3
"""
N-Body Simulation Benchmark Script
Compares Baseline, ORB, Hybrid, Adaptive, and Softened versions
"""

import subprocess
import numpy as np
import sys
import time
import os

def run_simulation(executable, N, T, P, omp_threads=1):
    """Run a simulation and return elapsed time"""
    cmd = f"mpirun -np {P} ./{executable} {N} {T}"
    
    if omp_threads > 1:
        cmd = f"export OMP_NUM_THREADS={omp_threads}; " + cmd
    
    start = time.time()
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - start
    return elapsed

def read_positions(filename):
    """Read positions from output file"""
    positions = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if 'px=' in line:
                    parts = line.strip().split(', ')
                    px = float(parts[0].split('=')[1])
                    py = float(parts[1].split('=')[1])
                    pz = float(parts[2].split('=')[1])
                    positions.append([px, py, pz])
    except:
        return None
    return np.array(positions)

def compare_accuracy(ref_file, test_file, name):
    """Compare two position files and return RMS error"""
    ref = read_positions(ref_file)
    test = read_positions(test_file)
    
    if ref is None or test is None:
        return None
    
    if len(ref) != len(test):
        return None
    
    # Sort by position magnitude to match particles
    ref_mag = np.sqrt(np.sum(ref**2, axis=1))
    test_mag = np.sqrt(np.sum(test**2, axis=1))
    
    ref_sorted = ref[np.argsort(ref_mag)]
    test_sorted = test[np.argsort(test_mag)]
    
    diff = ref_sorted - test_sorted
    rms = np.sqrt(np.mean(diff**2))
    return rms

def main():
    print("\n" + "="*60)
    print("N-BODY SIMULATION BENCHMARK")
    print("="*60)
    
    # Get parameters from user
    try:
        N = int(input("\nNumber of particles (default 5000): ") or "5000")
        T = int(input("Number of time steps (default 100): ") or "100")
        P = int(input("MPI processes (default 2): ") or "2")
        omp = int(input("OpenMP threads for Hybrid (default 2): ") or "2")
    except:
        print("Invalid input, using defaults")
        N, T, P, omp = 5000, 100, 2, 2
    
    print(f"\nRunning benchmark with:")
    print(f"  Particles: {N}")
    print(f"  Steps: {T}")
    print(f"  MPI Processes: {P}")
    print(f"  OpenMP Threads: {omp}")
    
    # Define versions to test
    versions = [
        ("Baseline", "pnbody", 1, 1),
        ("ORB", "nbody_orb", 1, 1),
        ("Hybrid", "nbody_hybrid", omp, 1),
        ("Adaptive", "nbody_adaptive", 1, 1),
        ("Softened", "nbody_softened", 1, 1),
    ]
    
    results = {}
    
    # Run baseline first (reference for accuracy)
    print("\n" + "-"*60)
    print("Running simulations...")
    print("-"*60)
    
    for name, exe, omp_threads, mpi_procs in versions:
        print(f"  Running {name}...", end=" ", flush=True)
        
        # Use specified MPI processes
        proc = P if name != "Baseline" else 1
        
        # Handle Hybrid specially
        if name == "Hybrid":
            time_taken = run_simulation(exe, N, T, proc, omp_threads)
        else:
            time_taken = run_simulation(exe, N, T, proc, 1)
        
        results[name] = {
            'time': time_taken,
            'file': f"pdist_{name.lower()}.dat" if name != "Baseline" and name != "Softened" else ("pdist.dat" if name == "Softened" else "pdist.dat"),
            'exe': exe
        }
        
        # Rename output files for non-baseline versions
        if name == "ORB":
            os.system("mv pdist_orb.dat pdist_orb.dat 2>/dev/null")
        elif name == "Hybrid":
            os.system("mv pdist_hybrid.dat pdist_hybrid.dat 2>/dev/null")
        elif name == "Adaptive":
            os.system("mv pdist_adaptive.dat pdist_adaptive.dat 2>/dev/null")
        
        print(f"{time_taken:.2f}s")
    
    # Get baseline time for speedup calculations
    baseline_time = results["Baseline"]['time']
    
    # Calculate accuracy (compare all against baseline)
    print("\n" + "-"*60)
    print("ACCURACY vs BASELINE (lower is better)")
    print("-"*60)
    
    # Run baseline again to get reference positions
    subprocess.run(f"mpirun -np 1 ./pnbody {N} {T}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.system("mv pdist.dat pdist_baseline_ref.dat")
    
    for name in ["ORB", "Hybrid", "Adaptive", "Softened"]:
        # Get the output file for this version
        if name == "Softened":
            test_file = "pdist.dat"
        else:
            test_file = f"pdist_{name.lower()}.dat"
        
        # Ensure file exists
        if not os.path.exists(test_file):
            # Re-run if needed
            if name == "Hybrid":
                run_simulation(results[name]['exe'], N, T, P, omp)
            else:
                run_simulation(results[name]['exe'], N, T, P, 1)
        
        rms = compare_accuracy("pdist_baseline_ref.dat", test_file, name)
        if rms is not None:
            results[name]['accuracy'] = rms
            if rms < 1e-3:
                status = "✓ EXCELLENT"
            elif rms < 1e-2:
                status = "✓ GOOD"
            elif rms < 1e-1:
                status = "⚠ MODERATE"
            else:
                status = "✗ LARGE (trade-off)"
            print(f"  {name:12s}: {rms:10.2f} m  {status}")
        else:
            results[name]['accuracy'] = None
            print(f"  {name:12s}: Could not compare")
    
    # Calculate speedups
    print("\n" + "-"*60)
    print("PERFORMANCE vs BASELINE (higher is better)")
    print("-"*60)
    print(f"  {'Version':12s} {'Time (s)':>10s} {'Speedup':>10s} {'Efficiency':>10s}")
    print("  " + "-"*45)
    
    for name, data in results.items():
        time_taken = data['time']
        speedup = baseline_time / time_taken
        efficiency = speedup / P if name != "Baseline" else 1.0
        print(f"  {name:12s} {time_taken:10.2f} {speedup:9.2f}x {efficiency:9.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find best speedup
    best_speedup = 0
    best_name = ""
    for name, data in results.items():
        if name != "Baseline":
            speedup = baseline_time / data['time']
            if speedup > best_speedup:
                best_speedup = speedup
                best_name = name
    
    print(f"\n  Fastest: {best_name} ({best_speedup:.2f}x speedup)")
    
    # Find best accuracy (excluding baseline)
    best_accuracy = float('inf')
    best_acc_name = ""
    for name in ["ORB", "Hybrid", "Adaptive", "Softened"]:
        if name in results and results[name].get('accuracy') is not None:
            acc = results[name]['accuracy']
            if acc < best_accuracy:
                best_accuracy = acc
                best_acc_name = name
    
    if best_acc_name:
        print(f"  Most Accurate: {best_acc_name} ({best_accuracy:.2f} m error)")
    
    # Recommendations
    print("\n" + "-"*60)
    print("RECOMMENDATIONS")
    print("-"*60)
    print("  • ORB & Hybrid: Best for accuracy-critical applications")
    print("  • Adaptive Theta: Best for speed-critical applications")
    print("  • Softened Baseline: Prevents numerical explosions")
    print("\n" + "="*60)
    
    # Clean up
    os.system("rm -f pdist*.dat pdist_baseline_ref.dat")

if __name__ == "__main__":
    main()