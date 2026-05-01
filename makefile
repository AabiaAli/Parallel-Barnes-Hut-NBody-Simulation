#=============================================================================
# N-Body Simulation Makefile
# Baseline + 3 Optimizations: ORB, Hybrid MPI+OpenMP, Adaptive Theta
#=============================================================================

# Compiler and flags
MPICC = mpicc
CFLAGS = -O3
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0
PROF_FLAGS = -pg -O3

# OpenMP flags
OMP_FLAGS = -fopenmp

# mpiP paths (optional)
MPIP_HOME ?= $(HOME)/mpip
MPIP_LIB = $(MPIP_HOME)/lib/libmpiP.so

# Target executables
BASELINE = pnbody
OPT_ORB = nbody_orb
OPT_HYBRID = nbody_hybrid
OPT_ADAPT = nbody_adaptive

# Source files
BASELINE_SRC = pnbody.c
OPT_ORB_SRC = nbody_orb.c
OPT_HYBRID_SRC = nbody_hybrid_omp.c
OPT_ADAPT_SRC = nbody_adaptive_theta.c

# Default parameters
N ?= 10000
T ?= 1000
P ?= 4
OMP_THREADS ?= 4

#=============================================================================
# Build targets
#=============================================================================

all: $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT)
	@echo ""
	@echo "All versions compiled successfully!"
	@echo "  - Baseline:      $(BASELINE)"
	@echo "  - ORB:           $(OPT_ORB)"
	@echo "  - Hybrid OMP:    $(OPT_HYBRID)"
	@echo "  - Adaptive Theta: $(OPT_ADAPT)"
	@echo ""

# Baseline version (original)
$(BASELINE): $(BASELINE_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Optimization 1: ORB (Orthogonal Recursive Bisection)
$(OPT_ORB): $(OPT_ORB_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Optimization 2: Hybrid MPI + OpenMP
$(OPT_HYBRID): $(OPT_HYBRID_SRC)
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -o $@ $^ $(LDFLAGS)

# Optimization 3: Adaptive Theta per process
$(OPT_ADAPT): $(OPT_ADAPT_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#=============================================================================
# Debug builds
#=============================================================================

debug: CFLAGS = $(DEBUG_FLAGS)
debug: $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT)
	@echo "Debug versions compiled"

debug-baseline: CFLAGS = $(DEBUG_FLAGS)
debug-baseline: $(BASELINE)

debug-orb: CFLAGS = $(DEBUG_FLAGS)
debug-orb: $(OPT_ORB)

debug-hybrid: CFLAGS = $(DEBUG_FLAGS)
debug-hybrid: $(OPT_HYBRID)

debug-adaptive: CFLAGS = $(DEBUG_FLAGS)
debug-adaptive: $(OPT_ADAPT)

#=============================================================================
# Profiling builds (gprof)
#=============================================================================

prof: CFLAGS = $(PROF_FLAGS)
prof: $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT)
	@echo "Profiling versions compiled"

prof-baseline: CFLAGS = $(PROF_FLAGS)
prof-baseline: $(BASELINE)

prof-orb: CFLAGS = $(PROF_FLAGS)
prof-orb: $(OPT_ORB)

prof-hybrid: CFLAGS = $(PROF_FLAGS)
prof-hybrid: $(OPT_HYBRID)

prof-adaptive: CFLAGS = $(PROF_FLAGS)
prof-adaptive: $(OPT_ADAPT)

#=============================================================================
# Run targets
#=============================================================================

# Run baseline
run-baseline: $(BASELINE)
	@echo "=== Running Baseline: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(BASELINE) $(N) $(T)

# Run ORB optimization
run-orb: $(OPT_ORB)
	@echo "=== Running ORB (Spatial Decomposition): N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(OPT_ORB) $(N) $(T)

# Run Hybrid (MPI+OpenMP)
run-hybrid: $(OPT_HYBRID)
	@echo "=== Running Hybrid MPI+OpenMP: N=$(N), T=$(T), P=$(P), OMP_THREADS=$(OMP_THREADS) ==="
	export OMP_NUM_THREADS=$(OMP_THREADS); \
	mpirun -np $(P) ./$(OPT_HYBRID) $(N) $(T)

# Run Adaptive Theta
run-adaptive: $(OPT_ADAPT)
	@echo "=== Running Adaptive Theta: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(OPT_ADAPT) $(N) $(T)

# Run all versions sequentially
run-all: run-baseline run-orb run-hybrid run-adaptive
	@echo "All simulations completed"

# Default run (baseline for backward compatibility)
run: run-baseline

# Custom run with parameters (usage: make run-custom N=20000 T=500 P=8)
run-custom: $(BASELINE)
	mpirun -np $(P) ./$(BASELINE) $(N) $(T)

#=============================================================================
# mpiP profiling targets
#=============================================================================

# Run baseline with mpiP
mpip-baseline: $(BASELINE)
	@echo "=== Profiling Baseline with mpiP ==="
	LD_PRELOAD=$(MPIP_LIB) mpirun -np $(P) ./$(BASELINE) $(N) $(T)

# Run ORB with mpiP
mpip-orb: $(OPT_ORB)
	@echo "=== Profiling ORB with mpiP ==="
	LD_PRELOAD=$(MPIP_LIB) mpirun -np $(P) ./$(OPT_ORB) $(N) $(T)

# Run Hybrid with mpiP
mpip-hybrid: $(OPT_HYBRID)
	@echo "=== Profiling Hybrid with mpiP ==="
	export OMP_NUM_THREADS=$(OMP_THREADS); \
	LD_PRELOAD=$(MPIP_LIB) mpirun -np $(P) ./$(OPT_HYBRID) $(N) $(T)

# Run Adaptive with mpiP
mpip-adaptive: $(OPT_ADAPT)
	@echo "=== Profiling Adaptive Theta with mpiP ==="
	LD_PRELOAD=$(MPIP_LIB) mpirun -np $(P) ./$(OPT_ADAPT) $(N) $(T)

# Default mpip (baseline)
mpip: mpip-baseline

#=============================================================================
# Scaling tests
#=============================================================================

# Strong scaling test (fixed problem size, varying processes)
scaling-baseline: $(BASELINE)
	@echo "=== Strong Scaling: Baseline (N=10000, T=500) ==="
	@echo "Procs\tTime(s)"
	@for p in 1 2 4 8; do \
		echo -n "$$p\t"; \
		(time mpirun -np $$p ./$(BASELINE) 10000 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

scaling-orb: $(OPT_ORB)
	@echo "=== Strong Scaling: ORB (N=10000, T=500) ==="
	@echo "Procs\tTime(s)"
	@for p in 1 2 4 8; do \
		echo -n "$$p\t"; \
		(time mpirun -np $$p ./$(OPT_ORB) 10000 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

scaling-hybrid: $(OPT_HYBRID)
	@echo "=== Strong Scaling: Hybrid (N=10000, T=500, OMP=2) ==="
	@echo "Procs\tTime(s)"
	@for p in 1 2 4 8; do \
		echo -n "$$p\t"; \
		(export OMP_NUM_THREADS=2; time mpirun -np $$p ./$(OPT_HYBRID) 10000 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

scaling-adaptive: $(OPT_ADAPT)
	@echo "=== Strong Scaling: Adaptive Theta (N=10000, T=500) ==="
	@echo "Procs\tTime(s)"
	@for p in 1 2 4 8; do \
		echo -n "$$p\t"; \
		(time mpirun -np $$p ./$(OPT_ADAPT) 10000 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

# Run all scaling tests
scaling-all: scaling-baseline scaling-orb scaling-hybrid scaling-adaptive

# Original scaling target (baseline only, for backward compatibility)
scaling: scaling-baseline

# Weak scaling test (problem size proportional to processes)
weak-scaling-baseline: $(BASELINE)
	@echo "=== Weak Scaling: Baseline (base N=5000 per proc) ==="
	@echo "Procs\tN\tTime(s)"
	@for p in 1 2 4 8; do \
		n=$$((5000 * $$p)); \
		echo -n "$$p\t$$n\t"; \
		(time mpirun -np $$p ./$(BASELINE) $$n 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

weak-scaling-orb: $(OPT_ORB)
	@echo "=== Weak Scaling: ORB (base N=5000 per proc) ==="
	@echo "Procs\tN\tTime(s)"
	@for p in 1 2 4 8; do \
		n=$$((5000 * $$p)); \
		echo -n "$$p\t$$n\t"; \
		(time mpirun -np $$p ./$(OPT_ORB) $$n 500) 2>&1 | grep real | sed 's/real\s*//g'; \
	done

#=============================================================================
# Speedup comparison
#=============================================================================

speedup: $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT)
	@echo ""
	@echo "============================================================"
	@echo "SPEEDUP COMPARISON: N=$(N), T=$(T)"
	@echo "============================================================"
	@echo ""
	@echo "Measuring baseline time with 1 process..."
	@t_baseline=$$( (time mpirun -np 1 ./$(BASELINE) $(N) $(T)) 2>&1 | grep real | awk '{print $$2}' | sed 's/[a-z]//g'); \
	echo "Baseline time (1 proc): $$t_baseline s"; \
	echo ""; \
	echo "Configuration\t\tTime(s)\tSpeedup"; \
	echo "----------------------------------------"; \
	for p in 1 2 4 8; do \
		t=$$( (time mpirun -np $$p ./$(BASELINE) $(N) $(T)) 2>&1 | grep real | awk '{print $$2}' | sed 's/[a-z]//g'); \
		speedup=$$(echo "scale=2; $$t_baseline / $$t" | bc); \
		printf "Baseline (%2d procs)\t%.4f\t%.2f\n" $$p $$t $$speedup; \
	done; \
	echo ""; \
	for p in 1 2 4 8; do \
		t=$$( (time mpirun -np $$p ./$(OPT_ORB) $(N) $(T)) 2>&1 | grep real | awk '{print $$2}' | sed 's/[a-z]//g'); \
		speedup=$$(echo "scale=2; $$t_baseline / $$t" | bc); \
		printf "ORB (%2d procs)\t\t%.4f\t%.2f\n" $$p $$t $$speedup; \
	done; \
	echo ""; \
	for p in 1 2 4 8; do \
		t=$$( (export OMP_NUM_THREADS=2; time mpirun -np $$p ./$(OPT_HYBRID) $(N) $(T)) 2>&1 | grep real | awk '{print $$2}' | sed 's/[a-z]//g'); \
		speedup=$$(echo "scale=2; $$t_baseline / $$t" | bc); \
		printf "Hybrid (%2d procs)\t%.4f\t%.2f\n" $$p $$t $$speedup; \
	done; \
	echo ""; \
	for p in 1 2 4 8; do \
		t=$$( (time mpirun -np $$p ./$(OPT_ADAPT) $(N) $(T)) 2>&1 | grep real | awk '{print $$2}' | sed 's/[a-z]//g'); \
		speedup=$$(echo "scale=2; $$t_baseline / $$t" | bc); \
		printf "Adaptive (%2d procs)\t%.4f\t%.2f\n" $$p $$t $$speedup; \
	done

#=============================================================================
# Large problem tests
#=============================================================================

large-baseline: $(BASELINE)
	@echo "=== Large test: Baseline N=50000, T=500, P=4 ==="
	time mpirun -np 4 ./$(BASELINE) 50000 500

large-orb: $(OPT_ORB)
	@echo "=== Large test: ORB N=50000, T=500, P=4 ==="
	time mpirun -np 4 ./$(OPT_ORB) 50000 500

large-hybrid: $(OPT_HYBRID)
	@echo "=== Large test: Hybrid N=50000, T=500, P=4, OMP=4 ==="
	export OMP_NUM_THREADS=4; time mpirun -np 4 ./$(OPT_HYBRID) 50000 500

large-adaptive: $(OPT_ADAPT)
	@echo "=== Large test: Adaptive Theta N=50000, T=500, P=4 ==="
	time mpirun -np 4 ./$(OPT_ADAPT) 50000 500

large-all: large-baseline large-orb large-hybrid large-adaptive

# Original large target (baseline only)
large: large-baseline

# Large test with mpiP
large-mpip: $(BASELINE)
	@echo "=== Profiling N=50000, T=500, P=4 with mpiP ==="
	LD_PRELOAD=$(MPIP_LIB) mpirun -np 4 ./$(BASELINE) 50000 500

#=============================================================================
# Clean targets
#=============================================================================

clean:
	rm -f $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT)
	rm -f *.o gmon.out *.mpiP pdist*.dat timing.txt
	@echo "Cleaned executables and data files"

clean-dat:
	rm -f pdist*.dat pdist_*.dat timing.txt
	@echo "Cleaned data files"

clean-all: clean clean-dat

#=============================================================================
# Utility targets
#=============================================================================

info:
	@echo "========================================"
	@echo "N-Body Simulation Build Configuration"
	@echo "========================================"
	@echo "MPI Compiler:     $(MPICC)"
	@echo "CFLAGS:           $(CFLAGS)"
	@echo "LDFLAGS:          $(LDFLAGS)"
	@echo "OMP Flags:        $(OMP_FLAGS)"
	@echo ""
	@echo "Default Parameters:"
	@echo "  N (particles):   $(N)"
	@echo "  T (steps):       $(T)"
	@echo "  P (MPI procs):   $(P)"
	@echo "  OMP_THREADS:     $(OMP_THREADS)"
	@echo ""
	@echo "Available targets:"
	@echo "  make            - Build all versions"
	@echo "  make debug      - Build debug versions"
	@echo "  make prof       - Build profiling versions"
	@echo "  make run-*      - Run specific version"
	@echo "  make run-all    - Run all versions"
	@echo "  make scaling-*  - Strong scaling tests"
	@echo "  make speedup    - Compare speedups"
	@echo "  make large-*    - Large problem tests"
	@echo "  make mpip-*     - mpiP profiling"
	@echo "  make clean      - Remove executables"
	@echo "  make clean-dat  - Remove data files"
	@echo "========================================"

help: info

#=============================================================================
# Default target
#=============================================================================

.DEFAULT_GOAL := all

# Prevent confusion with similarly named files
.PHONY: all clean clean-dat clean-all info help debug prof \
        run run-baseline run-orb run-hybrid run-adaptive run-all run-custom \
        scaling scaling-baseline scaling-orb scaling-hybrid scaling-adaptive scaling-all \
        weak-scaling-baseline weak-scaling-orb \
        speedup \
        large large-baseline large-orb large-hybrid large-adaptive large-all large-mpip \
        mpip mpip-baseline mpip-orb mpip-hybrid mpip-adaptive \
        debug-baseline debug-orb debug-hybrid debug-adaptive \
        prof-baseline prof-orb prof-hybrid prof-adaptive