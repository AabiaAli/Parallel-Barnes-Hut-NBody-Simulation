#===========================================================
# Barnes-Hut N-Body Simulation - Build Configuration
#===========================================================

MPICC = mpicc
CFLAGS = -O3
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0

# Source files
BASELINE_SRC = pnbody.c
ORB_SRC = nbody_orb.c
OMP_SRC = nbody_hybrid_omp.c
THETA_SRC = nbody_adaptive_theta.c
COMBINED_SRC = nbody_combined_all.c
PARALLEL_TREE_SRC = nbody_parallel_tree.c

# Executable names
BASELINE_TARGET = pnbody_baseline
ORB_TARGET = pnbody_orb
OMP_TARGET = pnbody_omp
THETA_TARGET = pnbody_theta
COMBINED_TARGET = pnbody_combined
PARALLEL_TREE_TARGET = pnbody_parallel_tree

# Default run parameters
N ?= 10000
T ?= 500
P ?= 4

#===========================================================
# Build Targets
#===========================================================

all: baseline orb omp theta combined parallel_tree
	@echo "All versions compiled successfully"

baseline:
	@echo "Building baseline version..."
	$(MPICC) $(CFLAGS) -o $(BASELINE_TARGET) $(BASELINE_SRC) $(LDFLAGS)

orb:
	@echo "Building ORB version..."
	$(MPICC) $(CFLAGS) -o $(ORB_TARGET) $(ORB_SRC) $(LDFLAGS)

omp:
	@echo "Building OpenMP hybrid version..."
	$(MPICC) $(CFLAGS) -fopenmp -o $(OMP_TARGET) $(OMP_SRC) $(LDFLAGS)

theta:
	@echo "Building Adaptive Theta version..."
	$(MPICC) $(CFLAGS) -o $(THETA_TARGET) $(THETA_SRC) $(LDFLAGS)

combined:
	@echo "Building Combined version..."
	$(MPICC) $(CFLAGS) -fopenmp -o $(COMBINED_TARGET) $(COMBINED_SRC) $(LDFLAGS)

parallel_tree:
	@echo "Building Parallel Tree version..."
	$(MPICC) $(CFLAGS) -fopenmp -o $(PARALLEL_TREE_TARGET) $(PARALLEL_TREE_SRC) $(LDFLAGS)

#===========================================================
# Run Targets
#===========================================================

run-baseline: baseline
	@echo "Running baseline with N=$(N), T=$(T), P=$(P)"
	mpirun -np $(P) ./$(BASELINE_TARGET) $(N) $(T)

run-orb: orb
	@echo "Running ORB with N=$(N), T=$(T), P=$(P)"
	mpirun -np $(P) ./$(ORB_TARGET) $(N) $(T)

run-omp: omp
	@echo "Running OpenMP with N=$(N), T=$(T), threads=$(P)"
	export OMP_NUM_THREADS=$(P) && mpirun -np 1 ./$(OMP_TARGET) $(N) $(T)

run-theta: theta
	@echo "Running Adaptive Theta with N=$(N), T=$(T), P=$(P)"
	mpirun -np $(P) ./$(THETA_TARGET) $(N) $(T)

run-combined: combined
	@echo "Running Combined with N=$(N), T=$(T), threads=$(P)"
	export OMP_NUM_THREADS=$(P) && mpirun -np 1 ./$(COMBINED_TARGET) $(N) $(T)

run-parallel_tree: parallel_tree
	@echo "Running Parallel Tree with N=$(N), T=$(T), P=$(P)"
	mpirun -np $(P) ./$(PARALLEL_TREE_TARGET) $(N) $(T)

#===========================================================
# Benchmark
#===========================================================

benchmark: baseline orb omp theta combined
	@echo "========== BENCHMARK START =========="
	@echo ""
	@echo "--- Baseline ---"
	time mpirun -np $(P) ./$(BASELINE_TARGET) $(N) $(T)
	@echo ""
	@echo "--- ORB Only ---"
	time mpirun -np $(P) ./$(ORB_TARGET) $(N) $(T)
	@echo ""
	@echo "--- OpenMP Only ---"
	export OMP_NUM_THREADS=$(P) && time mpirun -np 1 ./$(OMP_TARGET) $(N) $(T)
	@echo ""
	@echo "--- Adaptive Theta Only ---"
	time mpirun -np $(P) ./$(THETA_TARGET) $(N) $(T)
	@echo ""
	echo "--- Parallel Tree ---"
	time mpirun -np $(P) ./$(PARALLEL_TREE_TARGET) $(N) $(T)
	@echo ""
	@echo "--- Combined ---"
	export OMP_NUM_THREADS=$(P) && time mpirun -np 1 ./$(COMBINED_TARGET) $(N) $(T)
	@echo ""
	@echo "========== BENCHMARK COMPLETE =========="

#===========================================================
# Clean
#===========================================================

clean:
	rm -f $(BASELINE_TARGET) $(ORB_TARGET) $(OMP_TARGET) $(THETA_TARGET) $(COMBINED_TARGET) $(PARALLEL_TREE_TARGET)
	rm -f *.mpiP gmon.out pdist.dat

#===========================================================
# Help
#===========================================================

help:
	@echo "==========================================================="
	@echo "Barnes-Hut N-Body Simulation - Available Commands"
	@echo "==========================================================="
	@echo ""
	@echo "BUILD:"
	@echo "  make baseline        - Build baseline"
	@echo "  make orb             - Build ORB"
	@echo "  make omp             - Build OpenMP"
	@echo "  make theta           - Build Adaptive Theta"
	@echo "  make combined        - Build Combined"
	@echo "  make parallel_tree   - Build Parallel Tree"
	@echo "  make all             - Build all versions"
	@echo ""
	@echo "RUN (set N, T, P):"
	@echo "  make run-baseline    - Run baseline"
	@echo "  make run-orb         - Run ORB"
	@echo "  make run-omp         - Run OpenMP"
	@echo "  make run-theta       - Run Adaptive Theta"
	@echo "  make run-combined    - Run Combined"
	@echo "  make run-parallel_tree - Run Parallel Tree"
	@echo ""
	@echo "  Example: make run-combined N=50000 T=1000 P=4"
	@echo ""
	@echo "BENCHMARK:"
	@echo "  make benchmark N=10000 T=500 P=4"
	@echo ""
	@echo "OTHER:"
	@echo "  make clean           - Remove executables"
	@echo ""
	@echo "DEFAULT: N=$(N), T=$(T), P=$(P)"
	@echo "==========================================================="

.PHONY: all baseline orb omp theta combined parallel_tree clean help
.PHONY: run-baseline run-orb run-omp run-theta run-combined run-parallel_tree benchmark
