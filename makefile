#=============================================================================
# N-Body Simulation Makefile
#=============================================================================

MPICC = mpicc
CFLAGS = -O3
LDFLAGS = -lm
OMP_FLAGS = -fopenmp

BASELINE = pnbody
OPT_ORB = nbody_orb
OPT_HYBRID = nbody_hybrid
OPT_ADAPT = nbody_adaptive
SOFTENED = nbody_softened

BASELINE_SRC = pnbody.c
OPT_ORB_SRC = nbody_orb.c
OPT_HYBRID_SRC = nbody_hybrid_omp.c
OPT_ADAPT_SRC = nbody_adaptive_theta.c
SOFTENED_SRC = nbody_softened.c

N ?= 10000
T ?= 1000
P ?= 4
OMP_THREADS ?= 4

all: $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT) $(SOFTENED)

$(BASELINE): $(BASELINE_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OPT_ORB): $(OPT_ORB_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OPT_HYBRID): $(OPT_HYBRID_SRC)
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -o $@ $^ $(LDFLAGS)

$(OPT_ADAPT): $(OPT_ADAPT_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SOFTENED): $(SOFTENED_SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run-baseline: $(BASELINE)
	@echo "=== Baseline: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(BASELINE) $(N) $(T)

run-orb: $(OPT_ORB)
	@echo "=== ORB: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(OPT_ORB) $(N) $(T)

run-hybrid: $(OPT_HYBRID)
	@echo "=== Hybrid: N=$(N), T=$(T), P=$(P), OMP_THREADS=$(OMP_THREADS) ==="
	export OMP_NUM_THREADS=$(OMP_THREADS); mpirun -np $(P) ./$(OPT_HYBRID) $(N) $(T)

run-adaptive: $(OPT_ADAPT)
	@echo "=== Adaptive Theta: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(OPT_ADAPT) $(N) $(T)

run-softened: $(SOFTENED)
	@echo "=== Softened Baseline: N=$(N), T=$(T), P=$(P) ==="
	mpirun -np $(P) ./$(SOFTENED) $(N) $(T)

run-all: run-baseline run-orb run-hybrid run-adaptive run-softened

time-baseline: $(BASELINE)
	@echo "Timing Baseline: N=$(N), T=$(T), P=$(P)"
	/usr/bin/time -f "Elapsed: %e seconds" mpirun -np $(P) ./$(BASELINE) $(N) $(T)

time-softened: $(SOFTENED)
	@echo "Timing Softened Baseline: N=$(N), T=$(T), P=$(P)"
	/usr/bin/time -f "Elapsed: %e seconds" mpirun -np $(P) ./$(SOFTENED) $(N) $(T)

time-orb: $(OPT_ORB)
	@echo "Timing ORB: N=$(N), T=$(T), P=$(P)"
	/usr/bin/time -f "Elapsed: %e seconds" mpirun -np $(P) ./$(OPT_ORB) $(N) $(T)

time-hybrid: $(OPT_HYBRID)
	@echo "Timing Hybrid: N=$(N), T=$(T), P=$(P), OMP_THREADS=$(OMP_THREADS)"
	export OMP_NUM_THREADS=$(OMP_THREADS); /usr/bin/time -f "Elapsed: %e seconds" mpirun -np $(P) ./$(OPT_HYBRID) $(N) $(T)

time-adaptive: $(OPT_ADAPT)
	@echo "Timing Adaptive: N=$(N), T=$(T), P=$(P)"
	/usr/bin/time -f "Elapsed: %e seconds" mpirun -np $(P) ./$(OPT_ADAPT) $(N) $(T)

time-all: time-baseline time-softened time-orb time-hybrid time-adaptive

clean:
	rm -f $(BASELINE) $(OPT_ORB) $(OPT_HYBRID) $(OPT_ADAPT) $(SOFTENED)
	rm -f pdist*.dat

clean-dat:
	rm -f pdist*.dat

clean-all: clean

help:
	@echo "=========================================="
	@echo "N-Body Simulation - Makefile Help"
	@echo "=========================================="
	@echo ""
	@echo "BUILD: make              - Build all versions"
	@echo "       make clean        - Remove executables"
	@echo ""
	@echo "RUN:   make run-baseline P=4 N=10000 T=500"
	@echo "       make run-orb P=4 N=10000 T=500"
	@echo "       make run-hybrid P=4 OMP_THREADS=4 N=10000 T=500"
	@echo "       make run-adaptive P=4 N=10000 T=500"
	@echo "       make run-softened P=4 N=10000 T=500"
	@echo ""
	@echo "TIME:  make time-baseline P=4 N=10000 T=500"
	@echo "       make time-adaptive P=4 N=10000 T=500"
	@echo ""

.DEFAULT_GOAL := all

.PHONY: all clean clean-dat clean-all help run-baseline run-orb run-hybrid run-adaptive run-softened run-all time-baseline time-softened time-orb time-hybrid time-adaptive time-all accuracy-test