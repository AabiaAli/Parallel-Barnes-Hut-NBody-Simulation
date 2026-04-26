# Compiler and flags
MPICC = mpicc
CFLAGS = -O3
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0
PROF_FLAGS = -pg -O3

# Target executable
TARGET = pnbody

# Source files
SRCS = pnbody.c

# mpiP paths
MPIP_HOME ?= $(HOME)/mpip
MPIP_LIB = $(MPIP_HOME)/lib/libmpiP.so

# Default target
all: $(TARGET)

# Regular build
$(TARGET): $(SRCS)
	$(MPICC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Debug build (with symbols)
debug:
	$(MPICC) $(DEBUG_FLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Profiling build (for gprof)
prof:
	$(MPICC) $(PROF_FLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Clean build files
clean:
	rm -f $(TARGET) gmon.out *.mpiP pdist.dat timing.txt

# Run with default parameters
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 10000 1000

# Run with custom parameters (usage: make run-custom N=20000 T=500 P=4)
run-custom: $(TARGET)
	mpirun -np $(P) ./$(TARGET) $(N) $(T)

# Run with mpiP profiling
mpip: $(TARGET)
	LD_PRELOAD=$(MPIP_LIB) mpirun -np 4 ./$(TARGET) 10000 1000

# Run with mpiP and custom parameters
mpip-custom: $(TARGET)
	LD_PRELOAD=$(MPIP_LIB) mpirun -np $(P) ./$(TARGET) $(N) $(T)

# Run scaling test (P=1,2,4) with N=10000
scaling: $(TARGET)
	@echo "=== Scaling Test with N=10000, T=500 ==="
	@for p in 1 2 4; do \
		echo ""; \
		echo "--- Running with $$p processes ---"; \
		time mpirun -np $$p ./$(TARGET) 10000 500; \
	done

# Run large test (N=50000)
large: $(TARGET)
	@echo "=== Running with N=50000, T=500, P=4 ==="
	time mpirun -np 4 ./$(TARGET) 50000 500

# Run large test with mpiP
large-mpip: $(TARGET)
	@echo "=== Profiling N=50000, T=500, P=4 with mpiP ==="
	LD_PRELOAD=$(MPIP_LIB) mpirun -np 4 ./$(TARGET) 50000 500

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Build regular version"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make prof     - Build with gprof profiling"
	@echo "  make run      - Run with default settings"
	@echo "  make mpip     - Run with mpiP profiling"
	@echo "  make scaling  - Run scaling test (P=1,2,4)"
	@echo "  make large    - Run large test (N=50000)"
	@echo "  make clean    - Remove build files"
