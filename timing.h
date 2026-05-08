/*
 * timing.h — Drop-in timing instrumentation for all N-Body variants
 *
 * HOW TO USE IN YOUR C FILES:
 * 1. Add:  #include "timing.h"
 * 2. Wrap your force computation block like this:
 *
 *      FORCE_TIME_START();
 *      BH_compute_force();          // or compute_force() for O(N^2)
 *      FORCE_TIME_END(rank);
 *
 * 3. At the start of run_simulation(), call:
 *      TOTAL_TIME_START();
 *
 * 4. At the end of run_simulation() (after the loop), call:
 *      TOTAL_TIME_END(rank);
 *
 * The Python benchmark script parses the printed lines automatically.
 * Format it expects:
 *   [ForceTime] Rank 0: 0.123456 s
 *   [Time] Total wall time: 12.345 s
 */

#ifndef TIMING_H
#define TIMING_H

#include <mpi.h>

/* Internal static variables — one per translation unit, that's fine */
static double _t_force_start = 0.0;
static double _t_total_start = 0.0;
static double _t_force_accum = 0.0;   /* accumulated force time across all steps */

/*
 * FORCE_TIME_START / FORCE_TIME_END
 * Wrap around BH_compute_force() inside the time loop.
 * Accumulates across all TIME_STEPS, then prints at end of simulation.
 */
#define FORCE_TIME_START() \
    do { _t_force_start = MPI_Wtime(); } while(0)

#define FORCE_TIME_END(my_rank) \
    do { _t_force_accum += MPI_Wtime() - _t_force_start; } while(0)

/*
 * Print the accumulated force time at the end of the simulation.
 * Call once per rank after the time loop.
 * Python script looks for this exact format.
 */
#define FORCE_TIME_PRINT(my_rank) \
    do { \
        printf("[ForceTime] Rank %d: %.6f s\n", (my_rank), _t_force_accum); \
        fflush(stdout); \
    } while(0)

/*
 * Total wall time — call before MPI_Bcast, print after write_positions().
 */
#define TOTAL_TIME_START() \
    do { _t_total_start = MPI_Wtime(); } while(0)

#define TOTAL_TIME_END(my_rank) \
    do { \
        double _elapsed = MPI_Wtime() - _t_total_start; \
        if ((my_rank) == 0) { \
            printf("[Time] Total wall time: %.6f s\n", _elapsed); \
            fflush(stdout); \
        } \
    } while(0)

#endif /* TIMING_H */
