/*
 * nbody_combined.c  —  All Four Optimizations Combined
 *
 * Optimizations integrated:
 *
 *   1. ORB Spatial Load Balancing  (from nbody_orb.c)
 *      Rank 0 runs ORB_partition_once() before the simulation loop. It
 *      recursively bisects particles along the longest spatial axis so each
 *      rank owns a compact spatial block with equal particle count. Global
 *      arrays are physically reordered in-place so the existing pindex/
 *      part_size indexing requires zero changes to the simulation loop.
 *      Cost: O(N log N) once. Benefit: minimal barrier idle time across
 *      all steps because each rank always starts with spatially local work.
 *
 *   2. Adaptive Theta              (from nbody_adaptive_theta.c)
 *      Each rank computes local_theta = THETA_BASE / sqrt(local_den / avg_den)
 *      using an O(part_size) bounding-box pass + one MPI_Allreduce.
 *      Dense ranks get a smaller theta (more accurate), sparse ranks get a
 *      larger theta (fewer tree traversals, faster). Clamped to
 *      [THETA_MIN, THETA_MAX]. Recomputed every RECOMPUTE_INTERVAL steps so
 *      overhead is negligible. After ORB, each rank's particles are already
 *      spatially cohesive, making the density estimate more meaningful.
 *
 *   3. Arena Allocator             (from nbody_parallel_tree.c)
 *      Each MPI process allocates ONE Cell array of ARENA_FACTOR * N cells
 *      before the loop. BH_generate_subcells() bumps an integer counter
 *      (arena_used++) instead of calling malloc(), eliminating the global
 *      allocator lock entirely. BH_delete_octtree() is removed; the tree is
 *      reset with arena_used = 0 at the start of each step — O(1) vs
 *      O(N log N) free() calls. No pointers are freed mid-step so the
 *      OpenMP force traversal can safely read the tree without any memory
 *      management races.
 *
 *   4. Hybrid MPI + OpenMP         (from nbody_hybrid_omp.c / nbody_parallel_tree.c)
 *      BH_compute_force() is parallelized with:
 *        #pragma omp parallel for schedule(dynamic,32)
 *      The tree is built single-threaded (safe for arena_create_cell) then
 *      treated as read-only during force traversal — no writes, no races.
 *      Each loop iteration i writes only force[i], owned by exactly one
 *      thread. Dynamic scheduling handles the variable traversal depth that
 *      arises from non-uniform particle densities. MPI_Init_thread is called
 *      with MPI_THREAD_FUNNELED so all MPI calls remain on the main thread.
 *
 * Integration notes:
 *   - force_from_tree() uses local_theta (global, always shared in OpenMP)
 *     so adaptive theta works transparently inside the parallel region.
 *   - ORB runs before the arena is allocated; they touch disjoint data.
 *   - The simulation loop body is identical in structure to the baseline:
 *     tree build → cell properties → force → velocity → positions → allgather.
 *     The only structural removal is BH_delete_octtree (replaced by arena reset).
 *
 * Compile:  mpicc -O2 -fopenmp -o nbody_combined nbody_combined.c -lm
 * Run:      mkdir -p results
 *           export OMP_NUM_THREADS=4
 *           mpirun -np 4 ./nbody_combined 10000 500
 *
 * Recommended core assignment: np * OMP_NUM_THREADS = total physical cores.
 * Example 16-core node: mpirun -np 4 with OMP_NUM_THREADS=4.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/* ── Simulation parameters ──────────────────────────────────────── */
#define DEFAULT_N            10000
#define DEFAULT_TIME         500
#define G                    6.67300e-11
#define XBOUND               1.0e6
#define YBOUND               1.0e6
#define ZBOUND               1.0e6
#define RBOUND               10
#define DELTAT               0.01
#define MASS_OF_UNKNOWN      1.899e12

/* ── Adaptive theta parameters ──────────────────────────────────── */
#define THETA_BASE           1.0   /* opening angle at average density  */
#define THETA_MIN            0.4   /* floor: never less accurate than this */
#define THETA_MAX            1.4   /* ceiling: never more approximate   */
#define RECOMPUTE_INTERVAL   20    /* steps between theta recomputes    */

/* ── Arena sizing ───────────────────────────────────────────────── */
/* Empirically: N=10000 uses ~3.3*N cells, N=50000 uses ~3.26*N.
   Factor 4 gives ~21% headroom. Increase if you see arena overflow. */
#define ARENA_FACTOR         4

/* ── Types ──────────────────────────────────────────────────────── */
typedef struct { double px, py, pz; } Position;
typedef struct { double vx, vy, vz; } Velocity;
typedef struct { double fx, fy, fz; } Force;

typedef struct Cell {
    int    index, no_subcells;
    double mass, x, y, z, cx, cy, cz, width, height, depth;
    struct Cell *subcells[8];
} Cell;

/* ── Global simulation state ────────────────────────────────────── */
Position *position;
Velocity *ivelocity, *velocity;
double   *mass, *radius;
Force    *force;
Cell     *root_cell;

/* Arena — one per MPI process, allocated once before the loop */
Cell *arena;
int   arena_used;
int   arena_capacity;

MPI_Datatype MPI_POSITION, MPI_VELOCITY;
int    N, TIME_STEPS, rank, size, part_size, pindex;
double local_theta;          /* per-rank adaptive opening angle */
int    name_length;
char   name[MPI_MAX_PROCESSOR_NAME];

/* ═══════════════════════════════════════════════════════════════
 * SECTION 1: Utility functions
 * ═══════════════════════════════════════════════════════════════ */

double generate_rand()    { return rand() / ((double)RAND_MAX + 1.0); }
double generate_rand_ex() { return 2.0 * generate_rand() - 1.0; }

void initialize_space() {
    double ix = XBOUND - RBOUND, iy = YBOUND - RBOUND, iz = ZBOUND - RBOUND;
    for (int i = 0; i < N; i++) {
        mass[i]         = MASS_OF_UNKNOWN * generate_rand();
        radius[i]       = RBOUND * generate_rand();
        position[i].px  = generate_rand() * ix;
        position[i].py  = generate_rand() * iy;
        position[i].pz  = generate_rand() * iz;
        ivelocity[i].vx = generate_rand_ex();
        ivelocity[i].vy = generate_rand_ex();
        ivelocity[i].vz = generate_rand_ex();
    }
}

static inline double compute_distance(Position a, Position b) {
    double dx = a.px - b.px, dy = a.py - b.py, dz = a.pz - b.pz;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 2: ORB one-time spatial partition (Optimization 1)
 *
 * Recursively bisects the particle set along its longest spatial
 * axis, assigning equal-count spatial blocks to each rank, then
 * physically reorders all global arrays to match.
 * ═══════════════════════════════════════════════════════════════ */

static Position *_orb_pos; /* scratch pointer for qsort comparator */
static int       _orb_axis;

static int cmp_orb(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    double da, db;
    if      (_orb_axis == 0) { da = _orb_pos[ia].px; db = _orb_pos[ib].px; }
    else if (_orb_axis == 1) { da = _orb_pos[ia].py; db = _orb_pos[ib].py; }
    else                     { da = _orb_pos[ia].pz; db = _orb_pos[ib].pz; }
    return (da > db) - (da < db);
}

/* Recursively assign particles to ranks using longest-axis bisection */
static void bisect(int *idx, int n, int r0, int nr, int *asgn) {
    if (nr == 1) {
        for (int i = 0; i < n; i++) asgn[idx[i]] = r0;
        return;
    }
    double xmn=1e30,xmx=-1e30,ymn=1e30,ymx=-1e30,zmn=1e30,zmx=-1e30;
    for (int i = 0; i < n; i++) {
        int id = idx[i];
        if (_orb_pos[id].px < xmn) xmn = _orb_pos[id].px;
        if (_orb_pos[id].px > xmx) xmx = _orb_pos[id].px;
        if (_orb_pos[id].py < ymn) ymn = _orb_pos[id].py;
        if (_orb_pos[id].py > ymx) ymx = _orb_pos[id].py;
        if (_orb_pos[id].pz < zmn) zmn = _orb_pos[id].pz;
        if (_orb_pos[id].pz > zmx) zmx = _orb_pos[id].pz;
    }
    double dx = xmx - xmn, dy = ymx - ymn, dz = zmx - zmn;
    _orb_axis = (dx >= dy && dx >= dz) ? 0 : (dy >= dz) ? 1 : 2;
    qsort(idx, n, sizeof(int), cmp_orb);
    int lr    = nr / 2, rr = nr - lr;
    int split = (int)((double)n * lr / nr);
    bisect(idx,        split,   r0,      lr, asgn);
    bisect(idx + split, n - split, r0 + lr, rr, asgn);
}

/* Called on rank 0 only, before any broadcast. Reorders position, mass,
   radius and ivelocity so rank r owns indices [r*part_size, (r+1)*part_size). */
static void ORB_partition_once() {
    int *idx  = malloc(N * sizeof(int));
    int *asgn = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    _orb_pos  = position;
    _orb_axis = 0;
    bisect(idx, N, 0, size, asgn);
    free(idx);

    /* Build per-rank particle lists */
    int  *cnt  = calloc(size, sizeof(int));
    int **list = malloc(size * sizeof(int *));
    for (int r = 0; r < size; r++) list[r] = malloc(part_size * sizeof(int));
    for (int i = 0; i < N; i++) {
        int r = asgn[i];
        if (cnt[r] < part_size) list[r][cnt[r]++] = i;
    }

    /* Flatten into a single reordering: rank0 block, rank1 block, ... */
    int *order = malloc(N * sizeof(int));
    int cur = 0;
    for (int r = 0; r < size; r++)
        for (int j = 0; j < part_size; j++)
            order[cur++] = list[r][j];

    /* Apply reordering to all global arrays */
    Position *tp = malloc(N * sizeof(Position));
    double   *tm = malloc(N * sizeof(double));
    double   *tr = malloc(N * sizeof(double));
    Velocity *tv = malloc(N * sizeof(Velocity));
    for (int i = 0; i < N; i++) {
        tp[i] = position[order[i]];
        tm[i] = mass[order[i]];
        tr[i] = radius[order[i]];
        tv[i] = ivelocity[order[i]];
    }
    memcpy(position,  tp, N * sizeof(Position));
    memcpy(mass,      tm, N * sizeof(double));
    memcpy(radius,    tr, N * sizeof(double));
    memcpy(ivelocity, tv, N * sizeof(Velocity));
    free(tp); free(tm); free(tr); free(tv);
    free(order); free(asgn); free(cnt);
    for (int r = 0; r < size; r++) free(list[r]);
    free(list);
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 3: Adaptive theta (Optimization 2)
 *
 * O(part_size) bounding-box pass + one MPI_Allreduce.
 * Sets local_theta on each rank based on its density relative to
 * the global average. After ORB, particles per rank are spatially
 * contiguous so this density estimate is accurate and meaningful.
 * ═══════════════════════════════════════════════════════════════ */

static void compute_adaptive_theta() {
    double xmn=1e30,xmx=-1e30,ymn=1e30,ymx=-1e30,zmn=1e30,zmx=-1e30;
    for (int i = 0; i < part_size; i++) {
        int id = i + pindex;
        if (position[id].px < xmn) xmn = position[id].px;
        if (position[id].px > xmx) xmx = position[id].px;
        if (position[id].py < ymn) ymn = position[id].py;
        if (position[id].py > ymx) ymx = position[id].py;
        if (position[id].pz < zmn) zmn = position[id].pz;
        if (position[id].pz > zmx) zmx = position[id].pz;
    }
    /* +1.0 avoids zero-volume when all particles are colinear */
    double vol      = (xmx - xmn + 1.0) * (ymx - ymn + 1.0) * (zmx - zmn + 1.0);
    double local_den = (double)part_size / vol;
    double sum = 0.0;
    MPI_Allreduce(&local_den, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double avg = sum / size;
    if (avg < 1e-30) { local_theta = THETA_BASE; return; }
    local_theta = THETA_BASE / sqrt(local_den / avg);
    if (local_theta < THETA_MIN) local_theta = THETA_MIN;
    if (local_theta > THETA_MAX) local_theta = THETA_MAX;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 4: Arena allocator (Optimization 3)
 *
 * arena_create_cell() replaces malloc(sizeof(Cell)) everywhere
 * in the tree-build path. It bumps an integer counter in O(1)
 * with no locking. The entire tree is "freed" by setting
 * arena_used = 0 at the start of BH_generate_octtree().
 * ═══════════════════════════════════════════════════════════════ */

static inline Cell *arena_create_cell(double w, double h, double d) {
    if (arena_used >= arena_capacity) {
        fprintf(stderr, "Rank %d: arena overflow (used=%d cap=%d N=%d). "
                        "Increase ARENA_FACTOR.\n",
                rank, arena_used, arena_capacity, N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Cell *c = &arena[arena_used++];
    c->mass = 0.0;  c->no_subcells = 0;  c->index = -1;
    c->cx   = 0.0;  c->cy = 0.0;         c->cz    = 0.0;
    c->width = w;   c->height = h;        c->depth  = d;
    return c;
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 5: Barnes-Hut octree (arena-based, no malloc/free)
 * ═══════════════════════════════════════════════════════════════ */

static void BH_set_location_of_subcells(Cell *cell, double w, double h, double d) {
    cell->subcells[0]->x = cell->x;     cell->subcells[0]->y = cell->y;     cell->subcells[0]->z = cell->z;
    cell->subcells[1]->x = cell->x + w; cell->subcells[1]->y = cell->y;     cell->subcells[1]->z = cell->z;
    cell->subcells[2]->x = cell->x + w; cell->subcells[2]->y = cell->y;     cell->subcells[2]->z = cell->z + d;
    cell->subcells[3]->x = cell->x;     cell->subcells[3]->y = cell->y;     cell->subcells[3]->z = cell->z + d;
    cell->subcells[4]->x = cell->x;     cell->subcells[4]->y = cell->y + h; cell->subcells[4]->z = cell->z;
    cell->subcells[5]->x = cell->x + w; cell->subcells[5]->y = cell->y + h; cell->subcells[5]->z = cell->z;
    cell->subcells[6]->x = cell->x + w; cell->subcells[6]->y = cell->y + h; cell->subcells[6]->z = cell->z + d;
    cell->subcells[7]->x = cell->x;     cell->subcells[7]->y = cell->y + h; cell->subcells[7]->z = cell->z + d;
}

static void BH_generate_subcells(Cell *cell) {
    double w = cell->width / 2, h = cell->height / 2, d = cell->depth / 2;
    cell->no_subcells = 8;
    for (int i = 0; i < 8; i++)
        cell->subcells[i] = arena_create_cell(w, h, d); /* O(1), no lock */
    BH_set_location_of_subcells(cell, w, h, d);
}

static int BH_locate_subcell(Cell *cell, int idx) {
    if (position[idx].px > cell->subcells[6]->x) {
        if (position[idx].py > cell->subcells[6]->y)
            return (position[idx].pz > cell->subcells[6]->z) ? 6 : 5;
        else
            return (position[idx].pz > cell->subcells[6]->z) ? 2 : 1;
    } else {
        if (position[idx].py > cell->subcells[6]->y)
            return (position[idx].pz > cell->subcells[6]->z) ? 7 : 4;
        else
            return (position[idx].pz > cell->subcells[6]->z) ? 3 : 0;
    }
}

static void BH_add_to_cell(Cell *cell, int idx) {
    if (cell->index == -1) { cell->index = idx; return; }
    BH_generate_subcells(cell);
    int sc1 = BH_locate_subcell(cell, cell->index);
    cell->subcells[sc1]->index = cell->index;
    int sc2 = BH_locate_subcell(cell, idx);
    if (sc1 == sc2) BH_add_to_cell(cell->subcells[sc1], idx);
    else            cell->subcells[sc2]->index = idx;
}

static void BH_generate_octtree() {
    /* O(1) arena reset replaces the entire BH_delete_octtree pass */
    arena_used = 0;
    root_cell  = arena_create_cell(XBOUND, YBOUND, ZBOUND);
    root_cell->index = 0;
    root_cell->x = root_cell->y = root_cell->z = 0.0;
    for (int i = 1; i < N; i++) {
        Cell *cell = root_cell;
        while (cell->no_subcells != 0)
            cell = cell->subcells[BH_locate_subcell(cell, i)];
        BH_add_to_cell(cell, i);
    }
}

static Cell *BH_compute_cell_properties(Cell *cell) {
    if (cell->no_subcells == 0) {
        if (cell->index != -1) { cell->mass = mass[cell->index]; return cell; }
        return NULL;
    }
    double tx = 0, ty = 0, tz = 0;
    for (int i = 0; i < cell->no_subcells; i++) {
        Cell *t = BH_compute_cell_properties(cell->subcells[i]);
        if (t) {
            cell->mass += t->mass;
            tx += position[t->index].px * t->mass;
            ty += position[t->index].py * t->mass;
            tz += position[t->index].pz * t->mass;
        }
    }
    if (cell->mass > 0) {
        cell->cx = tx / cell->mass;
        cell->cy = ty / cell->mass;
        cell->cz = tz / cell->mass;
    }
    return cell;
}

/* No BH_delete_octtree needed — arena_used = 0 handles cleanup in O(1) */

/* ═══════════════════════════════════════════════════════════════
 * SECTION 6: Force computation — read-only tree + OpenMP (Optimization 4)
 *            Uses local_theta per rank (Optimization 2)
 *
 * Tree is fully built (and read-only) before this function is called.
 * force[i] is written by exactly one thread for each i — no race.
 * local_theta is a file-scope global, always shared across threads.
 * Dynamic scheduling handles the variable traversal depth caused by
 * non-uniform particle density (especially effective after ORB).
 * ═══════════════════════════════════════════════════════════════ */

static void force_from_cell(Cell *cell, int idx) {
    double d = compute_distance(position[idx], position[cell->index]);
    if (d == 0.0) return;
    double f = G * mass[idx] * mass[cell->index] / (d * d);
    force[idx - pindex].fx += f * (position[cell->index].px - position[idx].px) / d;
    force[idx - pindex].fy += f * (position[cell->index].py - position[idx].py) / d;
    force[idx - pindex].fz += f * (position[cell->index].pz - position[idx].pz) / d;
}

static void force_from_tree(Cell *cell, int idx) {
    if (cell->no_subcells == 0) {
        if (cell->index != -1 && cell->index != idx)
            force_from_cell(cell, idx);
        return;
    }
    double d = compute_distance(position[idx], position[cell->index]);
    if (d == 0.0) return;
    /* Use local_theta (adaptive, per-rank) instead of a fixed THETA */
    if (local_theta > (cell->width / d))
        force_from_cell(cell, idx);
    else
        for (int i = 0; i < cell->no_subcells; i++)
            force_from_tree(cell->subcells[i], idx);
}

static void BH_compute_force() {
    /*
     * Tree was built single-threaded above → arena_create_cell is safe.
     * Tree is now read-only → all threads can traverse concurrently.
     * force[i] has a unique owner per i → no false sharing, no locks.
     * local_theta is a file-scope global → shared by default in OpenMP.
     * dynamic,32 keeps threads busy despite varying traversal depth.
     */
    #pragma omp parallel for schedule(dynamic, 32) \
        default(none) \
        shared(force, root_cell, position, mass, pindex, part_size, local_theta)
    for (int i = 0; i < part_size; i++) {
        force[i].fx = force[i].fy = force[i].fz = 0.0;
        force_from_tree(root_cell, i + pindex);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 7: Physics update (unchanged from baseline)
 * ═══════════════════════════════════════════════════════════════ */

static void compute_velocity() {
    for (int i = 0; i < part_size; i++) {
        velocity[i].vx += (force[i].fx / mass[i + pindex]) * DELTAT;
        velocity[i].vy += (force[i].fy / mass[i + pindex]) * DELTAT;
        velocity[i].vz += (force[i].fz / mass[i + pindex]) * DELTAT;
    }
}

static void compute_positions() {
    for (int i = 0; i < part_size; i++) {
        position[i + pindex].px += velocity[i].vx * DELTAT;
        position[i + pindex].py += velocity[i].vy * DELTAT;
        position[i + pindex].pz += velocity[i].vz * DELTAT;
        if ((position[i+pindex].px + radius[i+pindex]) >= XBOUND ||
            (position[i+pindex].px - radius[i+pindex]) <= 0.0)
            velocity[i].vx *= -1.0;
        else if ((position[i+pindex].py + radius[i+pindex]) >= YBOUND ||
                 (position[i+pindex].py - radius[i+pindex]) <= 0.0)
            velocity[i].vy *= -1.0;
        else if ((position[i+pindex].pz + radius[i+pindex]) >= ZBOUND ||
                 (position[i+pindex].pz - radius[i+pindex]) <= 0.0)
            velocity[i].vz *= -1.0;
    }
}

static void init_velocity() {
    for (int i = 0; i < part_size; i++)
        velocity[i].vx = velocity[i].vy = velocity[i].vz = 0.0;
}

static void write_positions(const char *f) {
    if (rank != 0) return;
    FILE *fp = fopen(f, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", f); return; }
    for (int i = 0; i < N; i++)
        fprintf(fp, "%.6f %.6f %.6f\n",
                position[i].px, position[i].py, position[i].pz);
    fclose(fp);
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 8: Simulation driver
 * ═══════════════════════════════════════════════════════════════ */

static void run_simulation() {
    int nthreads = omp_get_max_threads();
    if (rank == 0)
        printf("[COMBINED] N=%d  steps=%d  procs=%d  threads/proc=%d  "
               "theta_base=%.2f  recompute_interval=%d  arena_factor=%d\n",
               N, TIME_STEPS, size, nthreads,
               THETA_BASE, RECOMPUTE_INTERVAL, ARENA_FACTOR);

    /* ── Optimization 1: ORB partition on rank 0, then broadcast ── */
    if (rank == 0) {
        printf("[COMBINED] Running ORB partition...\n");
        ORB_partition_once();
        printf("[COMBINED] ORB partition complete.\n");
    }
    MPI_Bcast(mass,      N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position,  N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Bcast(radius,    N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Scatter(ivelocity, part_size, MPI_VELOCITY,
                velocity,  part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    /* ── Optimization 3: Allocate arena once ────────────────────── */
    arena_capacity = ARENA_FACTOR * N;
    arena = malloc(arena_capacity * sizeof(Cell));
    if (!arena) {
        fprintf(stderr, "Rank %d: failed to allocate arena (%d cells)\n",
                rank, arena_capacity);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    arena_used = 0;

    /* ── Optimization 2: Initial adaptive theta ─────────────────── */
    local_theta = THETA_BASE;
    compute_adaptive_theta();
    if (rank == 0)
        printf("[COMBINED] Initial local_theta on rank 0: %.4f\n", local_theta);

    /* ── Timing accumulators ─────────────────────────────────────── */
    double t_tree = 0, t_force = 0, t_comm = 0, t_theta = 0;
    double t_start = MPI_Wtime();

    /* ── Main simulation loop ────────────────────────────────────── */
    for (int step = 0; step < TIME_STEPS; step++) {

        /* Opt 2: recompute adaptive theta periodically */
        if (step > 0 && step % RECOMPUTE_INTERVAL == 0) {
            double tt = MPI_Wtime();
            compute_adaptive_theta();
            t_theta += MPI_Wtime() - tt;
        }

        /* Opt 3: tree build using arena (no malloc/free overhead) */
        double t0 = MPI_Wtime();
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);
        double t1 = MPI_Wtime();

        /* Opt 4: OpenMP parallel force with Opt 2 local_theta */
        BH_compute_force();
        double t2 = MPI_Wtime();

        /* No BH_delete_octtree — arena reset at next BH_generate_octtree */
        compute_velocity();
        compute_positions();
        double t3 = MPI_Wtime();

        MPI_Allgather(position + pindex, part_size, MPI_POSITION,
                      position,          part_size, MPI_POSITION,
                      MPI_COMM_WORLD);
        double t4 = MPI_Wtime();

        t_tree  += t1 - t0;
        t_force += t2 - t1;
        t_comm  += t4 - t3;
    }

    double wall = MPI_Wtime() - t_start;
    if (rank == 0) {
        printf("[COMBINED] Wall=%.4fs  Tree=%.4fs  Force=%.4fs  "
               "Comm=%.4fs  Theta=%.4fs\n",
               wall, t_tree, t_force, t_comm, t_theta);
        printf("[COMBINED] Arena peak usage: %d / %d cells (%.1f%%)\n",
               arena_used, arena_capacity,
               100.0 * arena_used / arena_capacity);
        write_positions("results/combined_positions.dat");
    }

    free(arena);
}

/* ═══════════════════════════════════════════════════════════════
 * SECTION 9: Entry point
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    /* MPI_THREAD_FUNNELED: all MPI calls are on the main thread;
       OpenMP threads only do force traversal (no MPI inside). */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Warning: MPI does not support MPI_THREAD_FUNNELED "
                        "(got %d). Continuing anyway.\n", provided);
    }

    N          = (argc >= 2) ? atoi(argv[1]) : DEFAULT_N;
    TIME_STEPS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TIME;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_POSITION); MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VELOCITY); MPI_Type_commit(&MPI_VELOCITY);
    MPI_Get_processor_name(name, &name_length);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by number of "
                            "processes (%d)\n", N, size);
        MPI_Finalize();
        return 1;
    }

    part_size = N / size;
    pindex    = rank * part_size;

    /* Allocate global arrays */
    mass      = malloc(N * sizeof(double));
    radius    = malloc(N * sizeof(double));
    position  = malloc(N * sizeof(Position));
    ivelocity = malloc(N * sizeof(Velocity));
    velocity  = malloc(part_size * sizeof(Velocity));
    force     = malloc(part_size * sizeof(Force));

    init_velocity();
    if (rank == 0) { srand(42); initialize_space(); }

    double t0 = MPI_Wtime();
    run_simulation();
    if (rank == 0)
        printf("[COMBINED] Total wall time: %.4f s\n", MPI_Wtime() - t0);

    free(mass); free(radius); free(position);
    free(ivelocity); free(velocity); free(force);
    MPI_Type_free(&MPI_POSITION);
    MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize();
    return 0;
}
