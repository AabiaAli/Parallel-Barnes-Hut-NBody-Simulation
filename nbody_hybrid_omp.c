/*
 * N-Body Barnes-Hut Simulation — Optimization 2: Hybrid MPI + OpenMP
 *
 * Adds OpenMP parallelism inside the force-computation loop so that all
 * CPU cores on each MPI node contribute to the work.  Everything else
 * (tree build, velocity update, position update, MPI communication) is
 * unchanged from the original to keep the diff minimal and easy to audit.
 *
 * Compile:
 *   mpicc -O2 -fopenmp -o nbody_hybrid nbody_hybrid.c -lm
 *
 * Run:
 *   export OMP_NUM_THREADS=4
 *   mpirun -np 2 ./nbody_hybrid 10000 1000
 *
 * Threading model:
 *   - The outer loop over local particles is split across OMP threads.
 *   - Each thread accumulates into its own force[i], so there is NO race
 *     condition on the force array.
 *   - The Barnes-Hut tree is read-only during force computation, so tree
 *     traversal from multiple threads is safe without any locks.
 *   - Tree construction and deletion run single-threaded (their recursive
 *     pointer writes are not thread-safe and are not the bottleneck).
 *
 * Author: adapted from Dileban Karunamoorthy (dileban@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>    /* OpenMP header */

/* ── Constants ─────────────────────────────────────────────────── */
#define DEFAULT_N       10000
#define DEFAULT_TIME    1000
#define G               6.67300e-11
#define XBOUND          1.0e6
#define YBOUND          1.0e6
#define ZBOUND          1.0e6
#define RBOUND          10
#define DELTAT          0.01
#define THETA           1.0
#define MASS_OF_UNKNOWN 1.899e12

/* ── Structs ────────────────────────────────────────────────────── */
typedef struct { double px, py, pz; } Position;
typedef struct { double vx, vy, vz; } Velocity;
typedef struct { double fx, fy, fz; } Force;

typedef struct Cell {
    int    index;
    int    no_subcells;
    double mass;
    double x, y, z;
    double cx, cy, cz;
    double width, height, depth;
    struct Cell *subcells[8];
} Cell;

/* ── Globals ────────────────────────────────────────────────────── */
Position *position;
Velocity *ivelocity;
Velocity *velocity;
double   *mass;
double   *radius;
Force    *force;
Cell     *root_cell;

MPI_Datatype MPI_POSITION;
MPI_Datatype MPI_VELOCITY;

int N, TIME_STEPS;
int rank, size;
int part_size, pindex;

/* ── Utilities ──────────────────────────────────────────────────── */
double generate_rand()    { return rand() / ((double)RAND_MAX + 1); }
double generate_rand_ex() { return 2.0 * generate_rand() - 1.0; }

void initialize_space() {
    double ixbound = XBOUND - RBOUND;
    double iybound = YBOUND - RBOUND;
    double izbound = ZBOUND - RBOUND;
    for (int i = 0; i < N; i++) {
        mass[i]         = MASS_OF_UNKNOWN * generate_rand();
        radius[i]       = RBOUND * generate_rand();
        position[i].px  = generate_rand() * ixbound;
        position[i].py  = generate_rand() * iybound;
        position[i].pz  = generate_rand() * izbound;
        ivelocity[i].vx = generate_rand_ex();
        ivelocity[i].vy = generate_rand_ex();
        ivelocity[i].vz = generate_rand_ex();
    }
}

double compute_distance(Position a, Position b) {
    return sqrt(pow(a.px - b.px, 2.0) +
                pow(a.py - b.py, 2.0) +
                pow(a.pz - b.pz, 2.0));
}

/* ── Barnes-Hut Tree (single-threaded; read-only during force) ── */
Cell *BH_create_cell(double w, double h, double d) {
    Cell *c = (Cell *)malloc(sizeof(Cell));
    c->mass = 0; c->no_subcells = 0; c->index = -1;
    c->cx = c->cy = c->cz = 0;
    c->width = w; c->height = h; c->depth = d;
    return c;
}

void BH_set_location_of_subcells(Cell *cell, double w, double h, double d) {
    cell->subcells[0]->x = cell->x;     cell->subcells[0]->y = cell->y;     cell->subcells[0]->z = cell->z;
    cell->subcells[1]->x = cell->x + w; cell->subcells[1]->y = cell->y;     cell->subcells[1]->z = cell->z;
    cell->subcells[2]->x = cell->x + w; cell->subcells[2]->y = cell->y;     cell->subcells[2]->z = cell->z + d;
    cell->subcells[3]->x = cell->x;     cell->subcells[3]->y = cell->y;     cell->subcells[3]->z = cell->z + d;
    cell->subcells[4]->x = cell->x;     cell->subcells[4]->y = cell->y + h; cell->subcells[4]->z = cell->z;
    cell->subcells[5]->x = cell->x + w; cell->subcells[5]->y = cell->y + h; cell->subcells[5]->z = cell->z;
    cell->subcells[6]->x = cell->x + w; cell->subcells[6]->y = cell->y + h; cell->subcells[6]->z = cell->z + d;
    cell->subcells[7]->x = cell->x;     cell->subcells[7]->y = cell->y + h; cell->subcells[7]->z = cell->z + d;
}

void BH_generate_subcells(Cell *cell) {
    double w = cell->width / 2.0, h = cell->height / 2.0, d = cell->depth / 2.0;
    cell->no_subcells = 8;
    for (int i = 0; i < 8; i++) cell->subcells[i] = BH_create_cell(w, h, d);
    BH_set_location_of_subcells(cell, w, h, d);
}

int BH_locate_subcell(Cell *cell, int index) {
    int px = position[index].px > cell->subcells[6]->x;
    int py = position[index].py > cell->subcells[6]->y;
    int pz = position[index].pz > cell->subcells[6]->z;
    if  (px &&  py &&  pz) return 6;
    if  (px &&  py && !pz) return 5;
    if  (px && !py &&  pz) return 2;
    if  (px && !py && !pz) return 1;
    if (!px &&  py &&  pz) return 7;
    if (!px &&  py && !pz) return 4;
    if (!px && !py &&  pz) return 3;
    return 0;
}

void BH_add_to_cell(Cell *cell, int index) {
    if (cell->index == -1) { cell->index = index; return; }
    BH_generate_subcells(cell);
    int sc1 = BH_locate_subcell(cell, cell->index);
    cell->subcells[sc1]->index = cell->index;
    int sc2 = BH_locate_subcell(cell, index);
    if (sc1 == sc2) BH_add_to_cell(cell->subcells[sc1], index);
    else            cell->subcells[sc2]->index = index;
}

void BH_generate_octtree() {
    root_cell = BH_create_cell(XBOUND, YBOUND, ZBOUND);
    root_cell->index = 0; root_cell->x = root_cell->y = root_cell->z = 0;
    for (int i = 1; i < N; i++) {
        Cell *cell = root_cell;
        while (cell->no_subcells != 0) cell = cell->subcells[BH_locate_subcell(cell, i)];
        BH_add_to_cell(cell, i);
    }
}

Cell *BH_compute_cell_properties(Cell *cell) {
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

void BH_delete_octtree(Cell *cell) {
    if (cell->no_subcells != 0)
        for (int i = 0; i < cell->no_subcells; i++) BH_delete_octtree(cell->subcells[i]);
    free(cell);
}

/* ── Force computation — OMP parallelised ───────────────────────
 *
 * Design notes
 * ────────────
 * 1. The outer loop `for i in [0, part_size)` is the dominant cost (O(N log N)
 *    total, O(log N) per particle), so we parallelise it with `#pragma omp for`.
 *
 * 2. force[i] is exclusively written by thread i's loop iteration, so no
 *    reduction or critical section is needed on the force array.
 *
 * 3. root_cell is a global pointer that is only READ inside this loop
 *    (BH_compute_force_from_octtree does no writes); it is therefore safe
 *    for all threads to traverse simultaneously.
 *
 * 4. `schedule(dynamic, 16)` distributes chunks of 16 particles at a time.
 *    This matters because the Barnes-Hut work per particle varies with its
 *    location: particles near dense clusters do more tree traversal.  Dynamic
 *    scheduling equalises thread utilisation better than static.
 */

/* Forward declaration */
void BH_compute_force_from_octtree(Cell *cell, int index);

void BH_compute_force_from_cell(Cell *cell, int index) {
    double d = compute_distance(position[index], position[cell->index]);
    if (d == 0.0) return;
    double f = G * mass[index] * mass[cell->index] / (d * d);
    force[index - pindex].fx += f * (position[cell->index].px - position[index].px) / d;
    force[index - pindex].fy += f * (position[cell->index].py - position[index].py) / d;
    force[index - pindex].fz += f * (position[cell->index].pz - position[index].pz) / d;
}

void BH_compute_force_from_octtree(Cell *cell, int index) {
    if (cell->no_subcells == 0) {
        if (cell->index != -1 && cell->index != index)
            BH_compute_force_from_cell(cell, index);
        return;
    }
    double d = compute_distance(position[index], position[cell->index]);
    if (d == 0.0) return;
    if (THETA > cell->width / d) {
        BH_compute_force_from_cell(cell, index);
    } else {
        for (int i = 0; i < cell->no_subcells; i++)
            BH_compute_force_from_octtree(cell->subcells[i], index);
    }
}

void BH_compute_force() {
    /*
     * Parallel region: spawn threads once per call.
     * Each thread processes its own range of i with a private stack
     * for the recursive tree traversal.
     */
    #pragma omp parallel for schedule(dynamic, 16) default(shared)
    for (int i = 0; i < part_size; i++) {
        /* Initialise force for this particle (no race: unique i per thread) */
        force[i].fx = 0.0;
        force[i].fy = 0.0;
        force[i].fz = 0.0;

        /* Tree traversal: read-only on root_cell, writes only to force[i] */
        BH_compute_force_from_octtree(root_cell, i + pindex);
    }
}

/* ── Velocity and position update (serial; cheap) ─────────────── */
void compute_velocity() {
    for (int i = 0; i < part_size; i++) {
        velocity[i].vx += (force[i].fx / mass[i + pindex]) * DELTAT;
        velocity[i].vy += (force[i].fy / mass[i + pindex]) * DELTAT;
        velocity[i].vz += (force[i].fz / mass[i + pindex]) * DELTAT;
    }
}

void compute_positions() {
    for (int i = 0; i < part_size; i++) {
        int gi = i + pindex;
        position[gi].px += velocity[i].vx * DELTAT;
        position[gi].py += velocity[i].vy * DELTAT;
        position[gi].pz += velocity[i].vz * DELTAT;

        if (position[gi].px + radius[gi] >= XBOUND || position[gi].px - radius[gi] <= 0) velocity[i].vx *= -1;
        if (position[gi].py + radius[gi] >= YBOUND || position[gi].py - radius[gi] <= 0) velocity[i].vy *= -1;
        if (position[gi].pz + radius[gi] >= ZBOUND || position[gi].pz - radius[gi] <= 0) velocity[i].vz *= -1;
    }
}

void init_velocity() {
    for (int i = 0; i < part_size; i++) {
        velocity[i].vx = velocity[i].vy = velocity[i].vz = 0.0;
    }
}

/* ── Simulation loop ────────────────────────────────────────────── */
void run_simulation() {
    int nthreads;
    #pragma omp parallel
    #pragma omp single
    nthreads = omp_get_num_threads();

    if (rank == 0)
        printf("\n[Hybrid MPI+OMP] %d bodies, %d steps, %d MPI ranks x %d OMP threads\n\n",
               N, TIME_STEPS, size, nthreads);

    MPI_Bcast(mass,      N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position,  N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Scatter(ivelocity, part_size, MPI_VELOCITY,
                velocity,  part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    for (int step = 0; step < TIME_STEPS; step++) {
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);
        BH_compute_force();          /* <── parallel OMP loop here */
        BH_delete_octtree(root_cell);

        compute_velocity();
        compute_positions();

        MPI_Allgather(position + pindex, part_size, MPI_POSITION,
                      position,          part_size, MPI_POSITION,
                      MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *f = fopen("pdist_hybrid.dat", "w");
        if (f) {
            for (int i = 0; i < N; i++)
                fprintf(f, "px=%f, py=%f, pz=%f\n",
                        position[i].px, position[i].py, position[i].pz);
            fclose(f);
            printf("[Hybrid] Positions written to pdist_hybrid.dat\n");
        }
    }
}

/* ── Main ───────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    /*
     * MPI_Init_thread requests MPI_THREAD_FUNNELED: only the main thread
     * makes MPI calls, which matches our design (all MPI is outside the
     * OMP parallel region).
     */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED && rank == 0)
        fprintf(stderr, "Warning: MPI does not fully support MPI_THREAD_FUNNELED\n");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N          = (argc >= 2) ? atoi(argv[1]) : DEFAULT_N;
    TIME_STEPS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TIME;

    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_POSITION); MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VELOCITY); MPI_Type_commit(&MPI_VELOCITY);

    part_size = N / size;
    pindex    = rank * part_size;

    mass      = (double   *)malloc(N         * sizeof(double));
    radius    = (double   *)malloc(N         * sizeof(double));
    position  = (Position *)malloc(N         * sizeof(Position));
    ivelocity = (Velocity *)malloc(N         * sizeof(Velocity));
    velocity  = (Velocity *)malloc(part_size * sizeof(Velocity));
    force     = (Force    *)malloc(part_size * sizeof(Force));

    init_velocity();
    if (rank == 0) initialize_space();

    run_simulation();

    MPI_Finalize();
    return 0;
}
