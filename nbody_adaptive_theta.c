/*
 * N-Body Barnes-Hut Simulation — Optimization 3: Adaptive Theta Per Process
 *
 * The Barnes-Hut opening angle θ (THETA) controls the accuracy/speed
 * trade-off.  In the original code it is a single compile-time constant
 * applied uniformly to every particle on every rank.
 *
 * This version computes a per-rank θ each iteration based on the local
 * particle density and an inter-rank load imbalance signal:
 *
 *   1. LOCAL DENSITY SIGNAL
 *      Each rank measures the mean nearest-neighbour distance for its
 *      particles.  In a dense region the tree is deep and expensive;
 *      the rank is allowed to raise θ slightly to reduce that cost.
 *
 *   2. LOAD IMBALANCE SIGNAL
 *      After every step, each rank times its force-compute phase.
 *      The slowest rank determines the effective step time (barrier
 *      effect).  Any rank whose compute time is well below the maximum
 *      can LOWER its θ (gaining accuracy for free), while the overloaded
 *      rank may raise θ to speed up.
 *
 *   3. BOUNDS
 *      θ is clamped to [THETA_MIN, THETA_MAX] so accuracy never degrades
 *      catastrophically and approximation is always applied.
 *
 * The adaptation runs after MPI_Allgather so every rank has full
 * position knowledge for the next iteration.
 *
 * Compile:
 *   mpicc -O2 -o nbody_adaptive nbody_adaptive_theta.c -lm
 *
 * Run:
 *   mpirun -np 4 ./nbody_adaptive 10000 1000
 *
 * Author: adapted from Dileban Karunamoorthy (dileban@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ── Constants ─────────────────────────────────────────────────── */
#define DEFAULT_N       10000
#define DEFAULT_TIME    1000
#define G               6.67300e-11
#define XBOUND          1.0e6
#define YBOUND          1.0e6
#define ZBOUND          1.0e6
#define RBOUND          10
#define DELTAT          0.01
#define MASS_OF_UNKNOWN 1.899e12

/* Theta bounds and adaptation rate */
#define THETA_DEFAULT   1.0    /* starting opening angle */
#define THETA_MIN       0.3    /* most accurate allowed                      */
#define THETA_MAX       1.5    /* least accurate allowed                     */
#define THETA_STEP      0.05   /* maximum change per iteration               */
#define IMBALANCE_TOL   0.10   /* 10 % imbalance triggers adaptation         */

/* Number of nearest neighbours sampled per rank for density estimate */
#define DENSITY_SAMPLE  50

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

double local_theta;   /* per-rank current opening angle */

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

/* ── Theta adaptation ───────────────────────────────────────────
 *
 * adapt_theta() is called once per iteration after positions are
 * gathered.  It updates local_theta for the NEXT iteration using two
 * signals:
 *
 *   density_score  ∈ [0,1] — 0 = very dense (expensive), 1 = sparse
 *   balance_score  ∈ [0,1] — 0 = this rank is slowest, 1 = fastest
 *
 * Combined score > 0.5 → lower θ (be more accurate, we have headroom).
 * Combined score < 0.5 → raise θ (be faster, we are the bottleneck).
 */
void adapt_theta(double my_force_time) {
    /* ── Step 1: gather all force times ── */
    double all_times[size];
    MPI_Allgather(&my_force_time, 1, MPI_DOUBLE,
                  all_times, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    double t_max = 0.0, t_sum = 0.0;
    for (int r = 0; r < size; r++) {
        if (all_times[r] > t_max) t_max = all_times[r];
        t_sum += all_times[r];
    }
    double t_mean = (size > 0) ? t_sum / size : my_force_time;

    /* balance_score: 1 = we are far below average (headroom to lower θ) */
    double balance_score = 0.5;
    if (t_max > 1e-9) {
        /* How much faster than the slowest rank are we?
         * If we equal the max, score = 0 (we are the bottleneck).
         * If we are half the max, score = 0.5.                      */
        balance_score = 1.0 - (my_force_time / t_max);
        /* Only trigger if imbalance is significant */
        if (t_max - t_mean < IMBALANCE_TOL * t_mean)
            balance_score = 0.5; /* balanced enough — no change */
    }

    /* ── Step 2: local density estimate via sampled nearest-neighbour ── */
    int sample_count = (part_size < DENSITY_SAMPLE) ? part_size : DENSITY_SAMPLE;
    double nn_sum = 0.0;
    for (int s = 0; s < sample_count; s++) {
        int i = pindex + (int)(generate_rand() * part_size);
        double nn_dist = 1e300;
        /* Compare against a small random set for speed */
        for (int k = 0; k < 100 && k < N; k++) {
            int j = (int)(generate_rand() * N);
            if (j == i) continue;
            double d = compute_distance(position[i], position[j]);
            if (d < nn_dist) nn_dist = d;
        }
        nn_sum += nn_dist;
    }
    double mean_nn = nn_sum / sample_count;
    /*
     * Normalise: use XBOUND / cbrt(N) as a reference "expected" inter-particle
     * spacing in a uniform distribution.
     */
    double ref_spacing = XBOUND / cbrt((double)N);
    /*
     * density_score close to 1 → sparse (fast anyway → can afford lower θ).
     * density_score close to 0 → very dense (expensive → raise θ to compensate).
     */
    double density_score = (mean_nn > ref_spacing) ? 1.0 : mean_nn / ref_spacing;

    /* ── Step 3: combine signals and update θ ── */
    double combined = 0.5 * density_score + 0.5 * balance_score;

    double delta;
    if (combined > 0.55) {
        /* We have headroom — reduce θ for more accuracy */
        delta = -THETA_STEP * (combined - 0.5) * 2.0;
    } else if (combined < 0.45) {
        /* We are overloaded — raise θ to go faster */
        delta = +THETA_STEP * (0.5 - combined) * 2.0;
    } else {
        delta = 0.0;   /* within tolerance band — leave θ unchanged */
    }

    local_theta += delta;
    if (local_theta < THETA_MIN) local_theta = THETA_MIN;
    if (local_theta > THETA_MAX) local_theta = THETA_MAX;
}

/* ── Barnes-Hut Tree ────────────────────────────────────────────── */
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

/* ── Force computation — uses local_theta ───────────────────────── */
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
    /* Use local_theta instead of the global constant THETA */
    if (local_theta > cell->width / d) {
        BH_compute_force_from_cell(cell, index);
    } else {
        for (int i = 0; i < cell->no_subcells; i++)
            BH_compute_force_from_octtree(cell->subcells[i], index);
    }
}

void BH_compute_force() {
    for (int i = 0; i < part_size; i++) {
        force[i].fx = force[i].fy = force[i].fz = 0.0;
        BH_compute_force_from_octtree(root_cell, i + pindex);
    }
}

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
    for (int i = 0; i < part_size; i++)
        velocity[i].vx = velocity[i].vy = velocity[i].vz = 0.0;
}

/* ── Simulation loop ────────────────────────────────────────────── */
void run_simulation() {
    if (rank == 0)
        printf("\n[Adaptive Theta] %d bodies, %d steps, initial theta=%.2f\n\n",
               N, TIME_STEPS, local_theta);

    MPI_Bcast(mass,      N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position,  N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Scatter(ivelocity, part_size, MPI_VELOCITY,
                velocity,  part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    for (int step = 0; step < TIME_STEPS; step++) {
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);

        /* ── Timed force compute (used by adapt_theta) ── */
        double t0 = MPI_Wtime();
        BH_compute_force();
        double force_time = MPI_Wtime() - t0;

        BH_delete_octtree(root_cell);

        compute_velocity();
        compute_positions();

        MPI_Allgather(position + pindex, part_size, MPI_POSITION,
                      position,          part_size, MPI_POSITION,
                      MPI_COMM_WORLD);

        /* ── Adapt θ for next iteration ── */
        adapt_theta(force_time);

        /* ── FIXED: Print theta periodically using MPI_Allgather ── */
        if (step % 100 == 0) {
            double all_thetas[size];
            MPI_Allgather(&local_theta, 1, MPI_DOUBLE,
                          all_thetas, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("  step %4d | thetas:", step);
                for (int r = 0; r < size; r++) printf(" %.3f", all_thetas[r]);
                printf("\n");
            }
        }
    }

    if (rank == 0) {
        FILE *f = fopen("pdist_adaptive.dat", "w");
        if (f) {
            for (int i = 0; i < N; i++)
                fprintf(f, "px=%f, py=%f, pz=%f\n",
                        position[i].px, position[i].py, position[i].pz);
            fclose(f);
            printf("[Adaptive Theta] Positions written to pdist_adaptive.dat\n");
        }
    }
}

/* ── Main ───────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N          = (argc >= 2) ? atoi(argv[1]) : DEFAULT_N;
    TIME_STEPS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TIME;

    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_POSITION); MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VELOCITY); MPI_Type_commit(&MPI_VELOCITY);

    part_size    = N / size;
    pindex       = rank * part_size;
    local_theta  = THETA_DEFAULT;

    /* Seed differently per rank for density sampling */
    srand(42 + rank);

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