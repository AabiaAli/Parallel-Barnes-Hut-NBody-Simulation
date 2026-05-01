/*
 * N-Body Barnes-Hut Simulation — Optimization 1: ORB (Orthogonal Recursive Bisection)
 *
 * Replaces static block partitioning (rank * part_size) with spatial domain
 * decomposition using ORB, so each MPI process owns a geometrically balanced
 * region of space rather than a fixed contiguous slice of the particle array.
 *
 * Key changes vs. original:
 *   - orb_partition() builds a binary tree of axis-aligned cuts over all particles.
 *   - Each leaf of the tree is assigned to one MPI rank.
 *   - After every time-step the partition is rebuilt to follow particle movement.
 *   - Communication uses MPI_Allgatherv because partition sizes can differ.
 *
 * Author: adapted from Dileban Karunamoorthy (dileban@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ── Constants ─────────────────────────────────────────────────── */
#define DEFAULT_N      10000
#define DEFAULT_TIME   1000
#define G              6.67300e-11
#define XBOUND         1.0e6
#define YBOUND         1.0e6
#define ZBOUND         1.0e6
#define RBOUND         10
#define DELTAT         0.01
#define THETA          1.0
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
Velocity *velocity;   /* local particles only */
double   *mass;
double   *radius;
Force    *force;      /* local particles only */
Cell     *root_cell;

MPI_Datatype MPI_POSITION;
MPI_Datatype MPI_VELOCITY;

int N, TIME_STEPS;
int rank, size;

/* ORB partition result: each rank's local particle count and index list */
int  local_n;           /* particles owned by this rank */
int *local_indices;     /* global indices of owned particles  */
int *all_local_n;       /* [size]  how many particles per rank */
int *displs;            /* [size]  displacement for Allgatherv */

/* ── Utilities ──────────────────────────────────────────────────── */
double generate_rand()    { return rand() / ((double)RAND_MAX + 1); }
double generate_rand_ex() { return 2.0 * generate_rand() - 1.0; }

void initialize_space() {
    double ixbound = XBOUND - RBOUND;
    double iybound = YBOUND - RBOUND;
    double izbound = ZBOUND - RBOUND;
    for (int i = 0; i < N; i++) {
        mass[i]          = MASS_OF_UNKNOWN * generate_rand();
        radius[i]        = RBOUND * generate_rand();
        position[i].px   = generate_rand() * ixbound;
        position[i].py   = generate_rand() * iybound;
        position[i].pz   = generate_rand() * izbound;
        ivelocity[i].vx  = generate_rand_ex();
        ivelocity[i].vy  = generate_rand_ex();
        ivelocity[i].vz  = generate_rand_ex();
    }
}

double compute_distance(Position a, Position b) {
    return sqrt(pow(a.px - b.px, 2.0) +
                pow(a.py - b.py, 2.0) +
                pow(a.pz - b.pz, 2.0));
}

/* ── ORB Partition ──────────────────────────────────────────────── */
/*
 * Comparators for qsort along each axis.
 * We pass the axis via a global rather than a closure to stay C89-compatible.
 */
static int orb_axis; /* 0=x, 1=y, 2=z */

static int cmp_axis(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double da, db;
    if      (orb_axis == 0) { da = position[ia].px; db = position[ib].px; }
    else if (orb_axis == 1) { da = position[ia].py; db = position[ib].py; }
    else                    { da = position[ia].pz; db = position[ib].pz; }
    return (da > db) - (da < db);
}

/*
 * Recursive ORB: split idx[0..n-1] into `num_parts` balanced subsets.
 * On exit, for the leaf assigned to `target_rank`, fills
 * local_indices[0..local_n-1].
 */
static void orb_recursive(int *idx, int n, int num_parts,
                           int rank_offset, int target_rank,
                           int depth)
{
    if (num_parts == 1) {
        /* This leaf belongs to rank_offset; check if it's ours */
        if (rank_offset == target_rank) {
            local_n = n;
            memcpy(local_indices, idx, n * sizeof(int));
        }
        return;
    }

    /* Choose longest bounding-box axis */
    double xmin = 1e300, xmax = -1e300;
    double ymin = 1e300, ymax = -1e300;
    double zmin = 1e300, zmax = -1e300;
    for (int i = 0; i < n; i++) {
        int id = idx[i];
        if (position[id].px < xmin) xmin = position[id].px;
        if (position[id].px > xmax) xmax = position[id].px;
        if (position[id].py < ymin) ymin = position[id].py;
        if (position[id].py > ymax) ymax = position[id].py;
        if (position[id].pz < zmin) zmin = position[id].pz;
        if (position[id].pz > zmax) zmax = position[id].pz;
    }
    double rx = xmax - xmin;
    double ry = ymax - ymin;
    double rz = zmax - zmin;
    if      (rx >= ry && rx >= rz) orb_axis = 0;
    else if (ry >= rx && ry >= rz) orb_axis = 1;
    else                           orb_axis = 2;

    /* Sort and split at median */
    qsort(idx, n, sizeof(int), cmp_axis);

    int left_parts  = num_parts / 2;
    int right_parts = num_parts - left_parts;
    int split       = (int)((double)n * left_parts / num_parts);

    orb_recursive(idx,         split,     left_parts,  rank_offset,              target_rank, depth + 1);
    orb_recursive(idx + split, n - split, right_parts, rank_offset + left_parts, target_rank, depth + 1);
}

/*
 * Build ORB partition over all N particles; determine which particles
 * belong to this rank. Then share counts with every rank via Allgather.
 */
void orb_partition() {
    int *all_idx = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) all_idx[i] = i;

    local_n = 0;
    orb_recursive(all_idx, N, size, 0, rank, 0);
    free(all_idx);

    /* Share per-rank counts */
    MPI_Allgather(&local_n, 1, MPI_INT, all_local_n, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int r = 1; r < size; r++)
        displs[r] = displs[r - 1] + all_local_n[r - 1];

    /* Resize local arrays to fit new partition */
    free(velocity); free(force);
    velocity = (Velocity *)malloc(local_n * sizeof(Velocity));
    force    = (Force    *)malloc(local_n * sizeof(Force   ));
    for (int i = 0; i < local_n; i++) {
        int gi        = local_indices[i];
        velocity[i]   = ivelocity[gi];  /* copy initial velocity on first call */
    }
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
    cell->subcells[0]->x = cell->x;           cell->subcells[0]->y = cell->y;           cell->subcells[0]->z = cell->z;
    cell->subcells[1]->x = cell->x + w;       cell->subcells[1]->y = cell->y;           cell->subcells[1]->z = cell->z;
    cell->subcells[2]->x = cell->x + w;       cell->subcells[2]->y = cell->y;           cell->subcells[2]->z = cell->z + d;
    cell->subcells[3]->x = cell->x;           cell->subcells[3]->y = cell->y;           cell->subcells[3]->z = cell->z + d;
    cell->subcells[4]->x = cell->x;           cell->subcells[4]->y = cell->y + h;       cell->subcells[4]->z = cell->z;
    cell->subcells[5]->x = cell->x + w;       cell->subcells[5]->y = cell->y + h;       cell->subcells[5]->z = cell->z;
    cell->subcells[6]->x = cell->x + w;       cell->subcells[6]->y = cell->y + h;       cell->subcells[6]->z = cell->z + d;
    cell->subcells[7]->x = cell->x;           cell->subcells[7]->y = cell->y + h;       cell->subcells[7]->z = cell->z + d;
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

/* ── Force / Velocity / Position ───────────────────────────────── */
void BH_compute_force_from_cell(Cell *cell, int local_i, int global_i) {
    if (cell->index == global_i) return;
    double d = compute_distance(position[global_i], position[cell->index]);
    if (d == 0.0) return;
    double f = G * mass[global_i] * mass[cell->index] / (d * d);
    force[local_i].fx += f * (position[cell->index].px - position[global_i].px) / d;
    force[local_i].fy += f * (position[cell->index].py - position[global_i].py) / d;
    force[local_i].fz += f * (position[cell->index].pz - position[global_i].pz) / d;
}

void BH_compute_force_from_octtree(Cell *cell, int local_i, int global_i) {
    if (cell->no_subcells == 0) {
        if (cell->index != -1 && cell->index != global_i)
            BH_compute_force_from_cell(cell, local_i, global_i);
        return;
    }
    double d = compute_distance(position[global_i], position[cell->index]);
    if (d == 0.0) return;
    if (THETA > cell->width / d) {
        BH_compute_force_from_cell(cell, local_i, global_i);
    } else {
        for (int i = 0; i < cell->no_subcells; i++)
            BH_compute_force_from_octtree(cell->subcells[i], local_i, global_i);
    }
}

void BH_compute_force() {
    for (int i = 0; i < local_n; i++) {
        force[i].fx = force[i].fy = force[i].fz = 0.0;
        BH_compute_force_from_octtree(root_cell, i, local_indices[i]);
    }
}

void compute_velocity() {
    for (int i = 0; i < local_n; i++) {
        int gi = local_indices[i];
        velocity[i].vx += (force[i].fx / mass[gi]) * DELTAT;
        velocity[i].vy += (force[i].fy / mass[gi]) * DELTAT;
        velocity[i].vz += (force[i].fz / mass[gi]) * DELTAT;
    }
}

void compute_positions() {
    for (int i = 0; i < local_n; i++) {
        int gi = local_indices[i];
        position[gi].px += velocity[i].vx * DELTAT;
        position[gi].py += velocity[i].vy * DELTAT;
        position[gi].pz += velocity[i].vz * DELTAT;

        if (position[gi].px + radius[gi] >= XBOUND || position[gi].px - radius[gi] <= 0) velocity[i].vx *= -1;
        if (position[gi].py + radius[gi] >= YBOUND || position[gi].py - radius[gi] <= 0) velocity[i].vy *= -1;
        if (position[gi].pz + radius[gi] >= ZBOUND || position[gi].pz - radius[gi] <= 0) velocity[i].vz *= -1;
    }
}

/* ── Gather all positions using variable-length Allgatherv ──────── */
/*
 * We need a contiguous send buffer because Allgatherv requires
 * the send buffer to hold exactly sendcount elements.
 * Build a local position array from local_indices, then gather.
 */
void gather_positions() {
    Position *send_buf = (Position *)malloc(local_n * sizeof(Position));
    for (int i = 0; i < local_n; i++)
        send_buf[i] = position[local_indices[i]];

    /* Receive displacements are in units of MPI_POSITION (3 doubles) */
    MPI_Allgatherv(send_buf, local_n, MPI_POSITION,
                   position, all_local_n, displs, MPI_POSITION,
                   MPI_COMM_WORLD);

    free(send_buf);

    /*
     * After gathering, positions are packed in rank order; we need to
     * remap them back into the global index array so the tree is consistent.
     * Rebuild a temporary index list and scatter into position[].
     */
    Position *tmp = (Position *)malloc(N * sizeof(Position));
    memcpy(tmp, position, N * sizeof(Position));

    /* Re-scatter: for each rank r, its displs[r] elements go to local_indices */
    /* We already know our own local_indices; for others we need to reconstruct */
    /* The simplest approach: re-run orb_recursive for all ranks. */
    /* To avoid that cost, we broadcast local_indices from every rank.          */
    int *all_indices = (int *)malloc(N * sizeof(int));
    MPI_Allgatherv(local_indices, local_n, MPI_INT,
                   all_indices, all_local_n, displs, MPI_INT,
                   MPI_COMM_WORLD);

    for (int i = 0; i < N; i++)
        position[all_indices[i]] = tmp[i];

    free(tmp);
    free(all_indices);
}

/* ── Simulation ─────────────────────────────────────────────────── */
void run_simulation() {
    if (rank == 0)
        printf("\n[ORB] Running %d bodies for %d steps, DELTAT=%.4f\n\n",
               N, TIME_STEPS, DELTAT);

    MPI_Bcast(mass,     N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position, N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Bcast(ivelocity,N, MPI_VELOCITY, 0, MPI_COMM_WORLD);
    MPI_Bcast(radius,   N, MPI_DOUBLE,   0, MPI_COMM_WORLD);

    /* Allocate ORB helper arrays */
    local_indices = (int *)malloc(N * sizeof(int));
    all_local_n   = (int *)malloc(size * sizeof(int));
    displs        = (int *)malloc(size * sizeof(int));

    /* Initial ORB partition — also allocates velocity/force */
    orb_partition();

    for (int step = 0; step < TIME_STEPS; step++) {
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);
        BH_compute_force();
        BH_delete_octtree(root_cell);

        compute_velocity();
        compute_positions();

        /* Gather updated positions to all ranks */
        gather_positions();

        /* Re-partition every step to maintain load balance */
        /* (preserve velocities across repartition via ivelocity scratch) */
        for (int i = 0; i < local_n; i++) ivelocity[local_indices[i]] = velocity[i];
        MPI_Allgatherv(ivelocity + displs[rank], local_n, MPI_VELOCITY,
                       ivelocity, all_local_n, displs, MPI_VELOCITY,
                       MPI_COMM_WORLD);
        orb_partition();
        /* Restore velocities for new local set */
        for (int i = 0; i < local_n; i++) velocity[i] = ivelocity[local_indices[i]];
    }

    if (rank == 0) {
        FILE *f = fopen("pdist_orb.dat", "w");
        if (f) {
            for (int i = 0; i < N; i++)
                fprintf(f, "px=%f, py=%f, pz=%f\n",
                        position[i].px, position[i].py, position[i].pz);
            fclose(f);
            printf("[ORB] Positions written to pdist_orb.dat\n");
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

    mass      = (double   *)malloc(N * sizeof(double));
    radius    = (double   *)malloc(N * sizeof(double));
    position  = (Position *)malloc(N * sizeof(Position));
    ivelocity = (Velocity *)malloc(N * sizeof(Velocity));
    /* velocity and force allocated inside orb_partition() */
    velocity  = NULL;
    force     = NULL;

    if (rank == 0) initialize_space();

    run_simulation();

    MPI_Finalize();
    return 0;
}
