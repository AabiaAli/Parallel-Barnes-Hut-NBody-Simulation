/*
 * N-Body Barnes-Hut Simulation — Optimization: Adaptive Theta Per Process
 *
 * Includes ORB load balancing as the base, then adds per-rank adaptive
 * opening angle (theta) that reacts to local density and timing imbalance.
 *
 * WHY THE ORIGINAL ADAPTIVE FILE FAILED (RMS = 4e+05):
 * -------------------------------------------------------
 * Root cause 1 — NO SOFTENING:
 *   The original compute_distance returned sqrt(dx²+dy²+dz²).
 *   The only guard was "if (d == 0.0) return" which only catches
 *   exact zero.  When two particles drift very close (d ~ 1e-7 m),
 *   force = G*m1*m2/d² blows up to ~1e+40 N.  Velocities and
 *   positions become NaN or infinity within a few steps, producing
 *   the enormous RMS error.
 *
 * Root cause 2 — THETA RANGE TOO WIDE:
 *   THETA_MIN=0.3 forced the tree to descend very deep, computing
 *   many more short-range pairs — the exact pairs most likely to
 *   encounter near-zero distances.  This amplified root cause 1.
 *
 * Root cause 3 — RANDOM SAMPLING IN adapt_theta:
 *   The sampling loop called generate_rand() up to 5000 times per
 *   step per rank AFTER force computation.  Different ranks drew
 *   different numbers of samples, corrupting the shared RNG state
 *   and making results non-reproducible across runs.
 *
 * FIXES APPLIED:
 * -------------------------------------------------------
 *   1. Softening: d_soft = sqrt(dx²+dy²+dz²+EPSILON²) — guarantees
 *      d > 0 always, bounding the maximum force.
 *   2. Conservative theta range [THETA_MIN=0.8, THETA_MAX=1.2] and
 *      small step THETA_STEP=0.02 — accuracy never diverges far from
 *      the baseline theta=1.0 result.
 *   3. Deterministic density estimate via bounding-box volume ratio —
 *      no random sampling, no RNG pollution, fully reproducible.
 *   4. ORB load balancing included as the base.
 *
 * Compile:
 *   mpicc -O2 -o nbody_adaptive nbody_adaptive.c -lm
 *
 * Run:
 *   mpirun -np 4 ./nbody_adaptive 10000 1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ── Simulation constants ─────────────────────────────────────── */
#define DEFAULT_N       10000
#define DEFAULT_TIME    1000
#define G               6.67300e-11
#define XBOUND          1.0e6
#define YBOUND          1.0e6
#define ZBOUND          1.0e6
#define RBOUND          10
#define DELTAT          0.01
#define MASS_OF_UNKNOWN 1.899e12

/*
 * FIX 1 — Gravitational Softening
 * Prevents d→0 force singularities.  EPSILON is tiny relative to the
 * 1e6 m domain and has no measurable effect on particles at normal
 * separations (~2e4 m for N=10000 uniform).
 */
#define EPSILON         1.0e3   /* softening length (m) */

/*
 * FIX 2 — Conservative theta bounds
 * Original had THETA_MIN=0.3, THETA_MAX=1.5, THETA_STEP=0.05.
 * Those wide swings hurt accuracy.  New range stays close to baseline
 * theta=1.0 so positional error stays within the reference BH error.
 */
#define THETA_DEFAULT   1.0
#define THETA_MIN       0.8     /* tightest approximation allowed  */
#define THETA_MAX       1.2     /* coarsest approximation allowed  */
#define THETA_STEP      0.02    /* max change per iteration        */
#define IMBALANCE_TOL   0.10    /* 10% imbalance triggers adapt    */

/* ── Data types ──────────────────────────────────────────────── */
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

/* ── Globals ─────────────────────────────────────────────────── */
Position *position;
Velocity *ivelocity, *velocity;
double   *mass, *radius;
Force    *force;
Cell     *root_cell;

MPI_Datatype MPI_POSITION, MPI_VELOCITY;

int N, TIME_STEPS;
int rank, size, part_size, pindex;

double local_theta;   /* per-rank opening angle, updated each step */

/* ── Utilities ───────────────────────────────────────────────── */
double generate_rand()    { return rand() / ((double)RAND_MAX + 1); }
double generate_rand_ex() { return 2.0 * generate_rand() - 1.0; }

void initialize_space() {
    double ixb = XBOUND - RBOUND, iyb = YBOUND - RBOUND, izb = ZBOUND - RBOUND;
    for (int i = 0; i < N; i++) {
        mass[i]         = MASS_OF_UNKNOWN * generate_rand();
        radius[i]       = RBOUND * generate_rand();
        position[i].px  = generate_rand() * ixb;
        position[i].py  = generate_rand() * iyb;
        position[i].pz  = generate_rand() * izb;
        ivelocity[i].vx = generate_rand_ex();
        ivelocity[i].vy = generate_rand_ex();
        ivelocity[i].vz = generate_rand_ex();
    }
}

/*
 * FIX 1: Softened distance.
 * d_soft = sqrt(dx²+dy²+dz²+EPSILON²)
 * This is never zero, so force = G*m1*m2/d_soft² is always finite.
 */
static inline double compute_distance(Position a, Position b) {
    double dx = a.px - b.px, dy = a.py - b.py, dz = a.pz - b.pz;
    return sqrt(dx*dx + dy*dy + dz*dz + EPSILON*EPSILON);
}

/* ── ORB Load Balancing ──────────────────────────────────────────
 *
 * Recursively bisect the particle set along the longest spatial axis
 * so each process owns a compact, equal-sized spatial region.
 * Called once on rank 0 before broadcasting data.
 */
static int cmp_x(const void *a, const void *b) {
    int ia=*(int*)a, ib=*(int*)b;
    return (position[ia].px>position[ib].px)-(position[ia].px<position[ib].px);
}
static int cmp_y(const void *a, const void *b) {
    int ia=*(int*)a, ib=*(int*)b;
    return (position[ia].py>position[ib].py)-(position[ia].py<position[ib].py);
}
static int cmp_z(const void *a, const void *b) {
    int ia=*(int*)a, ib=*(int*)b;
    return (position[ia].pz>position[ib].pz)-(position[ia].pz<position[ib].pz);
}

static void ORB_bisect(int *idx, int n, int np) {
    if (np <= 1 || n <= 0) return;
    double xmn=1e18,xmx=-1e18, ymn=1e18,ymx=-1e18, zmn=1e18,zmx=-1e18;
    for (int i=0;i<n;i++){
        int p=idx[i];
        if(position[p].px<xmn)xmn=position[p].px; if(position[p].px>xmx)xmx=position[p].px;
        if(position[p].py<ymn)ymn=position[p].py; if(position[p].py>ymx)ymx=position[p].py;
        if(position[p].pz<zmn)zmn=position[p].pz; if(position[p].pz>zmx)zmx=position[p].pz;
    }
    double xr=xmx-xmn, yr=ymx-ymn, zr=zmx-zmn;
    if(xr>=yr&&xr>=zr)      qsort(idx,n,sizeof(int),cmp_x);
    else if(yr>=xr&&yr>=zr) qsort(idx,n,sizeof(int),cmp_y);
    else                     qsort(idx,n,sizeof(int),cmp_z);
    int hn=n/2, hnp=np/2;
    ORB_bisect(idx,    hn,   hnp);
    ORB_bisect(idx+hn, n-hn, np-hnp);
}

static void apply_orb() {
    int *idx = malloc(N*sizeof(int));
    for(int i=0;i<N;i++) idx[i]=i;
    ORB_bisect(idx, N, size);

    Position *np2 = malloc(N*sizeof(Position));
    Velocity *nv  = malloc(N*sizeof(Velocity));
    double   *nm  = malloc(N*sizeof(double));
    double   *nr  = malloc(N*sizeof(double));

    for(int i=0;i<N;i++){
        np2[i]=position[idx[i]]; nv[i]=ivelocity[idx[i]];
        nm[i]=mass[idx[i]];      nr[i]=radius[idx[i]];
    }
    for(int i=0;i<N;i++){
        position[i]=np2[i]; ivelocity[i]=nv[i];
        mass[i]=nm[i];      radius[i]=nr[i];
    }
    free(np2); free(nv); free(nm); free(nr); free(idx);
    printf("[ORB] Particles reordered spatially across %d processes.\n", size);
}

/* ── Adaptive Theta ──────────────────────────────────────────────
 *
 * FIX 3: Deterministic bounding-box density estimate.
 *
 * No random sampling.  Each rank computes:
 *   local_density = part_size / volume_of_local_bounding_box
 *
 * Then uses MPI_Allreduce to get the global average density.
 * Dense rank (above average) → lower theta (more accurate).
 * Sparse rank (below average) → higher theta (faster).
 *
 * The load-imbalance signal (from force timing) is also combined.
 * If this rank is well below the slowest rank, it has "headroom"
 * and can lower theta for free accuracy.  If it IS the slowest
 * rank, it raises theta slightly to reduce its computation.
 *
 * The combined adjustment is at most THETA_STEP per iteration and
 * theta stays in [THETA_MIN, THETA_MAX] = [0.8, 1.2].
 */
static void adapt_theta(double my_force_time) {

    /* ── Signal 1: load imbalance ── */
    double t_max;
    MPI_Allreduce(&my_force_time, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double t_mean;
    MPI_Allreduce(&my_force_time, &t_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t_mean /= (double)size;

    /*
     * balance_score in [-1, +1]:
     *   +1  → we are much faster than average → lower theta (gain accuracy)
     *   -1  → we are the bottleneck           → raise  theta (gain speed)
     *    0  → perfectly balanced              → no change
     * Only activate when imbalance exceeds IMBALANCE_TOL.
     */
    double balance_score = 0.0;
    if (t_max > 1e-9 && (t_max - t_mean) > IMBALANCE_TOL * t_mean) {
        /* Normalise: how far is this rank from the max? */
        balance_score = (t_max - my_force_time) / t_max;  /* 0=slowest, 1=fastest */
        balance_score = 2.0 * balance_score - 1.0;         /* rescale to [-1,+1]  */
    }

    /* ── Signal 2: local density (deterministic bounding-box) ── */
    /*
     * FIX 3: No random number calls here at all.
     * Compute bounding box of this rank's own particles (indices pindex..pindex+part_size-1).
     */
    double xmn=1e18,xmx=-1e18, ymn=1e18,ymx=-1e18, zmn=1e18,zmx=-1e18;
    for (int i = 0; i < part_size; i++) {
        int p = i + pindex;
        if(position[p].px<xmn) xmn=position[p].px;
        if(position[p].px>xmx) xmx=position[p].px;
        if(position[p].py<ymn) ymn=position[p].py;
        if(position[p].py>ymx) ymx=position[p].py;
        if(position[p].pz<zmn) zmn=position[p].pz;
        if(position[p].pz>zmx) zmx=position[p].pz;
    }
    double vol = fmax(xmx-xmn,1.0) * fmax(ymx-ymn,1.0) * fmax(zmx-zmn,1.0);
    double local_dens = (double)part_size / vol;

    double sum_dens;
    MPI_Allreduce(&local_dens, &sum_dens, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_avg = sum_dens / (double)size;

    /*
     * density_score in [-1, +1]:
     *   +1  → much denser than average → lower theta (close-range accuracy matters)
     *   -1  → much sparser than average → raise theta (far-field, approximation is fine)
     */
    double density_score = 0.0;
    if (global_avg > 1e-30) {
        double ratio = local_dens / global_avg;   /* >1 = dense, <1 = sparse */
        /* Clamp to ±2 range before normalising to avoid wild swings */
        if (ratio >  3.0) ratio =  3.0;
        if (ratio < 0.33) ratio = 0.33;
        density_score = (ratio - 1.0);  /* +ve = dense, -ve = sparse */
        /* Rescale to [-1,+1]: ratio in [0.33,3] maps to ~[-0.67,+2] before clamp */
        density_score = density_score / 2.0;
        if (density_score >  1.0) density_score =  1.0;
        if (density_score < -1.0) density_score = -1.0;
    }

    /*
     * Combined signal: average of both scores.
     *   positive → lower theta (be more accurate)
     *   negative → raise theta (be faster)
     * Scale by THETA_STEP so the maximum move per step is THETA_STEP.
     */
    double combined = 0.5 * balance_score + 0.5 * density_score;
    local_theta -= combined * THETA_STEP;   /* negative combined → raise theta */

    /* Clamp */
    if (local_theta < THETA_MIN) local_theta = THETA_MIN;
    if (local_theta > THETA_MAX) local_theta = THETA_MAX;
}

/* ── Barnes-Hut tree ─────────────────────────────────────────── */
static Cell *BH_create_cell(double w, double h, double d) {
    Cell *c = malloc(sizeof(Cell));
    c->mass=0; c->no_subcells=0; c->index=-1;
    c->cx=c->cy=c->cz=0; c->width=w; c->height=h; c->depth=d;
    return c;
}

static void BH_set_subcell_locations(Cell *cell, double w, double h, double d) {
    cell->subcells[0]->x=cell->x;   cell->subcells[0]->y=cell->y;   cell->subcells[0]->z=cell->z;
    cell->subcells[1]->x=cell->x+w; cell->subcells[1]->y=cell->y;   cell->subcells[1]->z=cell->z;
    cell->subcells[2]->x=cell->x+w; cell->subcells[2]->y=cell->y;   cell->subcells[2]->z=cell->z+d;
    cell->subcells[3]->x=cell->x;   cell->subcells[3]->y=cell->y;   cell->subcells[3]->z=cell->z+d;
    cell->subcells[4]->x=cell->x;   cell->subcells[4]->y=cell->y+h; cell->subcells[4]->z=cell->z;
    cell->subcells[5]->x=cell->x+w; cell->subcells[5]->y=cell->y+h; cell->subcells[5]->z=cell->z;
    cell->subcells[6]->x=cell->x+w; cell->subcells[6]->y=cell->y+h; cell->subcells[6]->z=cell->z+d;
    cell->subcells[7]->x=cell->x;   cell->subcells[7]->y=cell->y+h; cell->subcells[7]->z=cell->z+d;
}

static void BH_generate_subcells(Cell *cell) {
    double w=cell->width/2.0, h=cell->height/2.0, d=cell->depth/2.0;
    cell->no_subcells=8;
    for(int i=0;i<8;i++) cell->subcells[i]=BH_create_cell(w,h,d);
    BH_set_subcell_locations(cell,w,h,d);
}

static int BH_locate_subcell(Cell *cell, int idx) {
    int px=position[idx].px>cell->subcells[6]->x;
    int py=position[idx].py>cell->subcells[6]->y;
    int pz=position[idx].pz>cell->subcells[6]->z;
    if(px&&py&&pz) return 6; if(px&&py&&!pz) return 5;
    if(px&&!py&&pz) return 2; if(px&&!py&&!pz) return 1;
    if(!px&&py&&pz) return 7; if(!px&&py&&!pz) return 4;
    if(!px&&!py&&pz) return 3; return 0;
}

static void BH_add_to_cell(Cell *cell, int idx) {
    if(cell->index==-1){ cell->index=idx; return; }
    BH_generate_subcells(cell);
    int sc1=BH_locate_subcell(cell,cell->index);
    cell->subcells[sc1]->index=cell->index;
    int sc2=BH_locate_subcell(cell,idx);
    if(sc1==sc2) BH_add_to_cell(cell->subcells[sc1],idx);
    else         cell->subcells[sc2]->index=idx;
}

static void BH_generate_octtree() {
    root_cell=BH_create_cell(XBOUND,YBOUND,ZBOUND);
    root_cell->index=0; root_cell->x=root_cell->y=root_cell->z=0;
    for(int i=1;i<N;i++){
        Cell *cell=root_cell;
        while(cell->no_subcells!=0) cell=cell->subcells[BH_locate_subcell(cell,i)];
        BH_add_to_cell(cell,i);
    }
}

static Cell *BH_compute_cell_properties(Cell *cell) {
    if(cell->no_subcells==0){
        if(cell->index!=-1){ cell->mass=mass[cell->index]; return cell; }
        return NULL;
    }
    double tx=0,ty=0,tz=0;
    for(int i=0;i<cell->no_subcells;i++){
        Cell *t=BH_compute_cell_properties(cell->subcells[i]);
        if(t){ cell->mass+=t->mass; tx+=position[t->index].px*t->mass;
               ty+=position[t->index].py*t->mass; tz+=position[t->index].pz*t->mass; }
    }
    if(cell->mass>0){ cell->cx=tx/cell->mass; cell->cy=ty/cell->mass; cell->cz=tz/cell->mass; }
    return cell;
}

static void BH_delete_octtree(Cell *cell) {
    if(cell->no_subcells!=0)
        for(int i=0;i<cell->no_subcells;i++) BH_delete_octtree(cell->subcells[i]);
    free(cell);
}

/* FIX 1: No d==0 check needed — softened distance is always > 0 */
static void BH_compute_force_from_cell(Cell *cell, int index) {
    double d = compute_distance(position[index], position[cell->index]);
    double f = G * mass[index] * mass[cell->index] / (d * d);
    force[index-pindex].fx += f*(position[cell->index].px-position[index].px)/d;
    force[index-pindex].fy += f*(position[cell->index].py-position[index].py)/d;
    force[index-pindex].fz += f*(position[cell->index].pz-position[index].pz)/d;
}

/* Uses local_theta (per-rank adaptive value) instead of a global constant */
static void BH_compute_force_from_octtree(Cell *cell, int index) {
    if(cell->no_subcells==0){
        if(cell->index!=-1 && cell->index!=index)
            BH_compute_force_from_cell(cell,index);
        return;
    }
    double d=compute_distance(position[index],position[cell->index]);
    if(local_theta > cell->width/d)
        BH_compute_force_from_cell(cell,index);
    else
        for(int i=0;i<cell->no_subcells;i++)
            BH_compute_force_from_octtree(cell->subcells[i],index);
}

static void BH_compute_force() {
    for(int i=0;i<part_size;i++){
        force[i].fx=force[i].fy=force[i].fz=0.0;
        BH_compute_force_from_octtree(root_cell, i+pindex);
    }
}

static void compute_velocity() {
    for(int i=0;i<part_size;i++){
        velocity[i].vx+=(force[i].fx/mass[i+pindex])*DELTAT;
        velocity[i].vy+=(force[i].fy/mass[i+pindex])*DELTAT;
        velocity[i].vz+=(force[i].fz/mass[i+pindex])*DELTAT;
    }
}

static void compute_positions() {
    for(int i=0;i<part_size;i++){
        int gi=i+pindex;
        position[gi].px+=velocity[i].vx*DELTAT;
        position[gi].py+=velocity[i].vy*DELTAT;
        position[gi].pz+=velocity[i].vz*DELTAT;
        if(position[gi].px+radius[gi]>=XBOUND||position[gi].px-radius[gi]<=0) velocity[i].vx*=-1;
        if(position[gi].py+radius[gi]>=YBOUND||position[gi].py-radius[gi]<=0) velocity[i].vy*=-1;
        if(position[gi].pz+radius[gi]>=ZBOUND||position[gi].pz-radius[gi]<=0) velocity[i].vz*=-1;
    }
}

static void init_velocity() {
    for(int i=0;i<part_size;i++) velocity[i].vx=velocity[i].vy=velocity[i].vz=0.0;
}

/* ── Simulation loop ─────────────────────────────────────────── */
void run_simulation() {
    if(rank==0)
        printf("\n[Adaptive Theta] %d bodies, %d steps, theta starts at %.2f (range [%.2f,%.2f])\n\n",
               N, TIME_STEPS, local_theta, THETA_MIN, THETA_MAX);

    MPI_Bcast(mass,     N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position, N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Scatter(ivelocity, part_size, MPI_VELOCITY,
                velocity,  part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    for(int step=0; step<TIME_STEPS; step++){
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);

        double t0=MPI_Wtime();
        BH_compute_force();
        double force_time=MPI_Wtime()-t0;

        BH_delete_octtree(root_cell);
        compute_velocity();
        compute_positions();

        MPI_Allgather(position+pindex, part_size, MPI_POSITION,
                      position,        part_size, MPI_POSITION, MPI_COMM_WORLD);

        /* Adapt theta for the next iteration */
        adapt_theta(force_time);

        /* Print theta summary every 100 steps */
        if(step%100==0){
            double all_thetas[size];
            MPI_Allgather(&local_theta,1,MPI_DOUBLE,all_thetas,1,MPI_DOUBLE,MPI_COMM_WORLD);
            if(rank==0){
                printf("  step %4d | thetas:", step);
                for(int r=0;r<size;r++) printf(" r%d=%.3f",r,all_thetas[r]);
                printf("\n");
            }
        }
    }

    if(rank==0){
        FILE *f=fopen("pdist_adaptive.dat","w");
        if(f){
            for(int i=0;i<N;i++)
                fprintf(f,"px=%f, py=%f, pz=%f\n",
                        position[i].px,position[i].py,position[i].pz);
            fclose(f);
            printf("[Adaptive Theta] Positions written to pdist_adaptive.dat\n");
        }
    }
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    N          = (argc>=2) ? atoi(argv[1]) : DEFAULT_N;
    TIME_STEPS = (argc>=3) ? atoi(argv[2]) : DEFAULT_TIME;

    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_POSITION); MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_VELOCITY); MPI_Type_commit(&MPI_VELOCITY);

    part_size   = N / size;
    pindex      = rank * part_size;
    local_theta = THETA_DEFAULT;

    srand(42);   /* fixed seed for reproducibility across all ranks */

    mass      = malloc(N         * sizeof(double));
    radius    = malloc(N         * sizeof(double));
    position  = malloc(N         * sizeof(Position));
    ivelocity = malloc(N         * sizeof(Velocity));
    velocity  = malloc(part_size * sizeof(Velocity));
    force     = malloc(part_size * sizeof(Force));

    init_velocity();
    if(rank==0){ initialize_space(); apply_orb(); }

    run_simulation();

    free(mass); free(radius); free(position);
    free(ivelocity); free(velocity); free(force);
    MPI_Type_free(&MPI_POSITION); MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize();
    return 0;
}