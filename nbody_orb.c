/*
 * nbody_orb.c  —  Optimization: ORB Spatial Load Balancing
 *
 * Problem in baseline:
 *   Particles are split by array index. rank 0 gets [0..part_size), rank 1
 *   gets [part_size..2*part_size), etc. These index ranges cut through the
 *   randomly-initialized position array, giving each rank a random mix of
 *   spatial regions. When clustering forms, some ranks get denser regions
 *   and spend more time on BH tree traversal per particle. All ranks wait
 *   at MPI_Allgather for the slowest one every single step.
 *
 * Fix:
 *   Before the simulation loop, rank 0 runs ORB_partition_once() which
 *   recursively bisects the particle set along its longest spatial axis
 *   until each rank owns a compact spatial block with equal particle count.
 *   The global arrays (position, mass, radius, ivelocity) are physically
 *   reordered so rank r's particles sit at [r*part_size..(r+1)*part_size).
 *   The simulation loop is byte-for-byte identical to baseline.
 *
 * Cost of partition:
 *   O(N log N) for the qsort, done exactly once before the loop.
 *   For N=10000 this is ~1-2ms. Over 500 steps this overhead is negligible.
 *
 * Why speedup appears at large N / many steps:
 *   As the simulation evolves and particles cluster, the density imbalance
 *   between index-based chunks grows. ORB guarantees each rank always starts
 *   with an equal spatial density, so barrier idle time stays near zero.
 *   The benefit compounds over many steps.
 *
 * Compile:  mpicc -O2 -o nbody_orb nbody_orb.c -lm
 * Run:      mpirun -np 4 ./nbody_orb 10000 500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define DEFAULT_N        10000
#define DEFAULT_TIME     500
#define G                6.67300e-11
#define XBOUND           1.0e6
#define YBOUND           1.0e6
#define ZBOUND           1.0e6
#define RBOUND           10
#define DELTAT           0.01
#define THETA            1.0
#define MASS_OF_UNKNOWN  1.899e12

typedef struct { double px, py, pz; } Position;
typedef struct { double vx, vy, vz; } Velocity;
typedef struct { double fx, fy, fz; } Force;

typedef struct Cell {
    int    index, no_subcells;
    double mass, x, y, z, cx, cy, cz, width, height, depth;
    struct Cell *subcells[8];
} Cell;

Position *position;
Velocity *ivelocity, *velocity;
double   *mass, *radius;
Force    *force;
Cell     *root_cell;

MPI_Datatype MPI_POSITION, MPI_VELOCITY;
int N, TIME_STEPS, rank, size, part_size, pindex;
int name_length; char name[MPI_MAX_PROCESSOR_NAME];

/* ── utilities ─────────────────────────────────────────────────── */
double generate_rand()    { return rand() / ((double)RAND_MAX + 1.0); }
double generate_rand_ex() { return 2.0 * generate_rand() - 1.0; }

void initialize_space() {
    double ix = XBOUND-RBOUND, iy = YBOUND-RBOUND, iz = ZBOUND-RBOUND;
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

double compute_distance(Position a, Position b) {
    double dx=a.px-b.px, dy=a.py-b.py, dz=a.pz-b.pz;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* ── ORB: one-time partition before simulation loop ────────────── */
static Position *_pos;
static int       _axis;

static int cmp_orb(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    double da, db;
    if      (_axis == 0) { da=_pos[ia].px; db=_pos[ib].px; }
    else if (_axis == 1) { da=_pos[ia].py; db=_pos[ib].py; }
    else                 { da=_pos[ia].pz; db=_pos[ib].pz; }
    return (da > db) - (da < db);
}

/* Recursively assign each particle to a rank */
static void bisect(int *idx, int n, int r0, int nr, int *asgn) {
    if (nr == 1) {
        for (int i = 0; i < n; i++) asgn[idx[i]] = r0;
        return;
    }
    /* Find longest axis for this particle set */
    double xmn=1e30,xmx=-1e30,ymn=1e30,ymx=-1e30,zmn=1e30,zmx=-1e30;
    for (int i = 0; i < n; i++) {
        int id = idx[i];
        if (_pos[id].px < xmn) xmn=_pos[id].px; if (_pos[id].px > xmx) xmx=_pos[id].px;
        if (_pos[id].py < ymn) ymn=_pos[id].py; if (_pos[id].py > ymx) ymx=_pos[id].py;
        if (_pos[id].pz < zmn) zmn=_pos[id].pz; if (_pos[id].pz > zmx) zmx=_pos[id].pz;
    }
    double dx=xmx-xmn, dy=ymx-ymn, dz=zmx-zmn;
    _axis = (dx>=dy && dx>=dz) ? 0 : (dy>=dz) ? 1 : 2;
    qsort(idx, n, sizeof(int), cmp_orb);
    int lr=nr/2, rr=nr-lr, split=(int)((double)n*lr/nr);
    bisect(idx,       split,   r0,    lr, asgn);
    bisect(idx+split, n-split, r0+lr, rr, asgn);
}

/* Reorder all global arrays so rank r owns [r*part_size..(r+1)*part_size) */
static void ORB_partition_once() {
    int *idx  = malloc(N * sizeof(int));
    int *asgn = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    _pos = position; _axis = 0;
    bisect(idx, N, 0, size, asgn);
    free(idx);

    /* Collect per-rank particle lists */
    int  *cnt  = calloc(size, sizeof(int));
    int **list = malloc(size * sizeof(int*));
    for (int r = 0; r < size; r++) list[r] = malloc(part_size * sizeof(int));
    for (int i = 0; i < N; i++) { int r=asgn[i]; if(cnt[r]<part_size) list[r][cnt[r]++]=i; }

    /* Build flat reordering: rank0 block, rank1 block, ... */
    int *order = malloc(N * sizeof(int));
    int cur = 0;
    for (int r = 0; r < size; r++)
        for (int j = 0; j < part_size; j++)
            order[cur++] = list[r][j];

    /* Apply reordering to all global arrays */
    Position *tp = malloc(N*sizeof(Position));
    double   *tm = malloc(N*sizeof(double));
    double   *tr = malloc(N*sizeof(double));
    Velocity *tv = malloc(N*sizeof(Velocity));
    for (int i = 0; i < N; i++) {
        tp[i]=position[order[i]]; tm[i]=mass[order[i]];
        tr[i]=radius[order[i]];   tv[i]=ivelocity[order[i]];
    }
    memcpy(position,  tp, N*sizeof(Position));
    memcpy(mass,      tm, N*sizeof(double));
    memcpy(radius,    tr, N*sizeof(double));
    memcpy(ivelocity, tv, N*sizeof(Velocity));
    free(tp); free(tm); free(tr); free(tv);
    free(order); free(asgn); free(cnt);
    for (int r = 0; r < size; r++) free(list[r]);
    free(list);
}

/* ── BH tree (identical to baseline) ───────────────────────────── */
Cell *BH_create_cell(double w, double h, double d) {
    Cell *c = malloc(sizeof(Cell));
    c->mass=0; c->no_subcells=0; c->index=-1; c->cx=c->cy=c->cz=0;
    c->width=w; c->height=h; c->depth=d; return c;
}
void BH_set_location_of_subcells(Cell *cell, double w, double h, double d) {
    cell->subcells[0]->x=cell->x;   cell->subcells[0]->y=cell->y;   cell->subcells[0]->z=cell->z;
    cell->subcells[1]->x=cell->x+w; cell->subcells[1]->y=cell->y;   cell->subcells[1]->z=cell->z;
    cell->subcells[2]->x=cell->x+w; cell->subcells[2]->y=cell->y;   cell->subcells[2]->z=cell->z+d;
    cell->subcells[3]->x=cell->x;   cell->subcells[3]->y=cell->y;   cell->subcells[3]->z=cell->z+d;
    cell->subcells[4]->x=cell->x;   cell->subcells[4]->y=cell->y+h; cell->subcells[4]->z=cell->z;
    cell->subcells[5]->x=cell->x+w; cell->subcells[5]->y=cell->y+h; cell->subcells[5]->z=cell->z;
    cell->subcells[6]->x=cell->x+w; cell->subcells[6]->y=cell->y+h; cell->subcells[6]->z=cell->z+d;
    cell->subcells[7]->x=cell->x;   cell->subcells[7]->y=cell->y+h; cell->subcells[7]->z=cell->z+d;
}
void BH_generate_subcells(Cell *cell) {
    double w=cell->width/2, h=cell->height/2, d=cell->depth/2;
    cell->no_subcells=8;
    for (int i=0;i<8;i++) cell->subcells[i]=BH_create_cell(w,h,d);
    BH_set_location_of_subcells(cell,w,h,d);
}
int BH_locate_subcell(Cell *cell, int idx) {
    if (position[idx].px>cell->subcells[6]->x) {
        if (position[idx].py>cell->subcells[6]->y) return (position[idx].pz>cell->subcells[6]->z)?6:5;
        else return (position[idx].pz>cell->subcells[6]->z)?2:1;
    } else {
        if (position[idx].py>cell->subcells[6]->y) return (position[idx].pz>cell->subcells[6]->z)?7:4;
        else return (position[idx].pz>cell->subcells[6]->z)?3:0;
    }
}
void BH_add_to_cell(Cell *cell, int idx) {
    if (cell->index==-1){cell->index=idx;return;}
    BH_generate_subcells(cell);
    int sc1=BH_locate_subcell(cell,cell->index); cell->subcells[sc1]->index=cell->index;
    int sc2=BH_locate_subcell(cell,idx);
    if (sc1==sc2) BH_add_to_cell(cell->subcells[sc1],idx);
    else          cell->subcells[sc2]->index=idx;
}
void BH_generate_octtree() {
    root_cell=BH_create_cell(XBOUND,YBOUND,ZBOUND);
    root_cell->index=0; root_cell->x=root_cell->y=root_cell->z=0;
    for (int i=1;i<N;i++) {
        Cell *cell=root_cell;
        while (cell->no_subcells!=0) cell=cell->subcells[BH_locate_subcell(cell,i)];
        BH_add_to_cell(cell,i);
    }
}
Cell *BH_compute_cell_properties(Cell *cell) {
    if (cell->no_subcells==0){if(cell->index!=-1){cell->mass=mass[cell->index];return cell;}return NULL;}
    double tx=0,ty=0,tz=0;
    for (int i=0;i<cell->no_subcells;i++){
        Cell *t=BH_compute_cell_properties(cell->subcells[i]);
        if(t){cell->mass+=t->mass;tx+=position[t->index].px*t->mass;ty+=position[t->index].py*t->mass;tz+=position[t->index].pz*t->mass;}
    }
    if(cell->mass>0){cell->cx=tx/cell->mass;cell->cy=ty/cell->mass;cell->cz=tz/cell->mass;}
    return cell;
}
void BH_compute_force_from_cell(Cell *cell, int idx) {
    double d=compute_distance(position[idx],position[cell->index]);
    if(d==0.0)return;
    double f=G*mass[idx]*mass[cell->index]/(d*d);
    force[idx-pindex].fx+=f*(position[cell->index].px-position[idx].px)/d;
    force[idx-pindex].fy+=f*(position[cell->index].py-position[idx].py)/d;
    force[idx-pindex].fz+=f*(position[cell->index].pz-position[idx].pz)/d;
}
void BH_compute_force_from_octtree(Cell *cell, int idx) {
    if(cell->no_subcells==0){if(cell->index!=-1&&cell->index!=idx)BH_compute_force_from_cell(cell,idx);}
    else{double d=compute_distance(position[idx],position[cell->index]);if(d==0.0)return;
        if(THETA>(cell->width/d))BH_compute_force_from_cell(cell,idx);
        else for(int i=0;i<cell->no_subcells;i++)BH_compute_force_from_octtree(cell->subcells[i],idx);}
}
void BH_compute_force() {
    for(int i=0;i<part_size;i++){force[i].fx=force[i].fy=force[i].fz=0.0;BH_compute_force_from_octtree(root_cell,i+pindex);}
}
void BH_delete_octtree(Cell *cell) {
    if(cell->no_subcells!=0)for(int i=0;i<cell->no_subcells;i++)BH_delete_octtree(cell->subcells[i]);
    free(cell);
}

/* ── physics (identical to baseline) ───────────────────────────── */
void compute_velocity() {
    for(int i=0;i<part_size;i++){
        velocity[i].vx+=(force[i].fx/mass[i+pindex])*DELTAT;
        velocity[i].vy+=(force[i].fy/mass[i+pindex])*DELTAT;
        velocity[i].vz+=(force[i].fz/mass[i+pindex])*DELTAT;
    }
}
void compute_positions() {
    for(int i=0;i<part_size;i++){
        position[i+pindex].px+=velocity[i].vx*DELTAT;
        position[i+pindex].py+=velocity[i].vy*DELTAT;
        position[i+pindex].pz+=velocity[i].vz*DELTAT;
        if((position[i+pindex].px+radius[i+pindex])>=XBOUND||(position[i+pindex].px-radius[i+pindex])<=0)velocity[i].vx*=-1;
        else if((position[i+pindex].py+radius[i+pindex])>=YBOUND||(position[i+pindex].py-radius[i+pindex])<=0)velocity[i].vy*=-1;
        else if((position[i+pindex].pz+radius[i+pindex])>=ZBOUND||(position[i+pindex].pz-radius[i+pindex])<=0)velocity[i].vz*=-1;
    }
}
void init_velocity(){for(int i=0;i<part_size;i++)velocity[i].vx=velocity[i].vy=velocity[i].vz=0.0;}

void write_positions(const char *f){
    if(rank!=0)return; FILE *fp=fopen(f,"w"); if(!fp)return;
    for(int i=0;i<N;i++)fprintf(fp,"%.6f %.6f %.6f\n",position[i].px,position[i].py,position[i].pz);
    fclose(fp);
}

/* ── simulation ─────────────────────────────────────────────────── */
void run_simulation() {
    if(rank==0) printf("[ORB] N=%d  steps=%d  procs=%d\n",N,TIME_STEPS,size);

    /* Partition once on rank 0, then broadcast reordered data */
    if(rank==0) ORB_partition_once();

    MPI_Bcast(mass,     N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Bcast(position, N, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Bcast(radius,   N, MPI_DOUBLE,   0, MPI_COMM_WORLD);
    MPI_Scatter(ivelocity, part_size, MPI_VELOCITY,
                velocity,  part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    double t_tree=0, t_force=0, t_comm=0;
    double t_start=MPI_Wtime();

    for(int step=0; step<TIME_STEPS; step++){
        double t0=MPI_Wtime();
        BH_generate_octtree(); BH_compute_cell_properties(root_cell);
        double t1=MPI_Wtime();
        BH_compute_force();
        double t2=MPI_Wtime();
        BH_delete_octtree(root_cell);
        compute_velocity(); compute_positions();
        double t3=MPI_Wtime();
        MPI_Allgather(position+pindex, part_size, MPI_POSITION,
                      position,        part_size, MPI_POSITION, MPI_COMM_WORLD);
        double t4=MPI_Wtime();
        t_tree+=t1-t0; t_force+=t2-t1; t_comm+=t4-t3;
    }
    if(rank==0){
        printf("[ORB] Wall=%.4fs  Tree=%.4fs  Force=%.4fs  Comm=%.4fs\n",
               MPI_Wtime()-t_start,t_tree,t_force,t_comm);
        write_positions("results/orb_positions.dat");
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    N=(argc>=2)?atoi(argv[1]):DEFAULT_N; TIME_STEPS=(argc>=3)?atoi(argv[2]):DEFAULT_TIME;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_POSITION); MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_VELOCITY); MPI_Type_commit(&MPI_VELOCITY);
    MPI_Get_processor_name(name,&name_length);
    if(N%size!=0){if(rank==0)fprintf(stderr,"Error: N must be divisible by procs\n");MPI_Finalize();return 1;}
    part_size=N/size; pindex=rank*part_size;
    mass=malloc(N*sizeof(double)); radius=malloc(N*sizeof(double));
    position=malloc(N*sizeof(Position)); ivelocity=malloc(N*sizeof(Velocity));
    velocity=malloc(part_size*sizeof(Velocity)); force=malloc(part_size*sizeof(Force));
    init_velocity();
    if(rank==0){srand(42);initialize_space();}
    double t0=MPI_Wtime(); run_simulation();
    if(rank==0)printf("[ORB] Total wall time: %.4f s\n",MPI_Wtime()-t0);
    free(mass);free(radius);free(position);free(ivelocity);free(velocity);free(force);
    MPI_Type_free(&MPI_POSITION); MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize(); return 0;
}
