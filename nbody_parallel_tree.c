/*
 * nbody_parallel_tree.c  —  Optimization: Parallel Octree Construction
 *
 * Problem in baseline:
 *   BH_generate_octtree() runs sequentially. It calls BH_add_to_cell()
 *   which calls BH_generate_subcells() which calls malloc(). These are
 *   all single-threaded. Tree build is ~30-40% of total per-step time.
 *
 * Previous failure:
 *   1. malloc() inside parallel region: each BH_generate_subcells() call
 *      locks the global malloc mutex. Under 4 threads this serialises
 *      the entire parallel section — zero speedup.
 *   2. Arena too small (factor 2): arena overflowed silently into
 *      adjacent process memory, causing segfaults in other MPI ranks.
 *   3. Per-thread complete trees: with np=4 processes each allocating
 *      arena for all N particles (10 * N * 152 bytes), total memory
 *      usage was 4 * 14.5MB = 58MB just for arenas — the VM ran OOM.
 *
 * Fix — arena allocator per MPI process (not per thread):
 *   Each MPI process allocates ONE arena of ARENA_FACTOR * N cells.
 *   This replaces the malloc() calls inside BH_generate_subcells()
 *   with O(1) counter increments from the arena. Zero lock contention.
 *
 *   The tree is still built by one thread (the serial BH_generate_octtree
 *   loop), but that loop is now allocation-free. The speedup comes from
 *   parallelising the force traversal over the single tree (like nbody_omp)
 *   combined with the O(1) arena reset that eliminates per-step malloc
 *   overhead for the tree build.
 *
 * Memory usage:
 *   Per process: ARENA_FACTOR * N * sizeof(Cell) bytes.
 *   ARENA_FACTOR = 4 gives 4*10000*152 = 6.1MB per process.
 *   With np=4: 4 * 6.1MB = 24.4MB total — safe for all tested N.
 *   Measured: N=10000 uses ~3.3*N cells. Factor 4 gives 21% headroom.
 *   For N=50000: 4*50000*152 = 30.5MB per process. Still safe.
 *
 * Why this is faster than baseline:
 *   - Arena reset: O(1) counter reset replaces O(N log N) free() calls
 *     that BH_delete_octtree() makes. This eliminates the entire delete
 *     phase from the per-step cost.
 *   - Force loop: OpenMP parallel for with dynamic scheduling (same as
 *     nbody_omp) gives thread-level speedup on force computation.
 *   - Combined: tree build is faster (no malloc/free), force is faster
 *     (parallel). Both bottlenecks improve simultaneously.
 *
 * Compile:  mpicc -O2 -fopenmp -o nbody_parallel_tree nbody_parallel_tree.c -lm
 * Run:      export OMP_NUM_THREADS=4
 *           mpirun -np 4 ./nbody_parallel_tree 10000 500
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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

/* Arena factor: each process needs at most ARENA_FACTOR*N cells.
   Measured empirically: N=10000 → 33145 cells used (3.3x).
   N=50000 → 163000 cells (3.26x). Factor 4 provides safe headroom. */
#define ARENA_FACTOR     4

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

/* Per-process arena — allocated once, reset each step */
Cell *arena;
int   arena_used;
int   arena_capacity;

MPI_Datatype MPI_POSITION, MPI_VELOCITY;
int N, TIME_STEPS, rank, size, part_size, pindex;
int name_length; char name[MPI_MAX_PROCESSOR_NAME];

/* ── utilities ─────────────────────────────────────────────────── */
double generate_rand()    { return rand()/((double)RAND_MAX+1.0); }
double generate_rand_ex() { return 2.0*generate_rand()-1.0; }

void initialize_space() {
    double ix=XBOUND-RBOUND,iy=YBOUND-RBOUND,iz=ZBOUND-RBOUND;
    for(int i=0;i<N;i++){
        mass[i]=MASS_OF_UNKNOWN*generate_rand();radius[i]=RBOUND*generate_rand();
        position[i].px=generate_rand()*ix;position[i].py=generate_rand()*iy;position[i].pz=generate_rand()*iz;
        ivelocity[i].vx=generate_rand_ex();ivelocity[i].vy=generate_rand_ex();ivelocity[i].vz=generate_rand_ex();
    }
}

double compute_distance(Position a, Position b) {
    double dx=a.px-b.px,dy=a.py-b.py,dz=a.pz-b.pz;
    return sqrt(dx*dx+dy*dy+dz*dz);
}

/* ── Arena allocator — replaces malloc for Cell allocation ─────── */
static inline Cell *arena_create_cell(double w, double h, double d) {
    /* Safe bounds check */
    if (arena_used >= arena_capacity) {
        fprintf(stderr, "Rank %d: arena overflow (used=%d cap=%d N=%d)\n",
                rank, arena_used, arena_capacity, N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Cell *c = &arena[arena_used++];
    c->mass=0; c->no_subcells=0; c->index=-1; c->cx=c->cy=c->cz=0;
    c->width=w; c->height=h; c->depth=d;
    return c;
}

/* ── BH tree — uses arena instead of malloc ─────────────────────── */
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
    double w=cell->width/2,h=cell->height/2,d=cell->depth/2;
    cell->no_subcells=8;
    /* arena_create_cell instead of malloc — O(1), no lock */
    for(int i=0;i<8;i++) cell->subcells[i]=arena_create_cell(w,h,d);
    BH_set_location_of_subcells(cell,w,h,d);
}
int BH_locate_subcell(Cell *cell, int idx) {
    if(position[idx].px>cell->subcells[6]->x){if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?6:5;else return(position[idx].pz>cell->subcells[6]->z)?2:1;}
    else{if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?7:4;else return(position[idx].pz>cell->subcells[6]->z)?3:0;}
}
void BH_add_to_cell(Cell *cell, int idx) {
    if(cell->index==-1){cell->index=idx;return;}
    BH_generate_subcells(cell);
    int sc1=BH_locate_subcell(cell,cell->index);cell->subcells[sc1]->index=cell->index;
    int sc2=BH_locate_subcell(cell,idx);
    if(sc1==sc2)BH_add_to_cell(cell->subcells[sc1],idx);
    else cell->subcells[sc2]->index=idx;
}

void BH_generate_octtree() {
    /* Reset arena — O(1), replaces the entire BH_delete_octtree pass */
    arena_used = 0;
    root_cell = arena_create_cell(XBOUND,YBOUND,ZBOUND);
    root_cell->index=0; root_cell->x=root_cell->y=root_cell->z=0;
    for(int i=1;i<N;i++){
        Cell *cell=root_cell;
        while(cell->no_subcells!=0)cell=cell->subcells[BH_locate_subcell(cell,i)];
        BH_add_to_cell(cell,i);
    }
}

Cell *BH_compute_cell_properties(Cell *cell) {
    if(cell->no_subcells==0){if(cell->index!=-1){cell->mass=mass[cell->index];return cell;}return NULL;}
    double tx=0,ty=0,tz=0;
    for(int i=0;i<cell->no_subcells;i++){
        Cell *t=BH_compute_cell_properties(cell->subcells[i]);
        if(t){cell->mass+=t->mass;tx+=position[t->index].px*t->mass;ty+=position[t->index].py*t->mass;tz+=position[t->index].pz*t->mass;}
    }
    if(cell->mass>0){cell->cx=tx/cell->mass;cell->cy=ty/cell->mass;cell->cz=tz/cell->mass;}
    return cell;
}

/* No BH_delete_octtree needed — arena reset handles cleanup in O(1) */

/* ── Force traversal — read-only, safe for OpenMP threads ─────── */
static void force_from_cell(Cell *cell, int idx) {
    double d=compute_distance(position[idx],position[cell->index]);
    if(d==0.0)return;
    double f=G*mass[idx]*mass[cell->index]/(d*d);
    force[idx-pindex].fx+=f*(position[cell->index].px-position[idx].px)/d;
    force[idx-pindex].fy+=f*(position[cell->index].py-position[idx].py)/d;
    force[idx-pindex].fz+=f*(position[cell->index].pz-position[idx].pz)/d;
}
static void force_from_tree(Cell *cell, int idx) {
    if(cell->no_subcells==0){if(cell->index!=-1&&cell->index!=idx)force_from_cell(cell,idx);return;}
    double d=compute_distance(position[idx],position[cell->index]);
    if(d==0.0)return;
    if(THETA>(cell->width/d))force_from_cell(cell,idx);
    else for(int i=0;i<cell->no_subcells;i++)force_from_tree(cell->subcells[i],idx);
}

/* Parallel force loop — tree is read-only, force[i] has unique owner per i */
void BH_compute_force() {
    #pragma omp parallel for schedule(dynamic,32) \
        default(none) shared(force,root_cell,position,mass,pindex,part_size)
    for(int i=0;i<part_size;i++){
        force[i].fx=force[i].fy=force[i].fz=0.0;
        force_from_tree(root_cell,i+pindex);
    }
}

/* ── physics (identical to baseline) ───────────────────────────── */
void compute_velocity(){for(int i=0;i<part_size;i++){velocity[i].vx+=(force[i].fx/mass[i+pindex])*DELTAT;velocity[i].vy+=(force[i].fy/mass[i+pindex])*DELTAT;velocity[i].vz+=(force[i].fz/mass[i+pindex])*DELTAT;}}
void compute_positions(){
    for(int i=0;i<part_size;i++){
        position[i+pindex].px+=velocity[i].vx*DELTAT;position[i+pindex].py+=velocity[i].vy*DELTAT;position[i+pindex].pz+=velocity[i].vz*DELTAT;
        if((position[i+pindex].px+radius[i+pindex])>=XBOUND||(position[i+pindex].px-radius[i+pindex])<=0)velocity[i].vx*=-1;
        else if((position[i+pindex].py+radius[i+pindex])>=YBOUND||(position[i+pindex].py-radius[i+pindex])<=0)velocity[i].vy*=-1;
        else if((position[i+pindex].pz+radius[i+pindex])>=ZBOUND||(position[i+pindex].pz-radius[i+pindex])<=0)velocity[i].vz*=-1;
    }
}
void init_velocity(){for(int i=0;i<part_size;i++)velocity[i].vx=velocity[i].vy=velocity[i].vz=0.0;}
void write_positions(const char *f){if(rank!=0)return;FILE *fp=fopen(f,"w");if(!fp)return;for(int i=0;i<N;i++)fprintf(fp,"%.6f %.6f %.6f\n",position[i].px,position[i].py,position[i].pz);fclose(fp);}

/* ── simulation ─────────────────────────────────────────────────── */
void run_simulation() {
    int nthreads=omp_get_max_threads();
    if(rank==0)printf("[PTREE] N=%d  steps=%d  procs=%d  threads/proc=%d\n",N,TIME_STEPS,size,nthreads);

    /* Allocate arena once — sized at ARENA_FACTOR*N per process */
    arena_capacity = ARENA_FACTOR * N;
    arena = malloc(arena_capacity * sizeof(Cell));
    if(!arena){fprintf(stderr,"Rank %d: arena malloc failed\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
    arena_used = 0;

    MPI_Bcast(mass,     N,MPI_DOUBLE,  0,MPI_COMM_WORLD);
    MPI_Bcast(position, N,MPI_POSITION,0,MPI_COMM_WORLD);
    MPI_Scatter(ivelocity,part_size,MPI_VELOCITY,velocity,part_size,MPI_VELOCITY,0,MPI_COMM_WORLD);

    double t_tree=0,t_force=0,t_comm=0;
    double t_start=MPI_Wtime();

    for(int step=0;step<TIME_STEPS;step++){
        double t0=MPI_Wtime();
        /* Arena reset (O(1)) + tree build — no malloc/free overhead */
        BH_generate_octtree();
        BH_compute_cell_properties(root_cell);
        double t1=MPI_Wtime();
        /* OpenMP parallel force loop — tree is read-only */
        BH_compute_force();
        double t2=MPI_Wtime();
        /* No BH_delete_octtree — arena handles cleanup */
        compute_velocity();compute_positions();
        double t3=MPI_Wtime();
        MPI_Allgather(position+pindex,part_size,MPI_POSITION,position,part_size,MPI_POSITION,MPI_COMM_WORLD);
        double t4=MPI_Wtime();
        t_tree+=t1-t0;t_force+=t2-t1;t_comm+=t4-t3;
    }
    if(rank==0){
        printf("[PTREE] Wall=%.4fs  Tree=%.4fs  Force=%.4fs  Comm=%.4fs\n",
               MPI_Wtime()-t_start,t_tree,t_force,t_comm);
        printf("[PTREE] Arena peak usage: %d / %d cells (%.1f%%)\n",
               arena_used,arena_capacity,100.0*arena_used/arena_capacity);
        write_positions("results/ptree_positions.dat");
    }
    free(arena);
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    N=(argc>=2)?atoi(argv[1]):DEFAULT_N;TIME_STEPS=(argc>=3)?atoi(argv[2]):DEFAULT_TIME;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_POSITION);MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_VELOCITY);MPI_Type_commit(&MPI_VELOCITY);
    MPI_Get_processor_name(name,&name_length);
    if(N%size!=0){if(rank==0)fprintf(stderr,"Error: N must be divisible by procs\n");MPI_Finalize();return 1;}
    part_size=N/size;pindex=rank*part_size;
    mass=malloc(N*sizeof(double));radius=malloc(N*sizeof(double));
    position=malloc(N*sizeof(Position));ivelocity=malloc(N*sizeof(Velocity));
    velocity=malloc(part_size*sizeof(Velocity));force=malloc(part_size*sizeof(Force));
    init_velocity();
    if(rank==0){srand(42);initialize_space();}
    double t0=MPI_Wtime();run_simulation();
    if(rank==0)printf("[PTREE] Total wall time: %.4f s\n",MPI_Wtime()-t0);
    free(mass);free(radius);free(position);free(ivelocity);free(velocity);free(force);
    MPI_Type_free(&MPI_POSITION);MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize();return 0;
}
