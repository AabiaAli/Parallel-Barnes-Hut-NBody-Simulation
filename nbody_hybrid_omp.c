/*
 * nbody_omp.c  —  Optimization: Hybrid MPI + OpenMP
 *
 * Problem in baseline:
 *   The force loop inside each MPI process is sequential. Each process
 *   uses exactly one CPU core even if the node has many. All other cores
 *   on that node sit idle during the most expensive operation.
 *
 * Fix:
 *   Parallelize BH_compute_force() with OpenMP. The outer loop over
 *   local particles is independent: each iteration i computes force[i]
 *   and never touches force[j] for j != i. The tree root_cell is
 *   read-only during force traversal (no writes happen during traversal,
 *   only during build/delete which remain single-threaded). So the
 *   parallel for is race-free without any locks.
 *
 * Critical fix vs previous broken version:
 *   The tree uses malloc() internally (in BH_generate_subcells). malloc()
 *   has a global lock. Calling it from parallel threads causes all threads
 *   to serialise on that lock, negating all speedup. The fix is simple:
 *   build the tree BEFORE the parallel region (single-threaded, safe),
 *   then parallelise only the force traversal which is read-only and
 *   therefore makes ZERO malloc calls. This is what this version does.
 *
 * Dynamic scheduling:
 *   BH tree traversal time per particle varies depending on local density.
 *   Particles in dense regions descend deeper and take longer. Static
 *   scheduling would leave some threads idle while others finish slow
 *   particles. Dynamic scheduling with chunk=32 ensures all threads stay
 *   busy until the loop completes.
 *
 * How to run:
 *   This version should be run with np*OMP_NUM_THREADS = total cores.
 *   Example: mpirun -np 4 ./nbody_omp 10000 500  (with OMP_NUM_THREADS=4)
 *   gives 4 ranks × 4 threads = 16 cores used simultaneously.
 *
 * Compile:  mpicc -O2 -fopenmp -o nbody_omp nbody_omp.c -lm
 * Run:      export OMP_NUM_THREADS=4
 *           mpirun -np 4 ./nbody_omp 10000 500
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
    double ix=XBOUND-RBOUND, iy=YBOUND-RBOUND, iz=ZBOUND-RBOUND;
    for(int i=0;i<N;i++){
        mass[i]=MASS_OF_UNKNOWN*generate_rand(); radius[i]=RBOUND*generate_rand();
        position[i].px=generate_rand()*ix; position[i].py=generate_rand()*iy; position[i].pz=generate_rand()*iz;
        ivelocity[i].vx=generate_rand_ex(); ivelocity[i].vy=generate_rand_ex(); ivelocity[i].vz=generate_rand_ex();
    }
}

double compute_distance(Position a, Position b) {
    double dx=a.px-b.px, dy=a.py-b.py, dz=a.pz-b.pz;
    return sqrt(dx*dx+dy*dy+dz*dz);
}

/* ── BH tree — built single-threaded, read-only during force ───── */
Cell *BH_create_cell(double w, double h, double d) {
    Cell *c=malloc(sizeof(Cell));
    c->mass=0;c->no_subcells=0;c->index=-1;c->cx=c->cy=c->cz=0;
    c->width=w;c->height=h;c->depth=d;return c;
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
    double w=cell->width/2,h=cell->height/2,d=cell->depth/2;
    cell->no_subcells=8;
    for(int i=0;i<8;i++)cell->subcells[i]=BH_create_cell(w,h,d);
    BH_set_location_of_subcells(cell,w,h,d);
}
int BH_locate_subcell(Cell *cell, int idx) {
    if(position[idx].px>cell->subcells[6]->x){
        if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?6:5;
        else return(position[idx].pz>cell->subcells[6]->z)?2:1;
    } else {
        if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?7:4;
        else return(position[idx].pz>cell->subcells[6]->z)?3:0;
    }
}
void BH_add_to_cell(Cell *cell, int idx) {
    if(cell->index==-1){cell->index=idx;return;}
    BH_generate_subcells(cell);
    int sc1=BH_locate_subcell(cell,cell->index); cell->subcells[sc1]->index=cell->index;
    int sc2=BH_locate_subcell(cell,idx);
    if(sc1==sc2)BH_add_to_cell(cell->subcells[sc1],idx);
    else cell->subcells[sc2]->index=idx;
}
void BH_generate_octtree() {
    root_cell=BH_create_cell(XBOUND,YBOUND,ZBOUND);
    root_cell->index=0;root_cell->x=root_cell->y=root_cell->z=0;
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
void BH_delete_octtree(Cell *cell) {
    if(cell->no_subcells!=0)for(int i=0;i<cell->no_subcells;i++)BH_delete_octtree(cell->subcells[i]);
    free(cell);
}

/* ── Force traversal — read-only on tree, safe for all threads ── */
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

/*
 * BH_compute_force — the only changed function vs baseline.
 *
 * Tree was built single-threaded BEFORE this call.
 * root_cell is read-only here — no writes, completely thread-safe.
 * force[i] is written by exactly one thread for each i — no race.
 * No malloc() calls anywhere in this parallel region.
 * Dynamic scheduling handles uneven traversal depth per particle.
 */
void BH_compute_force() {
    #pragma omp parallel for schedule(dynamic,32) \
        default(none) shared(force,root_cell,position,mass,pindex,part_size)
    for(int i=0; i<part_size; i++){
        force[i].fx=force[i].fy=force[i].fz=0.0;
        force_from_tree(root_cell, i+pindex);
    }
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
    if(rank!=0)return; FILE *fp=fopen(f,"w");if(!fp)return;
    for(int i=0;i<N;i++)fprintf(fp,"%.6f %.6f %.6f\n",position[i].px,position[i].py,position[i].pz);
    fclose(fp);
}

/* ── simulation ─────────────────────────────────────────────────── */
void run_simulation() {
    int nthreads=omp_get_max_threads();
    if(rank==0)printf("[OMP] N=%d  steps=%d  procs=%d  threads/proc=%d\n",N,TIME_STEPS,size,nthreads);

    MPI_Bcast(mass,     N,MPI_DOUBLE,  0,MPI_COMM_WORLD);
    MPI_Bcast(position, N,MPI_POSITION,0,MPI_COMM_WORLD);
    MPI_Scatter(ivelocity,part_size,MPI_VELOCITY,velocity,part_size,MPI_VELOCITY,0,MPI_COMM_WORLD);

    double t_tree=0,t_force=0,t_comm=0;
    double t_start=MPI_Wtime();

    for(int step=0;step<TIME_STEPS;step++){
        double t0=MPI_Wtime();
        BH_generate_octtree(); BH_compute_cell_properties(root_cell);
        double t1=MPI_Wtime();
        BH_compute_force();          /* <-- only change: OpenMP parallel */
        double t2=MPI_Wtime();
        BH_delete_octtree(root_cell);
        compute_velocity(); compute_positions();
        double t3=MPI_Wtime();
        MPI_Allgather(position+pindex,part_size,MPI_POSITION,
                      position,       part_size,MPI_POSITION,MPI_COMM_WORLD);
        double t4=MPI_Wtime();
        t_tree+=t1-t0; t_force+=t2-t1; t_comm+=t4-t3;
    }
    if(rank==0){
        printf("[OMP] Wall=%.4fs  Tree=%.4fs  Force=%.4fs  Comm=%.4fs\n",
               MPI_Wtime()-t_start,t_tree,t_force,t_comm);
        write_positions("results/omp_positions.dat");
    }
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    N=(argc>=2)?atoi(argv[1]):DEFAULT_N; TIME_STEPS=(argc>=3)?atoi(argv[2]):DEFAULT_TIME;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_POSITION);MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_VELOCITY);MPI_Type_commit(&MPI_VELOCITY);
    MPI_Get_processor_name(name,&name_length);
    if(N%size!=0){if(rank==0)fprintf(stderr,"Error: N must be divisible by procs\n");MPI_Finalize();return 1;}
    part_size=N/size; pindex=rank*part_size;
    mass=malloc(N*sizeof(double));radius=malloc(N*sizeof(double));
    position=malloc(N*sizeof(Position));ivelocity=malloc(N*sizeof(Velocity));
    velocity=malloc(part_size*sizeof(Velocity));force=malloc(part_size*sizeof(Force));
    init_velocity();
    if(rank==0){srand(42);initialize_space();}
    double t0=MPI_Wtime(); run_simulation();
    if(rank==0)printf("[OMP] Total wall time: %.4f s\n",MPI_Wtime()-t0);
    free(mass);free(radius);free(position);free(ivelocity);free(velocity);free(force);
    MPI_Type_free(&MPI_POSITION);MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize();return 0;
}
