/*
 * nbody_theta.c  —  Optimization: Adaptive Theta
 *
 * Problem in baseline:
 *   THETA = 1.0 is a compile-time constant. Every rank uses the same
 *   opening angle everywhere. Dense regions (close-packed particles)
 *   need a smaller theta for accurate forces — the BH approximation
 *   is less valid when cells are close. Sparse regions can afford a
 *   larger theta, approximating more aggressively and running faster.
 *
 * Fix:
 *   Each rank computes its local particle density using its bounding box:
 *     local_density = part_size / (bbox_width * bbox_height * bbox_depth)
 *   One MPI_Allreduce gets the global average density across all ranks.
 *   Each rank then sets:
 *     local_theta = THETA_BASE / sqrt(local_density / avg_density)
 *   Dense rank (ratio > 1): local_theta < THETA_BASE → more accurate
 *   Sparse rank (ratio < 1): local_theta > THETA_BASE → faster
 *   Clamped to [THETA_MIN, THETA_MAX] for safety.
 *
 * Cost:
 *   One O(part_size) pass for bounding box + one MPI_Allreduce.
 *   Recomputed every RECOMPUTE_INTERVAL steps, not every step.
 *   Total overhead is negligible compared to force computation.
 *
 * Previous failure:
 *   Used O(N²) nearest-neighbor sampling to estimate density.
 *   Added thousands of sqrt() calls per step just for the estimate.
 *   Now replaced with O(N) bounding box — correct and fast.
 *
 * Compile:  mpicc -O2 -o nbody_theta nbody_theta.c -lm
 * Run:      mpirun -np 4 ./nbody_theta 10000 500
 */

#include <stdio.h>
#include <stdlib.h>
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
#define MASS_OF_UNKNOWN  1.899e12

#define THETA_BASE           1.0
#define THETA_MIN            0.4
#define THETA_MAX            1.4
#define RECOMPUTE_INTERVAL   20   /* steps between theta recomputes */

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
int    N, TIME_STEPS, rank, size, part_size, pindex;
double local_theta;   /* per-rank opening angle */
int    name_length; char name[MPI_MAX_PROCESSOR_NAME];

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

/* ── adaptive theta — O(part_size) + one collective ────────────── */
static void compute_adaptive_theta() {
    double xmn=1e30,xmx=-1e30,ymn=1e30,ymx=-1e30,zmn=1e30,zmx=-1e30;
    for(int i=0;i<part_size;i++){
        int id=i+pindex;
        if(position[id].px<xmn)xmn=position[id].px; if(position[id].px>xmx)xmx=position[id].px;
        if(position[id].py<ymn)ymn=position[id].py; if(position[id].py>ymx)ymx=position[id].py;
        if(position[id].pz<zmn)zmn=position[id].pz; if(position[id].pz>zmx)zmx=position[id].pz;
    }
    /* +1.0 prevents zero-volume when all particles are colinear */
    double vol=(xmx-xmn+1.0)*(ymx-ymn+1.0)*(zmx-zmn+1.0);
    double local_den=(double)part_size/vol;
    double sum=0.0;
    MPI_Allreduce(&local_den,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    double avg=sum/size;
    if(avg<1e-30){local_theta=THETA_BASE;return;}
    double ratio=local_den/avg;
    local_theta=THETA_BASE/sqrt(ratio);
    if(local_theta<THETA_MIN)local_theta=THETA_MIN;
    if(local_theta>THETA_MAX)local_theta=THETA_MAX;
}

/* ── BH tree (identical to baseline except uses local_theta) ────── */
Cell *BH_create_cell(double w, double h, double d){
    Cell *c=malloc(sizeof(Cell));
    c->mass=0;c->no_subcells=0;c->index=-1;c->cx=c->cy=c->cz=0;
    c->width=w;c->height=h;c->depth=d;return c;
}
void BH_set_location_of_subcells(Cell *cell,double w,double h,double d){
    cell->subcells[0]->x=cell->x;   cell->subcells[0]->y=cell->y;   cell->subcells[0]->z=cell->z;
    cell->subcells[1]->x=cell->x+w; cell->subcells[1]->y=cell->y;   cell->subcells[1]->z=cell->z;
    cell->subcells[2]->x=cell->x+w; cell->subcells[2]->y=cell->y;   cell->subcells[2]->z=cell->z+d;
    cell->subcells[3]->x=cell->x;   cell->subcells[3]->y=cell->y;   cell->subcells[3]->z=cell->z+d;
    cell->subcells[4]->x=cell->x;   cell->subcells[4]->y=cell->y+h; cell->subcells[4]->z=cell->z;
    cell->subcells[5]->x=cell->x+w; cell->subcells[5]->y=cell->y+h; cell->subcells[5]->z=cell->z;
    cell->subcells[6]->x=cell->x+w; cell->subcells[6]->y=cell->y+h; cell->subcells[6]->z=cell->z+d;
    cell->subcells[7]->x=cell->x;   cell->subcells[7]->y=cell->y+h; cell->subcells[7]->z=cell->z+d;
}
void BH_generate_subcells(Cell *cell){
    double w=cell->width/2,h=cell->height/2,d=cell->depth/2;cell->no_subcells=8;
    for(int i=0;i<8;i++)cell->subcells[i]=BH_create_cell(w,h,d);
    BH_set_location_of_subcells(cell,w,h,d);
}
int BH_locate_subcell(Cell *cell,int idx){
    if(position[idx].px>cell->subcells[6]->x){if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?6:5;else return(position[idx].pz>cell->subcells[6]->z)?2:1;}
    else{if(position[idx].py>cell->subcells[6]->y)return(position[idx].pz>cell->subcells[6]->z)?7:4;else return(position[idx].pz>cell->subcells[6]->z)?3:0;}
}
void BH_add_to_cell(Cell *cell,int idx){
    if(cell->index==-1){cell->index=idx;return;}BH_generate_subcells(cell);
    int sc1=BH_locate_subcell(cell,cell->index);cell->subcells[sc1]->index=cell->index;
    int sc2=BH_locate_subcell(cell,idx);
    if(sc1==sc2)BH_add_to_cell(cell->subcells[sc1],idx);else cell->subcells[sc2]->index=idx;
}
void BH_generate_octtree(){
    root_cell=BH_create_cell(XBOUND,YBOUND,ZBOUND);root_cell->index=0;root_cell->x=root_cell->y=root_cell->z=0;
    for(int i=1;i<N;i++){Cell *cell=root_cell;while(cell->no_subcells!=0)cell=cell->subcells[BH_locate_subcell(cell,i)];BH_add_to_cell(cell,i);}
}
Cell *BH_compute_cell_properties(Cell *cell){
    if(cell->no_subcells==0){if(cell->index!=-1){cell->mass=mass[cell->index];return cell;}return NULL;}
    double tx=0,ty=0,tz=0;
    for(int i=0;i<cell->no_subcells;i++){Cell *t=BH_compute_cell_properties(cell->subcells[i]);if(t){cell->mass+=t->mass;tx+=position[t->index].px*t->mass;ty+=position[t->index].py*t->mass;tz+=position[t->index].pz*t->mass;}}
    if(cell->mass>0){cell->cx=tx/cell->mass;cell->cy=ty/cell->mass;cell->cz=tz/cell->mass;}return cell;
}
void BH_compute_force_from_cell(Cell *cell,int idx){
    double d=compute_distance(position[idx],position[cell->index]);if(d==0.0)return;
    double f=G*mass[idx]*mass[cell->index]/(d*d);
    force[idx-pindex].fx+=f*(position[cell->index].px-position[idx].px)/d;
    force[idx-pindex].fy+=f*(position[cell->index].py-position[idx].py)/d;
    force[idx-pindex].fz+=f*(position[cell->index].pz-position[idx].pz)/d;
}
/* Uses local_theta (per-rank adaptive) instead of fixed THETA constant */
void BH_compute_force_from_octtree(Cell *cell,int idx){
    if(cell->no_subcells==0){if(cell->index!=-1&&cell->index!=idx)BH_compute_force_from_cell(cell,idx);}
    else{double d=compute_distance(position[idx],position[cell->index]);if(d==0.0)return;
        if(local_theta>(cell->width/d))BH_compute_force_from_cell(cell,idx); /* local_theta replaces THETA */
        else for(int i=0;i<cell->no_subcells;i++)BH_compute_force_from_octtree(cell->subcells[i],idx);}
}
void BH_compute_force(){
    for(int i=0;i<part_size;i++){force[i].fx=force[i].fy=force[i].fz=0.0;BH_compute_force_from_octtree(root_cell,i+pindex);}
}
void BH_delete_octtree(Cell *cell){
    if(cell->no_subcells!=0)for(int i=0;i<cell->no_subcells;i++)BH_delete_octtree(cell->subcells[i]);free(cell);
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
void run_simulation(){
    if(rank==0)printf("[THETA] N=%d  steps=%d  procs=%d  theta_base=%.2f  recompute_interval=%d\n",N,TIME_STEPS,size,THETA_BASE,RECOMPUTE_INTERVAL);

    MPI_Bcast(mass,     N,MPI_DOUBLE,  0,MPI_COMM_WORLD);
    MPI_Bcast(position, N,MPI_POSITION,0,MPI_COMM_WORLD);
    MPI_Scatter(ivelocity,part_size,MPI_VELOCITY,velocity,part_size,MPI_VELOCITY,0,MPI_COMM_WORLD);

    /* Initial theta */
    local_theta=THETA_BASE;
    compute_adaptive_theta();
    if(rank==0)printf("[THETA] Initial theta on rank 0: %.4f\n",local_theta);

    double t_tree=0,t_force=0,t_comm=0,t_theta=0;
    double t_start=MPI_Wtime();

    for(int step=0;step<TIME_STEPS;step++){
        /* Recompute theta every RECOMPUTE_INTERVAL steps */
        if(step>0 && step%RECOMPUTE_INTERVAL==0){
            double tt=MPI_Wtime();
            compute_adaptive_theta();
            t_theta+=MPI_Wtime()-tt;
        }
        double t0=MPI_Wtime();
        BH_generate_octtree();BH_compute_cell_properties(root_cell);
        double t1=MPI_Wtime();
        BH_compute_force();
        double t2=MPI_Wtime();
        BH_delete_octtree(root_cell);
        compute_velocity();compute_positions();
        double t3=MPI_Wtime();
        MPI_Allgather(position+pindex,part_size,MPI_POSITION,position,part_size,MPI_POSITION,MPI_COMM_WORLD);
        double t4=MPI_Wtime();
        t_tree+=t1-t0;t_force+=t2-t1;t_comm+=t4-t3;
    }
    if(rank==0){
        printf("[THETA] Wall=%.4fs  Tree=%.4fs  Force=%.4fs  Comm=%.4fs  Theta=%.4fs\n",
               MPI_Wtime()-t_start,t_tree,t_force,t_comm,t_theta);
        write_positions("results/theta_positions.dat");
    }
}

int main(int argc,char *argv[]){
    MPI_Init(&argc,&argv);
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
    if(rank==0)printf("[THETA] Total wall time: %.4f s\n",MPI_Wtime()-t0);
    free(mass);free(radius);free(position);free(ivelocity);free(velocity);free(force);
    MPI_Type_free(&MPI_POSITION);MPI_Type_free(&MPI_VELOCITY);
    MPI_Finalize();return 0;
}
