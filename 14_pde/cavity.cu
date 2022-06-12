#include <cstdio>
#include <sys/time.h>

__device__ const int nx = 41, ny = 41, nt = 500, nit=50;
__device__ const double rho = 1.0, nu = 0.02;
__device__ const double dx = 2/(double)(nx - 1), dy = 2/(double)(ny - 1)
__device__ const double dt = 0.01;

__global__ void initialize(double *array, double value) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  array[i] = value;
}

__global__ void cal_b(double *u, double *v, double *b) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double dudx = (u[y*nx+(x+1)] - u[y*nx+(x-1)])/(2*dx);
  double dudy = (u[(y+1)*nx+x] - u[(y-1)*nx+x])/(2*dy);
  double dvdx = (v[y*nx+(x+1)] - v[y*nx+(x-1)])/(2*dx);
  double dvdy = (v[(y+1)*nx+x] - v[(y-1)*nx+x])/(2*dy);
  b[i] = rho*((dudx + dvdy)/dt - (pow(dudx,2) + 2*dudy*dvdx + pow(dvdy,2)));
}

__global__ void cal_p(double *p, double *pn, double *b) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  p[i] = ((pn[y*nx+(x+1)] + pn[y*nx+(x-1)])*(dy*dy) + (pn[(y+1)*nx+x] + pn[(y-1)*nx+x])*(dx*dx)
          - b[i]*(dx*dx)*(dy*dy))/(2*((dx*dx) + (dy*dy)));
}

__global__ void cal_p_edge(double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0) p[i] = p[y*nx+(x+1)];
  else if (y==0) p[i] = p[(y+1)*nx+x];
  else if (x==(nx-1)) p[i] = p[y*nx+(x-1)];
  else if (y==(ny-1)) p[i] = 0.0;
}

__global__ void cal_u(double *u, double *un, double *vn, double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double du = 0.0;
  du = - un[i]*dt/dx*(un[i] - un[y*nx+(x-1)]) - vn[i]*dt/dy*(un[i] - un[(y-1)*nx+x]) - (dt/dx/(2*rho))*(p[y*nx+(x+1)]
       - p[y*nx+(x-1)]) + nu*dt/(dx*dx)*(un[y*nx+(x+1)] - 2*un[i] + un[y*nx+(x-1)]) 
       + nu*dt/(dy*dy)*(un[(y+1)*nx+x] - 2*un[i] + un[(y-1)*nx+x]);
  u[i] = un[i] + du;
}

__global__ void cal_v(double *v, double *un, double *vn, double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double dv = 0.0;
  dv = - un[i]*dt/dx*(vn[i] - vn[y*nx+(x-1)]) - vn[i]*dt/dy*(vn[i] - vn[(y-1)*nx+x]) - (dt/dx/(2*rho))*(p[(y+1)*nx+x] - p[(y-1)*nx+x])
       + nu*dt/(dx*dx)*(vn[y*nx+(x+1)] - 2*vn[i] + vn[y*nx+(x-1)]) + nu*dt/(dy*dy)*(vn[(y+1)*nx+x] - 2*vn[i] + vn[(y-1)*nx+x]);
  v[i] = vn[i] + dv;
}

__global__ void cal_u_edge(double *u) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  int x = i%nx, y = i/nx;
  if (y==(ny-1)) u[i] = 1.0;
  else if (x==0 || y==0 || x==(nx-1)) u[i] = 0.0;
}

__global__ void cal_v_edge(double *v) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  int x = i%nx, y = i/nx;
  if (x==0 || y==0 || x==(nx-1) || y==(ny-1)) v[i] = 0.0;
}

long eval_time_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int main() {
  int NUM_BLOCK = 256;
  int NUM_THREAD = (nx*ny+NUM_BLOCK-1)/NUM_BLOCK;

  struct timeval st;
  struct timeval et;
  long us;

  double *du,*dv,*db,*dp,*dpn,*dun,*dvn;
  cudaMallocManaged(&du, nx*ny*sizeof(double));
  cudaMallocManaged(&dv, nx*ny*sizeof(double));
  cudaMallocManaged(&dp, nx*ny*sizeof(double));
  cudaMallocManaged(&db, nx*ny*sizeof(double));
  cudaMallocManaged(&dpn, nx*ny*sizeof(double));
  cudaMallocManaged(&dun, nx*ny*sizeof(double));
  cudaMallocManaged(&dvn, nx*ny*sizeof(double));

  gettimeofday(&st, NULL);
  initialize<<<NUM_BLOCK,NUM_THREAD>>>(du, 0.0);
  initialize<<<NUM_BLOCK,NUM_THREAD>>>(dv, 0.0);
  initialize<<<NUM_BLOCK,NUM_THREAD>>>(dp, 0.0);
  initialize<<<NUM_BLOCK,NUM_THREAD>>>(db, 0.0);
  cudaDeviceSynchronize();
  for (int n=0; n<nt; n++) {
    cal_b<<<NUM_BLOCK,NUM_THREAD>>>(du, dv, db);
    cudaDeviceSynchronize();
    for (int it=0; it<nit; it++) {
      cudaMemcpy(dpn, dp, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
      cal_p<<<NUM_BLOCK,NUM_THREAD>>>(dp, dpn, db);
      cudaDeviceSynchronize();
      cal_p_edge<<<NUM_BLOCK,NUM_THREAD>>>(dp);
      cudaDeviceSynchronize();
    }
    cudaMemcpy(dun, du, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dvn, dv, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
    cal_u<<<NUM_BLOCK,NUM_THREAD>>>(du, dun, dvn, dp);
    cal_v<<<NUM_BLOCK,NUM_THREAD>>>(dv, dun, dvn, dp);
    cudaDeviceSynchronize();
    cal_u_edge<<<NUM_BLOCK,NUM_THREAD>>>(du);
    cal_v_edge<<<NUM_BLOCK,NUM_THREAD>>>(dv);
    cudaDeviceSynchronize();
  }

  cudaFree(du);
  cudaFree(dv);
  cudaFree(dp);
  cudaFree(db);

  gettimeofday(&et, NULL);

  us = eval_time_us(st, et);
  printf("time : %ld us\n",us);
}